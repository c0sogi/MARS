import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.dataset import get_dataloader
from library.model import BraTS2DCNN


def set_seed(seed=Config.SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch using Mixed Precision (AMP).
    """
    model.train()
    running_loss = 0.0

    # Initialize Scaler for AMP
    scaler = torch.cuda.amp.GradScaler()

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).view(-1, 1)  # Reshape to (Batch, 1)

        optimizer.zero_grad()

        # Use autocast for mixed precision
        with torch.cuda.amp.autocast():
            logits = model(inputs)
            loss = criterion(logits, targets)

        # Scale loss and backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1, 1)

            # Use autocast for inference to save memory
            with torch.cuda.amp.autocast():
                logits = model(inputs)
                loss = criterion(logits, targets)

            running_loss += loss.item() * inputs.size(0)

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(logits)

            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)

    # Concatenate lists to numpy arrays
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Handle edge case where only one class is present in the batch/dataset (rare but possible in debug)
    try:
        val_auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        val_auc = 0.5

    return val_loss, val_auc


def run_training():
    """
    Main function to run the training pipeline.
    """
    set_seed()

    print(f"Device: {Config.DEVICE}")

    # 1. Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # 2. Create DataLoaders
    train_loader = get_dataloader(
        df_train,
        split="train",
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug=Config.DEBUG,
    )

    val_loader = get_dataloader(
        df_val,
        split="val",
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug=Config.DEBUG,
    )

    # 3. Initialize Model, Optimizer, Loss
    model = BraTS2DCNN().to(Config.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE
        )
        val_loss, val_auc = validate(model, val_loader, criterion, Config.DEVICE)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc}"
        )

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")


def generate_submission():
    """
    Generates predictions for the test set using the best saved model.
    """
    set_seed()

    # 1. Load Test Metadata
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)

    # 2. Create Test DataLoader
    test_loader = get_dataloader(
        df_test,
        split="test",
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug=Config.DEBUG,
    )

    # 3. Load Model
    model = BraTS2DCNN().to(Config.DEVICE)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        )
        print("Loaded best model for inference.")
    else:
        print(
            "Warning: No saved model found. Using random weights (likely debug mode or training failed)."
        )

    model.eval()

    # 4. Inference
    predictions = []
    ids = []

    with torch.no_grad():
        for inputs, _ in test_loader:
            # Note: The dataset returns dummy targets for test set, we ignore them
            inputs = inputs.to(Config.DEVICE)

            # Use autocast for inference
            with torch.cuda.amp.autocast():
                logits = model(inputs)
                probs = torch.sigmoid(logits)

            predictions.extend(probs.cpu().numpy().flatten())

            # We need to map these predictions back to BraTS21IDs.
            # Since DataLoader preserves order (shuffle=False for test),
            # we can iterate the dataframe or rely on the loader order.
            # However, batching makes direct mapping slightly implicit.
            # A safer way is to rely on the order of df_test which was passed to the loader.

    # 5. Create Submission DataFrame
    # The loader iterates sequentially through the dataframe provided
    # If debug is on, the loader only took the head of the dataframe
    if Config.DEBUG:
        df_test = df_test.head(Config.DEBUG_SAMPLE_SIZE)

    # Ensure lengths match
    if len(predictions) != len(df_test):
        print(
            f"Warning: Number of predictions ({len(predictions)}) does not match number of test subjects ({len(df_test)})."
        )
        # Truncate or pad if necessary (should not happen in standard execution)
        min_len = min(len(predictions), len(df_test))
        predictions = predictions[:min_len]
        df_test = df_test.iloc[:min_len]

    submission_df = pd.DataFrame(
        {"BraTS21ID": df_test["BraTS21ID"], "MGMT_value": predictions}
    )

    # 6. Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
