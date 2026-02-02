import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.data import get_dataloader
from library.model import AsymmetricEfficientNet


def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one training epoch.
    Returns:
        epoch_loss (float): Average loss for the epoch.
        epoch_auc (float): ROC AUC score for the epoch.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # Ensure shape (B, 1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Store predictions for AUC calculation
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets_np = targets.cpu().numpy()

        all_preds.append(probs)
        all_targets.append(targets_np)

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate AUC
    # Handle potential edge case where batch/epoch might only have one class
    if len(np.unique(all_targets)) < 2:
        epoch_auc = 0.5
    else:
        epoch_auc = roc_auc_score(all_targets, all_preds)

    return epoch_loss, epoch_auc


def validate_epoch(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns:
        epoch_loss (float): Average loss for the epoch.
        epoch_auc (float): ROC AUC score for the epoch.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            targets_np = targets.cpu().numpy()

            all_preds.append(probs)
            all_targets.append(targets_np)

    epoch_loss = running_loss / len(loader.dataset)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    if len(np.unique(all_targets)) < 2:
        epoch_auc = 0.5
    else:
        epoch_auc = roc_auc_score(all_targets, all_preds)

    return epoch_loss, epoch_auc


def run_training(
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    patience=Config.PATIENCE,
    debug=False,
):
    """
    Orchestrates the training process, including data loading, model initialization,
    optimization loop, early stopping, and checkpointing.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 1. Load Metadata
    if not os.path.exists(Config.TRAIN_METADATA) or not os.path.exists(
        Config.VAL_METADATA
    ):
        raise FileNotFoundError(
            "Metadata files not found. Ensure metadata generation script has run."
        )

    train_df = pd.read_csv(Config.TRAIN_METADATA)
    val_df = pd.read_csv(Config.VAL_METADATA)

    if debug:
        print("Debug mode: using subset of data.")
        train_df = train_df.head(batch_size * 2)
        val_df = val_df.head(batch_size)

    # 2. Create DataLoaders
    train_loader = get_dataloader(train_df, phase="train", batch_size=batch_size)
    val_loader = get_dataloader(val_df, phase="val", batch_size=batch_size)

    # 3. Initialize Model
    model = AsymmetricEfficientNet(num_classes=1)
    model = model.to(device)

    # 4. Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0
    best_model_path = Config.MODEL_SAVE_PATH

    # Ensure save directory exists
    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

    print("Starting training...")
    for epoch in range(1, epochs + 1):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate_epoch(model, val_loader, criterion, device)

        # Print full precision metrics
        print(f"Epoch {epoch}/{epochs}")
        print(f"Train Loss: {train_loss}")
        print(f"Train AUC: {train_auc}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Checkpointing & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best Validation AUC: {best_auc}")
    return best_auc


def generate_submission(
    model_path=Config.MODEL_SAVE_PATH, output_path=Config.SUBMISSION_FILE
):
    """
    Generates predictions for the test set using the best saved model.
    Implements Test-Time Augmentation (TTA) by averaging predictions from:
    1. Original Image
    2. Horizontal Flip
    3. Vertical Flip
    """
    set_seed(Config.SEED)
    device = Config.DEVICE

    # 1. Load Test Metadata
    if not os.path.exists(Config.TEST_METADATA):
        raise FileNotFoundError("Test metadata not found.")

    test_df = pd.read_csv(Config.TEST_METADATA)
    test_loader = get_dataloader(
        test_df, phase="test", batch_size=Config.BATCH_SIZE, shuffle=False
    )

    # 2. Load Model
    model = AsymmetricEfficientNet(num_classes=1)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    results = []
    print("Generating submission with TTA...")

    with torch.no_grad():
        for images, subject_ids in test_loader:
            images = images.to(device)  # Shape: (B, 12, H, W)

            # TTA 1: Original
            logits_1 = model(images)
            probs_1 = torch.sigmoid(logits_1)

            # TTA 2: Horizontal Flip (dim 3 is width)
            images_h = torch.flip(images, [3])
            logits_2 = model(images_h)
            probs_2 = torch.sigmoid(logits_2)

            # TTA 3: Vertical Flip (dim 2 is height)
            images_v = torch.flip(images, [2])
            logits_3 = model(images_v)
            probs_3 = torch.sigmoid(logits_3)

            # Average Probabilities
            avg_probs = (probs_1 + probs_2 + probs_3) / 3.0
            avg_probs_np = avg_probs.cpu().numpy().flatten()

            # Store results
            for sub_id, prob in zip(subject_ids, avg_probs_np):
                results.append({"BraTS21ID": sub_id, "MGMT_value": prob})

    # 3. Save Submission
    submission_df = pd.DataFrame(results)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
