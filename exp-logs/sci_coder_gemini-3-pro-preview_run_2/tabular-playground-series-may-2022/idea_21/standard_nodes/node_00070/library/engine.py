import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import random
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.dataset import get_dataloaders
from library.network import HybridModel


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        continuous = batch["continuous"].to(device)
        sequence = batch["sequence"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)

        batch_size = continuous.size(0)
        dataset_size += batch_size

        optimizer.zero_grad()

        # Forward pass
        outputs = model(continuous, sequence)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and AUC score.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            continuous = batch["continuous"].to(device)
            sequence = batch["sequence"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)

            batch_size = continuous.size(0)
            dataset_size += batch_size

            outputs = model(continuous, sequence)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size

            # Apply sigmoid to logits for probability
            preds = torch.sigmoid(outputs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Compute AUC
    auc = roc_auc_score(all_targets, all_preds)

    return epoch_loss, auc


def run_training():
    """
    Main training loop.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Get DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # 2. Initialize Model
    model = HybridModel().to(device)

    # 3. Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.SCHEDULER_STEP_SIZE, gamma=Config.SCHEDULER_GAMMA
    )

    criterion = nn.BCEWithLogitsLoss()

    # 4. Training Loop
    best_auc = 0.0
    patience = 12
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Print metrics (full precision)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | LR: {current_lr} | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved to {Config.MODEL_PATH}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")
    return model


def generate_submission():
    """
    Generates predictions for the test set using the best model and saves to CSV.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Get Test Loader
    _, _, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # 2. Load Best Model
    model = HybridModel().to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print(f"Loaded model from {Config.MODEL_PATH}")
    else:
        print("Warning: No model checkpoint found. Using random initialization.")

    model.eval()
    all_preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            continuous = batch["continuous"].to(device)
            sequence = batch["sequence"].to(device)

            outputs = model(continuous, sequence)
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())

    # Flatten predictions
    all_preds = np.concatenate(all_preds).flatten()

    # 3. Create Submission DataFrame
    # We read the test metadata to ensure IDs match the order of the DataLoader
    test_meta = pd.read_csv(Config.TEST_METADATA)
    ids = test_meta["id"].values

    if len(ids) != len(all_preds):
        raise ValueError(f"Mismatch: {len(ids)} IDs vs {len(all_preds)} predictions.")

    submission = pd.DataFrame({"id": ids, "target": all_preds})

    # 4. Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
