import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import time
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.data_utils import get_dataloaders
from library.model import ResFunnelGLU


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        numeric = batch["numeric"].to(device)
        categorical = batch["categorical"].to(device)
        targets = batch["target"].to(device).view(-1, 1)

        optimizer.zero_grad()

        logits = model(numeric, categorical)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * numeric.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set using ROC AUC.
    """
    model.eval()
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            numeric = batch["numeric"].to(device)
            categorical = batch["categorical"].to(device)
            targets = batch["target"].to(device)

            logits = model(numeric, categorical)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_targets = np.concatenate(all_targets).ravel()

    auc = roc_auc_score(all_targets, all_preds)
    return auc


def predict_and_submit(model, test_loader, test_ids, device):
    """
    Generates predictions on the test set and saves the submission file.
    """
    print("Generating predictions on test set...")

    # Load best model weights
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print("Loaded best model weights.")
    else:
        print("Warning: Best model weights not found. Using current model weights.")

    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            numeric = batch["numeric"].to(device)
            categorical = batch["categorical"].to(device)

            logits = model(numeric, categorical)
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds).ravel()

    # Create Submission DataFrame
    submission = pd.DataFrame({"id": test_ids, "target": all_preds})

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission.head())


def train_model():
    """
    Main training pipeline.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=True
    )

    # Initialize Model
    model = ResFunnelGLU()
    model.to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()

    # Training Loop
    best_auc = 0.0
    patience_counter = 0

    print("Starting training...")
    start_time = time.time()

    for epoch in range(Config.EPOCHS):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_auc = validate(model, val_loader, device)

        epoch_time = time.time() - epoch_start
        # Printing full precision as requested
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Time: {epoch_time:.2f}s | Train Loss: {train_loss} | Val AUC: {val_auc}"
        )

        # Early Stopping & Model Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training complete. Total time: {total_time:.2f}s. Best Val AUC: {best_auc}")

    # Generate Submission
    predict_and_submit(model, test_loader, test_ids, device)
