import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np
import os

from library.config import (
    DEVICE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    NUM_WORKERS,
    MODEL_SAVE_PATH,
    PATIENCE,
    SEED,
    WORKING_DIR,
)
from library.utils import seed_everything, get_logger
from library.data_loader import get_datasets
from library.model_arch import MNSHDNetwork

# Initialize logger
logger = get_logger("trainer")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

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
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)

            # Apply sigmoid to get probabilities for AUC
            probs = torch.sigmoid(logits)

            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    total_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    # Calculate ROC AUC
    # Handle edge case where only one class is present in batch/set (though unlikely in full val set)
    try:
        auc_score = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc_score = 0.5

    return total_loss, auc_score


def run_training():
    """
    Main function to orchestrate the training process.
    """
    # 1. Setup
    seed_everything(SEED)
    os.makedirs(WORKING_DIR, exist_ok=True)

    logger.info(f"Starting training on device: {DEVICE}")

    # 2. Data Loading
    # Load cached data (or generate if not exists)
    train_dataset, val_dataset, _ = get_datasets(load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    # 3. Model Initialization
    model = MNSHDNetwork().to(DEVICE)

    # 4. Optimization
    criterion = nn.BCEWithLogitsLoss()
    # Adam optimizer without weight decay as per idea
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

        logger.info(
            f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val AUC: {val_auc}"
        )

        # Checkpoint & Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            logger.info(
                f"New best AUC found: {best_auc}. Saving model to {MODEL_SAVE_PATH}..."
            )
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            logger.info("Early stopping triggered.")
            break

    logger.info(f"Training complete. Best Validation AUC: {best_auc}")
