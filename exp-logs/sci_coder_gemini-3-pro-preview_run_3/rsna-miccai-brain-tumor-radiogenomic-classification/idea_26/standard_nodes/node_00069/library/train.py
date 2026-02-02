import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import numpy as np

from library.config import (
    TRAIN_META_PATH,
    VAL_META_PATH,
    MODEL_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_WORKERS,
    DEVICE,
    SEED,
)
from library.utils import seed_everything, AverageMeter
from library.data import get_dataset
from library.model import SSBHDNetwork


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # Ensure shape (B, 1)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    losses = AverageMeter()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, targets)

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(logits)

            losses.update(loss.item(), images.size(0))
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_targets = np.array(all_targets).flatten()
    all_probs = np.array(all_probs).flatten()

    # Handle edge case where only one class is present in the batch/set
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.5

    if np.isnan(auc):
        auc = 0.5

    return losses.avg, auc


def run_training(
    train_meta_path=TRAIN_META_PATH,
    val_meta_path=VAL_META_PATH,
    model_save_path=MODEL_PATH,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    load_cached_data=True,
    patience=5,
):
    """
    Main function to run the training pipeline.
    """
    # 1. Reproducibility
    seed_everything(SEED)
    print(f"Starting training on device: {DEVICE}")

    # 2. Data Loading
    # Use the factory function which handles caching logic
    train_dataset = get_dataset(
        metadata_path=train_meta_path,
        dataset_type="train",
        load_cached_data=load_cached_data,
    )
    val_dataset = get_dataset(
        metadata_path=val_meta_path,
        dataset_type="val",
        load_cached_data=load_cached_data,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 3. Model Setup
    model = SSBHDNetwork()
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. Training Loop
    best_auc = 0.0
    epochs_no_improve = 0

    # Ensure save directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # Validate
        val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

        # Print metrics with full precision
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val AUC: {val_auc}")

        # Checkpoint & Early Stopping
        if val_auc > best_auc:
            print(
                f"Validation AUC improved from {best_auc} to {val_auc}. Saving model..."
            )
            best_auc = val_auc
            torch.save(model.state_dict(), model_save_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement in AUC. Patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation AUC: {best_auc}")
