import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.dataset import InkDataset, get_training_transforms
from library.model import InkDetectorFCN
from library.utils import find_best_threshold


class BCEDiceLoss(nn.Module):
    """
    Balanced loss function combining Binary Cross Entropy and Dice Loss.
    """

    def __init__(self, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = 1.0 - bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        # BCE Loss
        bce_loss = self.bce(logits, targets)

        # Dice Loss
        probs = torch.sigmoid(logits)

        # Flatten for Dice calculation
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (
            probs_flat.sum() + targets_flat.sum() + self.smooth
        )
        dice_loss = 1.0 - dice_score

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (volumes, labels) in enumerate(loader):
        volumes = volumes.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        logits = model(volumes)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss, best F0.5 score, and the threshold used to achieve it.
    """
    model.eval()
    running_loss = 0.0

    all_probs = []
    all_targets = []

    with torch.no_grad():
        for volumes, labels in loader:
            volumes = volumes.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            logits = model(volumes)
            loss = criterion(logits, labels)
            running_loss += loss.item()

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits)

            # Move to CPU for metric calculation to save GPU memory
            all_probs.append(probs.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)

    # Concatenate all batches
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)

    # Find optimal threshold for F0.5 score
    best_threshold, best_score = find_best_threshold(all_targets, all_probs)

    return avg_loss, best_score, best_threshold


def train_model(load_cached_data=True):
    """
    Main function to train the PSDN model.

    Args:
        load_cached_data (bool): Whether to use cached .npy files for dataset.

    Returns:
        tuple: (best_val_score, best_threshold)
    """
    # 1. Setup
    Config.set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Preparation
    print("Initializing Datasets...")
    train_dataset = InkDataset(
        split="train", transform=get_training_transforms(), cache_data=load_cached_data
    )

    val_dataset = InkDataset(split="val", transform=None, cache_data=load_cached_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model & Optimizer
    print("Initializing Model...")
    model = InkDetectorFCN().to(device)

    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = BCEDiceLoss()

    # 4. Training Loop
    best_val_score = -1.0
    best_threshold = 0.5
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_score, val_thresh = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val F0.5: {val_score} | "
            f"Best Thresh: {val_thresh:.4f}"
        )

        # Checkpoint & Early Stopping
        if val_score > best_val_score:
            best_val_score = val_score
            best_threshold = val_thresh
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(
        f"Training complete. Best Validation F0.5: {best_val_score} at Threshold: {best_threshold}"
    )
    return best_val_score, best_threshold
