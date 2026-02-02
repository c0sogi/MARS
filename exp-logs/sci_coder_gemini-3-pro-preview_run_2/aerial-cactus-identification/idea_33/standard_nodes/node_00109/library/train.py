import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from library.config import Config
from library.utils import set_seed, calculate_roc_auc, save_checkpoint
from library.dataset import get_dataloaders
from library.model import RepVGG


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # BCEWithLogitsLoss expects (N, 1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Store for metrics
        probs = torch.sigmoid(outputs)
        all_targets.append(labels.detach().cpu().numpy())
        all_preds.append(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    epoch_auc = calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs)
            all_targets.append(labels.detach().cpu().numpy())
            all_preds.append(probs.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    epoch_auc = calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def run_training(seed, epochs=Config.EPOCHS, load_cached_data=True):
    """
    Manages the training process for a specific seed.

    Args:
        seed (int): Random seed for reproducibility.
        epochs (int): Number of training epochs.
        load_cached_data (bool): Whether to load data from cache or process from scratch.
    """
    set_seed(seed)
    device = Config.DEVICE

    # Data
    train_loader, val_loader, _, _ = get_dataloaders(load_cached_data=load_cached_data)

    # Model
    model = RepVGG(deploy=False).to(device)

    # Optimization
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    # Use the passed 'epochs' argument for T_max to ensure scheduler aligns with run length
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=Config.ETA_MIN)

    # Training Loop
    best_val_auc = -float("inf")
    patience_counter = 0

    print(f"\n[Seed {seed}] Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss} | Train AUC: {train_auc} | "
            f"Val Loss: {val_loss} | Val AUC: {val_auc}"
        )

        # Checkpoint & Early Stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            save_checkpoint(model.state_dict(), seed)
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"[Seed {seed}] Best Val AUC: {best_val_auc}")
