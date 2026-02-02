import os
import copy
import torch
import torch.nn as nn
import numpy as np
from library import config, data, model, utils


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Stores the best model state in memory using deepcopy.
    """

    def __init__(self, patience=7, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.best_model_state = None

    def __call__(self, val_loss, model_obj):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model_obj)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model_obj)
            self.counter = 0

    def save_checkpoint(self, val_loss, model_obj):
        """Saves the best model state in memory."""
        self.best_model_state = copy.deepcopy(model_obj.state_dict())
        self.val_loss_min = val_loss

    def restore_best_weights(self, model_obj):
        """Restores the best model weights from memory."""
        if self.best_model_state is not None:
            model_obj.load_state_dict(self.best_model_state)
            print(f"Restored model with validation loss: {self.val_loss_min}")


def train_one_epoch(model_obj, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model_obj.train()
    running_loss = 0.0

    for batch_idx, (images, angles, targets) in enumerate(loader):
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        # Forward pass
        outputs = model_obj(images, angles)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model_obj, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model_obj.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, angles, targets in loader:
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model_obj(images, angles)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)

    val_loss = running_loss / len(loader.dataset)
    return val_loss


def run_fold(fold_idx):
    """
    Runs the training pipeline for a specific fold.

    Args:
        fold_idx (int): The index of the fold to train (0 to NUM_FOLDS-1).
    """
    utils.seed_everything(config.SEED)

    device = config.DEVICE
    print(f"Starting training for Fold {fold_idx} on device: {device}")

    # 1. Prepare Data
    train_loader, val_loader = data.get_dataloaders(fold_idx, load_cached_data=True)

    # 2. Initialize Model
    net = model.RDP_WBN()
    net = net.to(device)

    # 3. Setup Optimization
    criterion = nn.BCEWithLogitsLoss()
    # Use AdamW for better regularization (Cite Lesson 00077)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # "Low and Slow" scheduler as per Idea
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # 4. Early Stopping
    early_stopping = EarlyStopping(patience=config.PATIENCE, delta=0.0001)

    # 5. Training Loop
    for epoch in range(config.NUM_EPOCHS):
        train_loss = train_one_epoch(net, train_loader, criterion, optimizer, device)
        val_loss = validate(net, val_loader, criterion, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Print metrics (full precision)
        print(
            f"Fold {fold_idx} | Epoch {epoch+1}/{config.NUM_EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss}"
        )

        # Check Early Stopping
        early_stopping(val_loss, net)

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 6. Restore best weights and save to disk
    early_stopping.restore_best_weights(net)

    save_path = os.path.join(config.MODEL_CHECKPOINT_DIR, f"model_fold_{fold_idx}.pth")
    torch.save(net.state_dict(), save_path)
    print(f"Model for Fold {fold_idx} saved to {save_path}")

    return early_stopping.val_loss_min
