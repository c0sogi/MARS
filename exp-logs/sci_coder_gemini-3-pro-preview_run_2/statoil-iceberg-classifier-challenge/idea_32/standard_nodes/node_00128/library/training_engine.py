import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os
import sys

# Import from provided library files
from library.config import Config
from library.model_architecture import RobustDualPathWideBodyNet
from library.data_processing import get_fold_loaders

# Set seeds for reproducibility within this module scope as well
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model state in memory.
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

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model state when validation loss decreases."""
        self.best_model_state = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, inc_angles, labels) in enumerate(loader):
        images = images.to(device)
        inc_angles = inc_angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # BCEWithLogitsLoss expects (N, 1)

        optimizer.zero_grad()

        outputs = model(images, inc_angles)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, inc_angles, labels in loader:
            images = images.to(device)
            inc_angles = inc_angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, inc_angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def train_fold(fold_idx):
    """
    Orchestrates the training process for a single fold.
    """
    print(f"\nStarting training for Fold {fold_idx}")

    # Ensure directories exist
    Config.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Data
    train_loader, val_loader = get_fold_loaders(fold_idx, load_cached_data=True)

    # Initialize Model
    model = RobustDualPathWideBodyNet()
    model.to(device)

    # Optimizer
    # Reverting to standard Adam as per solution description
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Scheduler
    # Reduce LR when validation loss stagnates
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=5
    )

    # Early Stopping
    early_stopping = EarlyStopping(patience=Config.PATIENCE, delta=0)

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Fold {fold_idx} Epoch {epoch+1}/{Config.EPOCHS} - "
            f"Train Loss: {train_loss:.8f} - "
            f"Val Loss: {val_loss:.8f} - "
            f"LR: {current_lr:.8f}"
        )

        # Check Early Stopping
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # Save best model
    save_path = Config.get_model_path(fold_idx)
    print(
        f"Saving best model for Fold {fold_idx} (Val Loss: {early_stopping.val_loss_min:.8f}) to {save_path}"
    )
    torch.save(early_stopping.best_model_state, save_path)

    return early_stopping.val_loss_min
