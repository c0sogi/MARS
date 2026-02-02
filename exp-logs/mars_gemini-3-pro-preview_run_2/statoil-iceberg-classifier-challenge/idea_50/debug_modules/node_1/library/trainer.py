import os
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import (
    DEVICE,
    MODEL_DIR,
    PATIENCE,
    MAX_EPOCHS,
    LEARNING_RATE,
    NUM_FOLDS,
    SEED,
)
from library.utils import set_seed
from library.model import CA_WBN
from library.data_loader import get_loaders


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model state in memory using deepcopy.
    """

    def __init__(self, patience=15, delta=0):
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
        self.best_state = None
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model state when validation loss decreases."""
        self.best_state = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for imgs, metas, targets, _ in loader:
        imgs = imgs.to(device)
        metas = metas.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(imgs, metas)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for imgs, metas, targets, _ in loader:
            imgs = imgs.to(device)
            metas = metas.to(device)
            targets = targets.to(device)

            outputs = model(imgs, metas)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def run_training(debug=False):
    """
    Main training function implementing Stratified 5-Fold Cross-Validation.

    Args:
        debug (bool): If True, runs with a smaller subset of data for debugging.
    """
    set_seed(SEED)

    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    fold_perf = []

    for fold in range(NUM_FOLDS):
        print(f"\n=== Fold {fold} ===")

        # Get DataLoaders for the current fold
        train_loader, val_loader = get_loaders(fold, debug=debug)

        # Initialize Model, Optimizer, Criterion
        model = CA_WBN().to(DEVICE)

        # Revert from AdamW to Adam as per idea description ("Low and Slow" strategy)
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

        # Scheduler: ReduceLROnPlateau with patience=5 (distinct from early stopping patience)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        criterion = nn.BCELoss()

        # Initialize Early Stopping
        early_stopping = EarlyStopping(patience=PATIENCE, delta=0)

        for epoch in range(MAX_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )
            val_loss = validate(model, val_loader, criterion, DEVICE)

            # Print full precision metrics
            print(
                f"Epoch {epoch+1}/{MAX_EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss}"
            )

            # Step scheduler
            scheduler.step(val_loss)

            # Check early stopping
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Save the best model for this fold
        best_loss = early_stopping.val_loss_min
        print(f"Fold {fold} finished. Best Validation Loss: {best_loss}")
        fold_perf.append(best_loss)

        save_path = os.path.join(MODEL_DIR, f"model_fold_{fold}.pth")
        if early_stopping.best_state is not None:
            torch.save(early_stopping.best_state, save_path)
            print(f"Saved best model to {save_path}")
        else:
            # Fallback if training failed to produce a better model than init (unlikely)
            torch.save(model.state_dict(), save_path)
            print(f"Saved final model to {save_path} (No improvement found)")

    print(f"\nAverage CV Loss: {np.mean(fold_perf)}")
