import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import DIDPNet


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model state in memory using deepcopy.
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
        self.best_state_dict = None

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
        """Saves model state dict to memory."""
        self.best_state_dict = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def run_fold(fold_index, debug=False):
    """
    Runs the training pipeline for a single fold.
    """
    print(f"\nStarting Fold {fold_index}...")

    # Set seed for this fold to ensure reproducibility
    seed_everything(Config.SEED + fold_index)

    device = get_device()

    # Get DataLoaders
    train_loader, val_loader, _, _ = get_dataloaders(
        Config, fold_index=fold_index, debug=debug
    )

    # Initialize Model
    model = DIDPNet(
        backbone_filters=Config.BACKBONE_FILTERS, dropout_rate=Config.DROPOUT_RATE
    )
    model.to(device)

    # Loss Function (BCEWithLogitsLoss includes Sigmoid)
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer ("Low and Slow" initialization)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR,
    )

    # Early Stopping
    early_stopping = EarlyStopping(patience=Config.PATIENCE, delta=0)

    # Training Loop
    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Check Early Stopping
        early_stopping(val_loss, model)

        elapsed = time.time() - start_time

        print(
            f"Fold {fold_index} | Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss} | Val Loss: {val_loss} | "
            f"LR: {current_lr} | Time: {elapsed}s"
        )

        if early_stopping.early_stop:
            print(f"Early stopping triggered for Fold {fold_index}")
            break

    # Save best model
    save_path = Config.MODEL_PATH_TEMPLATE.format(fold_index)
    print(
        f"Saving best model for Fold {fold_index} to {save_path} with Loss: {early_stopping.val_loss_min}"
    )
    torch.save(early_stopping.best_state_dict, save_path)

    return early_stopping.val_loss_min


def train_all_folds(debug=False):
    """
    Orchestrates training across all folds.
    """
    Config.setup()
    seed_everything(Config.SEED)

    fold_scores = []

    for fold in range(Config.N_FOLDS):
        score = run_fold(fold, debug=debug)
        fold_scores.append(score)

    print("\n" + "=" * 30)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 30)
    for i, score in enumerate(fold_scores):
        print(f"Fold {i}: {score}")

    avg_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print(f"Average CV Score: {avg_score} (Std: {std_score})")

    return fold_scores
