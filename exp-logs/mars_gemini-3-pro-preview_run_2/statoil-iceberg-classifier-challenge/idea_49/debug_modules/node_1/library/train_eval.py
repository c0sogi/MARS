import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import logging
from sklearn.metrics import log_loss, accuracy_score
from library.config import Config
from library.model import DCSWBN


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model state.
    """

    def __init__(self, patience=7, delta=0, verbose=False, path="checkpoint.pt"):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.best_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        self.best_state = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch_idx, (images, angles, labels, _) in enumerate(dataloader):
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # Ensure shape (B, 1)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Store predictions for accuracy calculation
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.append(probs)
        all_targets.append(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate accuracy (threshold 0.5)
    preds_binary = (all_preds > 0.5).astype(int)
    epoch_acc = accuracy_score(all_targets, preds_binary)

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, angles, labels, _ in dataloader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.cpu().numpy())

    total_loss = running_loss / len(dataloader.dataset)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    # Calculate metrics
    # Log loss requires probabilities
    # Clip probabilities to avoid log(0) errors, though sklearn handles this usually
    val_log_loss = log_loss(all_targets, all_preds, eps=1e-15)

    preds_binary = (all_preds > 0.5).astype(int)
    val_acc = accuracy_score(all_targets, preds_binary)

    return total_loss, val_log_loss, val_acc


def train_fold(fold_index, train_loader, val_loader, logger=None):
    """
    Executes the training pipeline for a single fold.
    Returns the best model state dict.
    """
    device = torch.device(Config.DEVICE)

    # Initialize Model
    model = DCSWBN().to(device)

    # Loss Function
    # BCEWithLogitsLoss combines Sigmoid and BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    if Config.OPTIMIZER == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
    else:
        # Fallback to Adam if not specified
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.SCHEDULER_MIN_LR,
    )

    # Early Stopping
    early_stopping = EarlyStopping(patience=Config.PATIENCE, verbose=False)

    if logger:
        logger.info(f"Starting training for Fold {fold_index}...")
        logger.info(f"Device: {device}")
        logger.info(f"Training samples: {len(train_loader.dataset)}")
        logger.info(f"Validation samples: {len(val_loader.dataset)}")

    best_val_loss = float("inf")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_metric_logloss, val_acc = validate(
            model, val_loader, criterion, device
        )

        # Update Scheduler
        scheduler.step(val_loss)

        # Logging
        if logger:
            logger.info(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
                f"Train Loss: {train_loss:.10f} - Train Acc: {train_acc:.10f} - "
                f"Val Loss: {val_loss:.10f} - Val LogLoss: {val_metric_logloss:.10f} - Val Acc: {val_acc:.10f}"
            )
        else:
            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - "
                f"Train Loss: {train_loss:.10f} - Train Acc: {train_acc:.10f} - "
                f"Val Loss: {val_loss:.10f} - Val LogLoss: {val_metric_logloss:.10f} - Val Acc: {val_acc:.10f}"
            )

        # Check Early Stopping
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            if logger:
                logger.info("Early stopping triggered.")
            else:
                print("Early stopping triggered.")
            break

    if logger:
        logger.info(
            f"Fold {fold_index} finished. Best Val Loss: {early_stopping.val_loss_min:.10f}"
        )

    # Return the best model state found during training
    return early_stopping.best_state
