import copy
import numpy as np
import torch
import torch.nn as nn
from library.utils import calculate_score


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model state using deepcopy.
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

    def restore_best_weights(self, model):
        """Restores the best weights to the model."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        return model


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(loader):
        # Unpack inputs (image, angle)
        images, angles = inputs

        # Move to device
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device).unsqueeze(1)  # Ensure shape (N, 1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, angles)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and log loss.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            images, angles = inputs

            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)

            # Apply sigmoid to get probabilities for metric calculation
            probs = torch.sigmoid(outputs)

            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)

    # Calculate Log Loss metric
    # Flatten arrays
    y_true = np.array(all_targets).ravel()
    y_pred = np.array(all_preds).ravel()

    # Use library utility for consistent metric calculation
    metric_score = calculate_score(y_true, y_pred)

    return avg_loss, metric_score


def fit_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs,
    patience,
    scheduler=None,
):
    """
    Main training loop.
    """
    early_stopping = EarlyStopping(patience=patience, delta=0.0001)

    print(f"Starting training for {num_epochs} epochs with patience {patience}...")

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metric = validate(model, val_loader, criterion, device)

        # Step scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {train_loss} - "
            f"Val Loss: {val_loss} - "
            f"Val LogLoss: {val_metric}"
        )

        # Check early stopping
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Restore best model
    print(f"Restoring best model with Val Loss: {early_stopping.val_loss_min}")
    model = early_stopping.restore_best_weights(model)

    return model
