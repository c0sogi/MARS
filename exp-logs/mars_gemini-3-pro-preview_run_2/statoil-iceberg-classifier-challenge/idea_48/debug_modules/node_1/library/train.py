import copy
import torch
import torch.nn as nn
import numpy as np
from library.model import DCWBN


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model state using deepcopy.
    """

    def __init__(self, patience=7, min_delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, inc_angles, labels in loader:
        images = images.to(device)
        inc_angles = inc_angles.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass: DCWBN takes both images and incidence angles
        outputs = model(images, inc_angles)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, inc_angles, labels in loader:
            images = images.to(device)
            inc_angles = inc_angles.to(device)
            labels = labels.to(device)

            outputs = model(images, inc_angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Get probabilities for Class 1 (Iceberg)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc, np.array(all_preds), np.array(all_targets)


def run_fold(
    fold_idx, train_loader, val_loader, device, learning_rate, num_epochs, patience
):
    """
    Orchestrates the training process for a single fold.

    Args:
        fold_idx (int): Index of the current fold.
        train_loader (DataLoader): Loader for training data.
        val_loader (DataLoader): Loader for validation data.
        device (torch.device): Device to train on.
        learning_rate (float): Learning rate for Adam.
        num_epochs (int): Maximum number of epochs.
        patience (int): Patience for early stopping.

    Returns:
        dict: The state dictionary of the best model found during training.
    """
    print(f"Starting Fold {fold_idx}")

    # Initialize Model
    model = DCWBN().to(device)

    # Initialize Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    # Initialize Loss
    # Using CrossEntropyLoss as DCWBN outputs 2 units (logits for class 0 and 1)
    criterion = nn.CrossEntropyLoss()

    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=patience)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device)

        # Print metrics with full precision
        print(
            f"Fold {fold_idx} Epoch {epoch+1}: Train Loss {train_loss}, Val Loss {val_loss}, Val Acc {val_acc}"
        )

        # Step scheduler
        scheduler.step(val_loss)

        # Check early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Fold {fold_idx} finished. Best Val Loss: {early_stopping.best_loss}")

    # Return best model state
    return early_stopping.best_model_state
