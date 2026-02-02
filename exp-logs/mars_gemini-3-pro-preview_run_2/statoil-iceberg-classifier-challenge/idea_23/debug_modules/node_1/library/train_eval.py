import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
import os
import time
from library.config import Config
from library.utils import save_checkpoint
from library.model import WBMGNet


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    Saves the best model state using copy.deepcopy and file persistence.
    """

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pth",
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.best_model_state = None

    def __call__(self, val_loss, model, optimizer):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min} --> {val_loss}).  Saving model ..."
            )

        # Deepcopy the state dict to preserve best weights in memory
        self.best_model_state = copy.deepcopy(model.state_dict())

        # Save to disk
        state = {
            "state_dict": self.best_model_state,
            "optimizer": optimizer.state_dict(),
            "val_loss": val_loss,
        }
        save_checkpoint(state, self.path)
        self.val_loss_min = val_loss


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).float().unsqueeze(1)

        optimizer.zero_grad()

        # Forward pass
        # model outputs logits
        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        # Backward and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Calculate accuracy
        probs = torch.sigmoid(outputs)
        predicted = (probs > 0.5).float()
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

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs)
            predicted = (probs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


def train_fold(fold_idx, train_loader, val_loader, device):
    """
    Orchestrates the training process for a single fold.

    Args:
        fold_idx (int): Index of the current fold.
        train_loader (DataLoader): Training data.
        val_loader (DataLoader): Validation data.
        device (torch.device): Compute device.

    Returns:
        model (nn.Module): The trained model with best weights loaded.
        history (dict): Training history.
    """
    print(f"\n{'='*20} Fold {fold_idx} {'='*20}")

    # Initialize Model
    model = WBMGNet().to(device)

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    # Early Stopping
    checkpoint_path = Config.get_checkpoint_path(fold_idx)
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE, verbose=True, path=checkpoint_path
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    start_time = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # Print metrics (Full precision)
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.10f} | Train Acc: {train_acc:.10f} | "
            f"Val Loss: {val_loss:.10f} | Val Acc: {val_acc:.10f}"
        )

        # Scheduler Step
        scheduler.step(val_loss)

        # Early Stopping Step
        early_stopping(val_loss, model, optimizer)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Fold {fold_idx} finished in {total_time:.2f}s")

    # Load best weights
    if early_stopping.best_model_state is not None:
        print(
            f"Loading best model weights from epoch with loss {early_stopping.val_loss_min:.10f}"
        )
        model.load_state_dict(early_stopping.best_model_state)

    return model, history


def predict(model, loader, device):
    """
    Generates predictions for a dataloader.

    Returns:
        np.array: Flattened array of probabilities.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in loader:
            # Handle cases where loader returns (img, angle) or (img, angle, label)
            if len(batch) == 3:
                images, angles, _ = batch
            else:
                images, angles = batch

            images = images.to(device)
            angles = angles.to(device)

            outputs = model(images, angles)
            probs = torch.sigmoid(outputs)

            preds.extend(probs.cpu().numpy().flatten())

    return np.array(preds)
