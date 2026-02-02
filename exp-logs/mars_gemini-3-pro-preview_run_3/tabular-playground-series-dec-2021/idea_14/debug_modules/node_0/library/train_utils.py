import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from library.config import Config


class EarlyStopping:
    """
    Implements early stopping to terminate training when validation accuracy
    stops improving. Saves the best model weights using deepcopy.
    """

    def __init__(self, patience=7, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation accuracy improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = -np.Inf
        self.best_model_wts = None

    def __call__(self, val_acc, model):
        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        """Saves model when validation accuracy increases."""
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.val_acc_max = val_acc

    def load_best_weights(self, model):
        """Restores the model weights with the best validation accuracy."""
        if self.best_model_wts is not None:
            model.load_state_dict(self.best_model_wts)
        return model


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """
    Performs validation on the validation set.
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = correct / total
    return val_loss, val_acc


def run_training(model, train_loader, val_loader):
    """
    Orchestrates the training process: initializes optimizer/scheduler,
    runs the loop, handles early stopping, and returns the best model.
    """
    Config.set_seed(Config.SEED)
    device = Config.DEVICE
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler as per strategy
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    early_stopping = EarlyStopping(patience=Config.PATIENCE)

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss} Acc: {train_acc} | "
            f"Val Loss: {val_loss} Acc: {val_acc}"
        )

        early_stopping(val_acc, model)

        if early_stopping.early_stop:
            print(
                f"Early stopping triggered at epoch {epoch+1}. Best Val Acc: {early_stopping.val_acc_max}"
            )
            break

    # Ensure we return the model with the best weights found during training
    model = early_stopping.load_best_weights(model)

    return model
