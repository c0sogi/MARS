import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import save_checkpoint


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training set.
        optimizer (Optimizer): The optimizer.
        criterion (Loss): The loss function.
        device (str): Device to run training on.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_samples = 0

    for x_cont, x_cat, y in dataloader:
        x_cont = x_cont.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(x_cont, x_cat)
        loss = criterion(outputs, y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)
        num_samples += y.size(0)

    return running_loss / num_samples


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation set.
        criterion (Loss): The loss function.
        device (str): Device to run evaluation on.

    Returns:
        tuple: (average_loss, auc_score)
    """
    model.eval()
    running_loss = 0.0
    num_samples = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for x_cont, x_cat, y in dataloader:
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)

            outputs = model(x_cont, x_cat)
            loss = criterion(outputs, y)

            running_loss += loss.item() * y.size(0)
            num_samples += y.size(0)

            all_targets.append(y.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

    avg_loss = running_loss / num_samples

    # Concatenate all batches
    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    # Calculate AUC
    # Ensure there are at least 2 classes to calculate AUC
    if len(np.unique(all_targets)) > 1:
        auc = roc_auc_score(all_targets, all_preds)
    else:
        auc = 0.5

    return avg_loss, auc


def predict(model, dataloader, device):
    """
    Generates predictions for the test set.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on.

    Returns:
        np.ndarray: Array of predicted probabilities.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            # Test loader returns (x_cont, x_cat)
            x_cont, x_cat = batch
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device)

            outputs = model(x_cont, x_cat)
            all_preds.append(outputs.cpu().numpy())

    return np.concatenate(all_preds)


class Trainer:
    """
    Encapsulates the training loop, validation, and early stopping logic.
    """

    def __init__(self, model, optimizer, scheduler, criterion, device, config):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.config = config

    def fit(self, train_loader, val_loader, epochs, patience=5):
        """
        Runs the full training process.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            epochs (int): Maximum number of epochs.
            patience (int): Early stopping patience.
        """
        best_auc = 0.0
        patience_counter = 0

        print(
            f"Starting training on {self.device} for {epochs} epochs with patience {patience}..."
        )

        for epoch in range(epochs):
            # Train
            train_loss = train_one_epoch(
                self.model, train_loader, self.optimizer, self.criterion, self.device
            )

            # Validate
            val_loss, val_auc = evaluate(
                self.model, val_loader, self.criterion, self.device
            )

            # Step Scheduler
            if self.scheduler:
                self.scheduler.step()

            # Logging (Full precision as requested)
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")
            print(f"Val AUC: {val_auc}")

            # Checkpoint & Early Stopping
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                print("New best model found. Saving checkpoint...")
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_auc": best_auc,
                    },
                    self.config.MODEL_PATH,
                )
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation AUC: {best_auc}")
