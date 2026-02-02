import torch
import torch.nn as nn
import numpy as np
import copy
import sys


class Trainer:
    """
    Manages the training, validation, and optimization of the SEA-HN model.
    """

    def __init__(self, model, device, optimizer, scheduler=None, criterion=None):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            device (torch.device): Device to run training on (CPU/GPU).
            optimizer (torch.optim.Optimizer): Optimizer instance.
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler.
            criterion (nn.Module, optional): Loss function. Defaults to BCEWithLogitsLoss.
        """
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion if criterion is not None else nn.BCEWithLogitsLoss()

        self.best_model_wts = None
        self.best_loss = float("inf")

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch in train_loader:
            # Unpack batch: (images, stats, labels)
            imgs, stats, labels = batch

            imgs = imgs.to(self.device)
            stats = stats.to(self.device)
            # BCEWithLogitsLoss requires target shape (N, 1) to match output
            labels = labels.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(imgs, stats)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Accumulate loss (multiply by batch size to handle last partial batch correctly)
            running_loss += loss.item() * imgs.size(0)
            dataset_size += imgs.size(0)

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def evaluate(self, val_loader):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        with torch.no_grad():
            for batch in val_loader:
                imgs, stats, labels = batch

                imgs = imgs.to(self.device)
                stats = stats.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                outputs = self.model(imgs, stats)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * imgs.size(0)
                dataset_size += imgs.size(0)

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def fit(self, train_loader, val_loader, epochs, patience):
        """
        Executes the full training loop with Early Stopping and LR Scheduling.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            epochs (int): Maximum number of epochs.
            patience (int): Early stopping patience.

        Returns:
            float: The best validation loss achieved.
        """
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_loss = float("inf")
        early_stopping_counter = 0

        print(f"Starting training for {epochs} epochs with patience {patience}...")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)

            # Print metrics with full precision
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss} | Val Loss: {val_loss}"
            )

            # Scheduler Step
            if self.scheduler:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Early Stopping Logic
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Load best model weights
        print(f"Training complete. Best Val Loss: {self.best_loss}")
        self.model.load_state_dict(self.best_model_wts)
        return self.best_loss


def predict(model, test_loader, device):
    """
    Generates predictions for the test set.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): Test data loader.
        device (torch.device): Device to run inference on.

    Returns:
        np.array: Flattened array of predicted probabilities.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in test_loader:
            # Test loader returns (imgs, stats) - no labels
            imgs, stats = batch

            imgs = imgs.to(device)
            stats = stats.to(device)

            # Forward pass
            logits = model(imgs, stats)

            # Apply sigmoid to convert logits to probabilities
            probs = torch.sigmoid(logits)

            # Move to CPU and flatten
            preds.extend(probs.cpu().numpy().flatten())

    return np.array(preds)
