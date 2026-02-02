import time
import copy
import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import compute_auc


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        loader (DataLoader): Training data loader.
        optimizer (Optimizer): PyTorch optimizer.
        criterion (nn.Module): Loss function (e.g., BCEWithLogitsLoss).
        device (str): Device to train on.
        epoch (int): Current epoch number (for logging).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch_idx, (images, targets, _) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        batch_size = images.size(0)

        # Mixup Augmentation
        if Config.USE_MIXUP and Config.MIXUP_ALPHA > 0:
            # Sample lambda from Beta distribution
            lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)

            # Random permutation of indices
            index = torch.randperm(batch_size).to(device)

            # Mix inputs
            mixed_images = lam * images + (1 - lam) * images[index, :]

            # Mix targets (works for both hard and soft labels)
            mixed_targets = lam * targets + (1 - lam) * targets[index, :]

            # Forward pass
            outputs = model(mixed_images)
            loss = criterion(outputs, mixed_targets)
        else:
            # Standard training
            outputs = model(images)
            loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The model to evaluate.
        loader (DataLoader): Validation data loader.
        criterion (nn.Module): Loss function.
        device (str): Device to evaluate on.

    Returns:
        tuple: (average_loss, macro_auc)
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)

            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid to logits to get probabilities
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    epoch_loss = running_loss / dataset_size

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute AUC
    auc_score = compute_auc(all_targets, all_preds)

    return epoch_loss, auc_score


class Trainer:
    """
    Manages the training lifecycle, including training loops, validation,
    learning rate scheduling, and early stopping.
    """

    def __init__(
        self, model, train_loader, val_loader, optimizer, scheduler, criterion, device
    ):
        """
        Args:
            model (nn.Module): The model to train.
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            optimizer (Optimizer): PyTorch optimizer.
            scheduler (LRScheduler): Learning rate scheduler (optional).
            criterion (nn.Module): Loss function.
            device (str): Device to run on.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

        # Early Stopping parameters
        self.patience = 5
        self.best_auc = 0.0
        self.best_model_wts = copy.deepcopy(model.state_dict())
        self.counter = 0

    def fit(self, num_epochs=Config.EPOCHS):
        """
        Runs the full training loop.

        Args:
            num_epochs (int): Maximum number of epochs to train.

        Returns:
            nn.Module: The model with the best validation weights loaded.
            list: History of training metrics.
        """
        history = {"train_loss": [], "val_loss": [], "val_auc": []}

        print(f"Starting training for {num_epochs} epochs on {self.device}...")

        start_time = time.time()

        for epoch in range(num_epochs):
            epoch_start = time.time()

            # Train
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.device,
                epoch,
            )

            # Validate
            val_loss, val_auc = validate(
                self.model, self.val_loader, self.criterion, self.device
            )

            # Update Scheduler
            if self.scheduler:
                # Step based on validation metric if ReduceLROnPlateau, else standard step
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # Logging
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Time: {epoch_time:.1f}s | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val AUC: {val_auc}"
            )

            # Store history
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)

            # Early Stopping Check
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.counter = 0
                print(f"Validation AUC improved to {val_auc}. Saving model...")
            else:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    print("Early stopping triggered.")
                    break

        total_time = time.time() - start_time
        print(f"Training complete in {total_time // 60:.0f}m {total_time % 60:.0f}s")
        print(f"Best Validation AUC: {self.best_auc}")

        # Load best model weights
        self.model.load_state_dict(self.best_model_wts)

        return self.model, history
