import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from library.config import (
    DEVICE,
    MODEL_SAVE_PATH,
    PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
)
from library.utils import save_checkpoint, save_submission
from library.dataset import TargetTransformer


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Trains the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Training data loader.
        optimizer (Optimizer): The optimizer.
        criterion (Loss): The loss function.
        device (str): Device to run on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        # Unpack batch from CrystalCollate
        # batch structure: (atomic_features, batch_index, global_features, targets, ids)
        atomic_features = batch[0].to(device)
        batch_index = batch[1].to(device)
        global_features = batch[2].to(device)
        targets = batch[3].to(device)

        optimizer.zero_grad()

        outputs = model(atomic_features, batch_index, global_features)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        # Accumulate loss (MSE is mean, so multiply by batch size)
        running_loss += loss.item() * targets.size(0)
        total_samples += targets.size(0)

    epoch_loss = running_loss / total_samples
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The PyTorch model.
        dataloader (DataLoader): Validation data loader.
        criterion (Loss): The loss function.
        device (str): Device to run on.

    Returns:
        float: Average loss for the validation set.
    """
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            atomic_features = batch[0].to(device)
            batch_index = batch[1].to(device)
            global_features = batch[2].to(device)
            targets = batch[3].to(device)

            outputs = model(atomic_features, batch_index, global_features)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)

    epoch_loss = running_loss / total_samples
    return epoch_loss


class Trainer:
    """
    Manages the training lifecycle including optimization, scheduling, and early stopping.
    """

    def __init__(self, model, train_loader, val_loader, device=DEVICE):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function: MSE on log-transformed targets
        self.criterion = nn.MSELoss()

        # Optimizer: AdamW with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Scheduler: Reduce LR when validation loss stops improving
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

    def fit(self, epochs=EPOCHS, patience=PATIENCE):
        """
        Runs the training loop.

        Args:
            epochs (int): Maximum number of epochs.
            patience (int): Early stopping patience.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training on {self.device} for {epochs} epochs...")

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.device,
            )

            val_loss = validate(
                self.model, self.val_loader, self.criterion, self.device
            )

            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)

            # Calculate RMSLE (Root Mean Squared Logarithmic Error)
            # Since targets are already log(1+y), RMSE on these targets is RMSLE on original scale.
            train_rmsle = np.sqrt(train_loss)
            val_rmsle = np.sqrt(val_loss)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train MSE: {train_loss} | Train RMSLE: {train_rmsle} | "
                f"Val MSE: {val_loss} | Val RMSLE: {val_rmsle}"
            )

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(self.model, MODEL_SAVE_PATH)
                print(f"  New best model saved! Val Loss: {val_loss}")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Loss: {best_val_loss}")


def generate_submission(model, test_loader, device, save_path):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): Test data loader.
        device (str): Device to run on.
        save_path (str): Path to save the submission CSV.
    """
    model.eval()
    ids_all = []
    preds_all = []

    # Transformer to inverse the log1p operation
    target_transformer = TargetTransformer()

    print("Generating predictions for test set...")
    with torch.no_grad():
        for batch in test_loader:
            atomic_features = batch[0].to(device)
            batch_index = batch[1].to(device)
            global_features = batch[2].to(device)
            # batch[3] contains placeholder targets, ignore them
            ids = batch[4]

            outputs = model(atomic_features, batch_index, global_features)

            # Inverse transform: log(1+y) -> y
            # outputs are (B, 2)
            preds = target_transformer.inverse_transform(outputs)

            ids_all.extend(ids)
            preds_all.append(preds.cpu().numpy())

    # Concatenate all predictions
    preds_all = np.concatenate(preds_all, axis=0)

    # Separate columns
    formation_energy_preds = preds_all[:, 0]
    bandgap_energy_preds = preds_all[:, 1]

    # Save using utility function
    save_submission(ids_all, formation_energy_preds, bandgap_energy_preds, save_path)
    print(f"Submission saved to {save_path}")
