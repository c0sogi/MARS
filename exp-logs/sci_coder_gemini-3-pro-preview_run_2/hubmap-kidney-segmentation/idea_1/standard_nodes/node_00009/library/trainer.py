import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.losses import BCEDiceLoss
from library.utils import dice_coef


class Trainer:
    """
    Trainer class to handle model training, validation, and checkpointing.
    """

    def __init__(self, model: nn.Module, device: str = Config.DEVICE):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            device (str): Device to run training on ('cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.device = device

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # Loss Function
        self.criterion = BCEDiceLoss()

        # Mixed Precision Scaler
        self.scaler = GradScaler()

        # Checkpointing
        self.best_score = -np.inf
        self.checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    def train_one_epoch(self, dataloader):
        """
        Runs one epoch of training.

        Args:
            dataloader (DataLoader): Training dataloader.

        Returns:
            float: Average loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for images, masks, _ in dataloader:
            batch_size = images.size(0)

            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.float32)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            # Backward Pass and Optimizer Step
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate_one_epoch(self, dataloader):
        """
        Runs validation on the validation set.

        Args:
            dataloader (DataLoader): Validation dataloader.

        Returns:
            float: Average Dice score for the epoch.
        """
        self.model.eval()
        running_dice = 0.0
        dataset_size = 0

        with torch.no_grad():
            for images, masks, _ in dataloader:
                batch_size = images.size(0)

                images = images.to(self.device, dtype=torch.float32)
                masks = masks.to(self.device, dtype=torch.float32)

                # Inference (no autocast needed for eval usually, but consistent behavior is good)
                with autocast():
                    outputs = self.model(images)

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs)

                # Convert probabilities to binary predictions based on threshold
                preds = (probs > Config.THRESHOLD).float()

                # Calculate Dice Score
                # dice_coef expects tensors or numpy arrays
                score = dice_coef(preds, masks)

                running_dice += score * batch_size
                dataset_size += batch_size

        epoch_dice = running_dice / dataset_size
        return epoch_dice

    def fit(self, train_loader, val_loader, epochs=Config.NUM_EPOCHS, patience=5):
        """
        Main training loop with Early Stopping.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            epochs (int): Maximum number of epochs.
            patience (int): Early stopping patience (epochs without improvement).
        """
        print(f"Starting training on device: {self.device}")
        print(
            f"Epochs: {epochs}, Batch Size: {Config.BATCH_SIZE}, LR: {Config.LEARNING_RATE}"
        )

        early_stopping_counter = 0

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_one_epoch(train_loader)

            # Validate
            val_dice = self.validate_one_epoch(val_loader)

            # Update Scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            # Print Metrics
            print(
                f"Epoch {epoch}/{epochs} | LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.6f} | Val Dice: {val_dice}"
            )

            # Checkpointing & Early Stopping
            if val_dice > self.best_score:
                print(
                    f"Validation Dice improved from {self.best_score} to {val_dice}. Saving model..."
                )
                self.best_score = val_dice
                torch.save(self.model.state_dict(), self.checkpoint_path)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                print(
                    f"No improvement. Early stopping counter: {early_stopping_counter}/{patience}"
                )

            if early_stopping_counter >= patience:
                print("Early stopping triggered. Training finished.")
                break

        print(f"Training complete. Best Val Dice: {self.best_score}")
        print(f"Best model saved to: {self.checkpoint_path}")

    def load_best_model(self):
        """
        Loads the best model weights from the checkpoint.
        """
        if os.path.exists(self.checkpoint_path):
            print(f"Loading best model from {self.checkpoint_path}")
            self.model.load_state_dict(
                torch.load(self.checkpoint_path, map_location=self.device)
            )
        else:
            print("No checkpoint found. Using current model weights.")
        return self.model
