import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import set_seed
from library.model import SD_DGN
from library.losses import CascadedLoss


class Trainer:
    """
    Trainer class for the Stochastic-Depth Decoupled-Gated Network (SD-DGN).
    Manages training, validation, early stopping, and checkpointing.
    """

    def __init__(self, train_loader, val_loader):
        """
        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
        """
        # Ensure reproducibility
        set_seed(Config.SEED)

        self.device = torch.device(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Initialize Model
        self.model = SD_DGN().to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize Loss Function
        self.criterion = CascadedLoss().to(self.device)

        # Checkpoint Path
        self.checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        Returns:
            avg_loss (float): Average training loss.
            avg_acc (float): Average training accuracy (Stage 3).
        """
        self.model.train()
        running_loss = 0.0
        correct_preds = 0
        total_frames = 0

        for batch_idx, (features, targets) in enumerate(self.train_loader):
            features = features.to(self.device)
            targets = targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass (returns tuple of logits for 3 stages)
            outputs = self.model(features)

            # Calculate loss (CascadedLoss handles tuple unpacking)
            loss, loss_dict = self.criterion(outputs, targets)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()

            # Calculate Accuracy on Stage 3 (Final Refinement)
            # outputs[2] is logits3: [Batch, Classes, Time]
            logits3 = outputs[2]
            preds = torch.argmax(logits3, dim=1)  # [Batch, Time]

            correct_preds += (preds == targets).sum().item()
            total_frames += targets.numel()

        avg_loss = running_loss / len(self.train_loader)
        avg_acc = correct_preds / total_frames if total_frames > 0 else 0.0

        return avg_loss, avg_acc

    def validate(self):
        """
        Runs validation on the validation set.
        Returns:
            avg_loss (float): Average validation loss.
            avg_acc (float): Average validation accuracy (Stage 3).
        """
        self.model.eval()
        running_loss = 0.0
        correct_preds = 0
        total_frames = 0

        with torch.no_grad():
            for features, targets in self.val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = self.model(features)

                # Calculate loss
                loss, _ = self.criterion(outputs, targets)

                running_loss += loss.item()

                # Calculate Accuracy on Stage 3
                logits3 = outputs[2]
                preds = torch.argmax(logits3, dim=1)

                correct_preds += (preds == targets).sum().item()
                total_frames += targets.numel()

        avg_loss = running_loss / len(self.val_loader)
        avg_acc = correct_preds / total_frames if total_frames > 0 else 0.0

        return avg_loss, avg_acc

    def fit(self, epochs=Config.EPOCHS):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate()

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch} | Train Loss: {train_loss} | Train Acc: {train_acc} | Val Loss: {val_loss} | Val Acc: {val_acc}"
            )

            # Early Stopping & Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(
                    f"Validation loss improved. Model saved to {self.checkpoint_path}"
                )
            else:
                patience_counter += 1
                print(
                    f"Validation loss did not improve. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

        print("Training complete.")
