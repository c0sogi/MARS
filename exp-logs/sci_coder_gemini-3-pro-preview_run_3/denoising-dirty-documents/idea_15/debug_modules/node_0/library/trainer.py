import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config


class ModelTrainer:
    """
    Manages the training, validation, and checkpointing of a single ResDnCNN model.
    """

    def __init__(self, model, device=None):
        """
        Initialize the trainer.

        Args:
            model (nn.Module): The neural network model to train.
            device (str, optional): Device to train on ('cuda' or 'cpu'). Defaults to Config.DEVICE.
        """
        self.device = device if device else Config.DEVICE
        self.model = model.to(self.device)

        # Loss function: MSE between predicted noise and actual noise
        # Since the model predicts the residual (noise), MSE on residuals is equivalent
        # to MSE on the reconstructed clean image.
        self.criterion = nn.MSELoss()

        # Optimizer: AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Cosine Annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.

        Args:
            train_loader (DataLoader): DataLoader for training data.

        Returns:
            float: Average training loss (MSE) for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        for noisy_imgs, residual_targets in train_loader:
            noisy_imgs = noisy_imgs.to(self.device)
            residual_targets = residual_targets.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            pred_residuals = self.model(noisy_imgs)

            # Compute loss
            loss = self.criterion(pred_residuals, residual_targets)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            if Config.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), Config.GRAD_CLIP
                )

            # Optimizer step
            self.optimizer.step()

            # Accumulate metrics
            # MSELoss with reduction='mean' averages over the batch.
            # We multiply by batch size to accumulate total loss, then divide by total samples.
            batch_size = noisy_imgs.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = running_loss / total_samples
        return avg_loss

    def validate(self, val_loader):
        """
        Runs validation on the validation set.

        Args:
            val_loader (DataLoader): DataLoader for validation data.

        Returns:
            tuple: (average_loss, average_rmse)
        """
        self.model.eval()
        running_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for noisy_imgs, residual_targets in val_loader:
                noisy_imgs = noisy_imgs.to(self.device)
                residual_targets = residual_targets.to(self.device)

                pred_residuals = self.model(noisy_imgs)

                loss = self.criterion(pred_residuals, residual_targets)

                batch_size = noisy_imgs.size(0)
                running_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = running_loss / total_samples
        # RMSE is simply the square root of MSE
        rmse = math.sqrt(avg_loss)
        return avg_loss, rmse

    def train(self, train_loader, val_loader, model_id=0):
        """
        Full training loop with Early Stopping and Scheduler.
        Saves the best model to Config.WORKING_DIR.

        Args:
            train_loader (DataLoader): Training dataloader.
            val_loader (DataLoader): Validation dataloader.
            model_id (int): Identifier for the model (used for saving unique checkpoints).

        Returns:
            float: Best validation loss achieved.
        """
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(Config.WORKING_DIR, f"model_{model_id}.pth")

        print(f"Model {model_id}: Starting training on device {self.device}")

        for epoch in range(Config.EPOCHS):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_rmse = self.validate(val_loader)

            # Step scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Print metrics with full precision
            print(
                f"Model {model_id} | Epoch {epoch+1}/{Config.EPOCHS} | "
                f"LR: {current_lr} | Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | Val RMSE: {val_rmse}"
            )

            # Checkpointing & Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(
                        f"Model {model_id}: Early stopping triggered at epoch {epoch+1}"
                    )
                    break

        print(f"Model {model_id}: Training finished. Best Val Loss: {best_val_loss}")

        # Load best weights to ensure model state is optimal if used immediately
        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )

        return best_val_loss
