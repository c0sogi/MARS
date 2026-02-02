import torch
import torch.optim as optim
import numpy as np
import os
import time
from library.config import Config
from library.model import HR1DResNet
from library.loss import MultiResolutionMAELoss


class Trainer:
    def __init__(self, model, device=None):
        """
        Trainer for the HR-1D-ResNet model.

        Args:
            model (torch.nn.Module): The model to train.
            device (torch.device, optional): Device to run on. Defaults to Config.DEVICE.
        """
        self.device = device if device else torch.device(Config.DEVICE)
        self.model = model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
        )

        # Loss Function
        self.criterion = MultiResolutionMAELoss()

        # Best metric for early stopping
        self.best_val_loss = float("inf")

    def train_one_epoch(self, dataloader, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch in dataloader:
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass (returns list of outputs for deep supervision)
            preds = self.model(features)

            # Compute loss
            loss = self.criterion(preds, targets, mask)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.GRAD_CLIP_NORM
            )

            self.optimizer.step()

            running_loss += loss.item() * features.size(0)
            count += features.size(0)

        avg_loss = running_loss / count if count > 0 else 0.0
        return avg_loss

    def evaluate(self, dataloader):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device)
                targets = batch["targets"].to(self.device)
                mask = batch["mask"].to(self.device)

                # Forward pass
                preds = self.model(features)

                # Compute loss
                loss = self.criterion(preds, targets, mask)

                running_loss += loss.item() * features.size(0)
                count += features.size(0)

        avg_loss = running_loss / count if count > 0 else 0.0
        return avg_loss

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=Config.EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
        save_path="best_model.pth",
    ):
        """
        Main training loop with early stopping.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            epochs (int): Maximum number of epochs.
            patience (int): Early stopping patience.
            save_path (str): Filename to save the best model.
        """
        print(f"Starting training on device: {self.device}")

        # Ensure working directory exists for saving model
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        full_save_path = os.path.join(Config.WORKING_DIR, save_path)

        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(train_loader, epoch)

            # Validate
            val_loss = self.evaluate(val_loader)

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.2f}s | LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f}"
            )

            # Early Stopping and Checkpointing
            if val_loss < self.best_val_loss:
                print(
                    f"Validation loss improved from {self.best_val_loss:.10f} to {val_loss:.10f}. Saving model..."
                )
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), full_save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"Validation loss did not improve. Patience: {patience_counter}/{patience}"
                )

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Loss: {self.best_val_loss:.10f}")

        # Load best model weights
        self.model.load_state_dict(torch.load(full_save_path, map_location=self.device))
        return self.model
