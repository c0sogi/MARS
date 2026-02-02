import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, MCRMSELoss, calculate_metric
from library.data import get_dataloaders
from library.model import DeepResGLUBiGRU


class Trainer:
    """
    Manages the training, validation, and model checkpointing process
    for the Deep Residual High-Capacity GLU-BiGRU model.
    """

    def __init__(self, model, train_loader, val_loader, config):
        """
        Args:
            model (nn.Module): The neural network model.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            config (class): Configuration class with hyperparameters.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(config.DEVICE)

        # Move model to device
        self.model.to(self.device)

        # Optimization components
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.T_MAX, eta_min=config.ETA_MIN
        )

        self.criterion = MCRMSELoss()

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            # Move data to device
            inputs = batch["inputs"].to(self.device)
            pair_indices = batch["pair_indices"].to(self.device)
            targets = batch["targets"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(inputs, pair_indices)

            # Calculate loss (MCRMSE on all 5 targets)
            loss = self.criterion(preds, targets)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.MAX_GRAD_NORM
            )

            # Optimizer step
            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self):
        """
        Runs validation and calculates the competition metric.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch["inputs"].to(self.device)
                pair_indices = batch["pair_indices"].to(self.device)
                targets = batch["targets"]  # Keep on CPU for metric calc

                # Forward pass
                preds = self.model(inputs, pair_indices)

                # Move preds to CPU
                preds_cpu = preds.detach().cpu().numpy()
                targets_cpu = targets.numpy()

                all_preds.append(preds_cpu)
                all_targets.append(targets_cpu)

        # Concatenate all batches
        if not all_preds:
            return float("inf")

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate metric using the utility function
        # This handles slicing to seq_scored and selecting specific columns
        score = calculate_metric(all_preds, all_targets)

        return score

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        best_score = float("inf")
        patience_counter = 0

        print(f"Starting training on {self.device}...")
        print(f"Epochs: {self.config.EPOCHS}, Batch Size: {self.config.BATCH_SIZE}")

        for epoch in range(self.config.EPOCHS):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_score = self.validate()

            # Scheduler Step
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            elapsed = time.time() - start_time

            # Print metrics
            print(
                f"Epoch {epoch + 1}/{self.config.EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val MCRMSE: {val_score} | "  # Full precision
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed:.2f}s"
            )

            # Early Stopping and Checkpointing
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
                print(f"  >>> New Best Model Saved (Score: {best_score})")
            else:
                patience_counter += 1
                print(f"  >>> Patience: {patience_counter}/{self.config.PATIENCE}")

            if patience_counter >= self.config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Score: {best_score}")


def run_training():
    """
    Sets up the environment, loads data, and runs the training process.
    """
    # 1. Reproducibility
    seed_everything(Config.SEED)

    # 2. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing Model...")
    model = DeepResGLUBiGRU()

    # 4. Trainer Initialization
    trainer = Trainer(model, train_loader, val_loader, Config)

    # 5. Start Training
    trainer.fit()
