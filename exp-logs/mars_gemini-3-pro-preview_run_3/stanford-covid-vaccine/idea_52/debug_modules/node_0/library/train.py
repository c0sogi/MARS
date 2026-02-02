import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import set_seed, mcrmse_loss, calculate_metric
from library.data import get_dataloaders
from library.model import DeepStabilizedBiGRU


class Trainer:
    """
    Manages the training and validation lifecycle of the Deep Stabilized BiGRU model.
    """

    def __init__(self, model, train_loader, val_loader, epochs=None, device=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs if epochs is not None else Config.EPOCHS
        self.device = device if device is not None else Config.DEVICE

        # Move model to computation device
        self.model.to(self.device)

        # Optimizer: AdamW
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Cosine Annealing
        # T_max is set to total epochs as we step once per epoch
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # State tracking
        self.best_score = float("inf")
        self.patience_counter = 0

    def train_epoch(self):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for inputs, pair_indices, targets in self.train_loader:
            batch_size = inputs.size(0)

            # Move data to device
            inputs = inputs.to(self.device)
            pair_indices = pair_indices.to(self.device)
            targets = targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs, pair_indices)

            # Compute Loss (MCRMSE on all 5 targets)
            loss = mcrmse_loss(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient Clipping (Mandatory for stability)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            # Optimization step
            self.optimizer.step()

            # Accumulate loss (weighted by batch size for accurate mean)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self):
        """
        Runs validation using the strict competition metric.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, pair_indices, targets in self.val_loader:
                inputs = inputs.to(self.device)
                pair_indices = pair_indices.to(self.device)

                # Forward pass
                outputs = self.model(inputs, pair_indices)

                # Collect predictions and targets
                # Move to CPU immediately to save GPU memory
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.numpy())

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate Metric
        # calculate_metric handles slicing to seq_scored and selecting specific columns
        score = calculate_metric(all_preds, all_targets)
        return score

    def fit(self):
        """
        Executes the full training loop with early stopping.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(self.epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_score = self.validate()

            # Scheduler Step
            self.scheduler.step()

            elapsed = time.time() - start_time

            # Print metrics (Full precision for validation score)
            print(
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val MCRMSE: {val_score} | "
                f"Time: {elapsed:.2f}s"
            )

            # Checkpointing & Early Stopping
            if val_score < self.best_score:
                self.best_score = val_score
                self.patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"Saved Best Model: {Config.MODEL_SAVE_PATH}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= Config.PATIENCE:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

        print(f"Training finished. Best Validation Score: {self.best_score}")


def run_training(load_cached_data=True):
    """
    Main entry point for training the model.

    Args:
        load_cached_data (bool): Whether to load preprocessed data from cache.
    """
    # 1. Set Reproducibility
    set_seed(Config.SEED)

    # 2. Prepare Data
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=load_cached_data)

    # 3. Initialize Model
    model = DeepStabilizedBiGRU()

    # 4. Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader)

    # 5. Execute Training
    trainer.fit()
