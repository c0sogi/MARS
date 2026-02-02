import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time

from library.config import Config
from library.data import get_datasets, CollateFn
from library.model import SIRDS_SP
from library.utils import set_seed, MetricTracker


class Trainer:
    """
    Manages the training lifecycle of the SI-RDS-SP model.
    """

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

        # Optimizer with weight decay for regularization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler to reduce LR when validation metric plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            min_lr=Config.MIN_LR,
        )

        # Loss function: MSE on log-transformed targets aligns with RMSLE
        self.criterion = nn.MSELoss()

        # Early stopping state
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        """Runs one epoch of training."""
        self.model.train()
        tracker = MetricTracker()

        for batch in train_loader:
            # Move batch data to device
            atom_features = batch["atom_features"].to(self.device)
            batch_indices = batch["batch_indices"].to(self.device)
            global_features = batch["global_features"].to(self.device)
            spacegroups = batch["spacegroups"].to(self.device)
            targets = batch["targets"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            preds = self.model(
                atom_features, batch_indices, global_features, spacegroups
            )

            # Compute loss
            loss = self.criterion(preds, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Update statistics
            batch_size = targets.size(0)
            tracker.update(loss.item(), batch_size)

        return tracker.avg

    def validate(self, val_loader):
        """Evaluates the model on the validation set."""
        self.model.eval()
        tracker = MetricTracker()

        with torch.no_grad():
            for batch in val_loader:
                atom_features = batch["atom_features"].to(self.device)
                batch_indices = batch["batch_indices"].to(self.device)
                global_features = batch["global_features"].to(self.device)
                spacegroups = batch["spacegroups"].to(self.device)
                targets = batch["targets"].to(self.device)

                preds = self.model(
                    atom_features, batch_indices, global_features, spacegroups
                )
                loss = self.criterion(preds, targets)

                batch_size = targets.size(0)
                tracker.update(loss.item(), batch_size)

        return tracker.avg

    def fit(self, train_loader, val_loader):
        """Main training loop with early stopping."""
        print(f"Starting training on {self.device}...")
        start_time = time.time()

        for epoch in range(1, Config.EPOCHS + 1):
            epoch_start = time.time()

            # Train and Validate
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            # Update scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step(val_loss)

            # Calculate RMSLE (approximate, since loss is MSE on log targets)
            train_rmsle = np.sqrt(train_loss)
            val_rmsle = np.sqrt(val_loss)

            epoch_time = time.time() - epoch_start

            print(
                f"Epoch {epoch:03d} | "
                f"Train MSE: {train_loss:.6f} (RMSLE: {train_rmsle:.6f}) | "
                f"Val MSE: {val_loss:.6f} (RMSLE: {val_rmsle:.6f}) | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.2f}s"
            )

            # Early Stopping and Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"  >>> New best model saved! (Val MSE: {val_loss:.6f})")
            else:
                self.patience_counter += 1
                print(
                    f"  >>> Early stopping counter: {self.patience_counter}/{Config.PATIENCE}"
                )

            if self.patience_counter >= Config.PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time/60:.2f} minutes.")
        print(f"Best Validation MSE: {self.best_val_loss:.6f}")
        print(f"Best Validation RMSLE: {np.sqrt(self.best_val_loss):.6f}")


def run_training():
    """
    Orchestrates the data loading, model initialization, and training process.
    """
    # 1. Reproducibility
    set_seed(Config.SEED)

    # 2. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Data Preparation
    # load_cached_data=True will try to load .npz files from working dir
    train_dataset, val_dataset, _ = get_datasets(load_cached_data=True)

    collate_fn = CollateFn()

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )

    # 4. Model Initialization
    model = SIRDS_SP()

    # 5. Training
    trainer = Trainer(model, device)
    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    run_training()
