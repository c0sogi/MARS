import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time

from library.config import Config
from library.dataset import load_data
from library.model import (
    ScalarAggregatedBiGRU,
    set_seed,
    masked_mse_loss,
    compute_mcrmse,
)


class Trainer:
    """
    Manages the training, validation, and checkpointing of the RNA degradation model.
    """

    def __init__(self, config=Config):
        self.config = config
        self.device = config.DEVICE

        # Ensure reproducibility
        set_seed(config.SEED)

        # Initialize Model
        self.model = ScalarAggregatedBiGRU(config).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Scheduler (ReduceLROnPlateau)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            min_lr=config.MIN_LR,
        )

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0

        for batch in train_loader:
            # Move data to device
            sequences = batch["sequence"].to(self.device)
            loop_types = batch["loop_type"].to(self.device)
            pair_dists = batch["pair_dist"].to(self.device)
            targets = batch["targets"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(sequences, loop_types, pair_dists)

            # Compute Loss
            loss = masked_mse_loss(outputs, targets, mask)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.MAX_GRAD_NORM
            )

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        """
        Runs validation and computes MCRMSE.
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        all_masks = []

        with torch.no_grad():
            for batch in val_loader:
                sequences = batch["sequence"].to(self.device)
                loop_types = batch["loop_type"].to(self.device)
                pair_dists = batch["pair_dist"].to(self.device)
                targets = batch["targets"].to(self.device)
                mask = batch["mask"].to(self.device)

                outputs = self.model(sequences, loop_types, pair_dists)

                all_preds.append(outputs)
                all_targets.append(targets)
                all_masks.append(mask)

        # Concatenate all batches
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        masks = torch.cat(all_masks, dim=0)

        # Compute Metric
        score = compute_mcrmse(preds, targets, masks)
        return score.item()

    def fit(self, debug=False, early_stopping_patience=10):
        """
        Main training loop with Early Stopping and Model Checkpointing.
        """
        print(f"Starting training on device: {self.device}")
        print(f"Debug Mode: {debug}")

        # 1. Load Data
        train_dataset = load_data("train", debug=debug)
        val_dataset = load_data("val", debug=debug)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        # Setup tracking variables
        best_score = float("inf")
        best_model_path = os.path.join(self.config.WORKING_DIR, "best_model.pth")
        patience_counter = 0

        # Ensure working directory exists
        os.makedirs(self.config.WORKING_DIR, exist_ok=True)

        start_time = time.time()

        for epoch in range(self.config.EPOCHS):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_score = self.validate(val_loader)

            # Scheduler Step
            self.scheduler.step(val_score)

            epoch_duration = time.time() - epoch_start

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch+1}/{self.config.EPOCHS} | "
                f"Time: {epoch_duration:.2f}s | "
                f"Train Loss: {train_loss:.8f} | "
                f"Val MCRMSE: {val_score:.16f}"
            )

            # Checkpoint & Early Stopping
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"  >>> New Best Model Saved! Score: {best_score:.16f}")
            else:
                patience_counter += 1
                print(
                    f"  ... No improvement. Patience: {patience_counter}/{early_stopping_patience}"
                )

            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break

        total_time = time.time() - start_time
        print(f"\nTraining finished in {total_time:.2f}s.")
        print(f"Best Validation MCRMSE: {best_score:.16f}")
        return best_score


def run_training(debug=Config.DEBUG, epochs=None):
    """
    Helper function to instantiate Trainer and run fit.
    Allows overriding epochs for quick testing.
    """
    config = Config
    if epochs is not None:
        # Create a temporary config modification if needed,
        # but Config is a class, so we modify it directly or subclass.
        # For simplicity, we just modify the class attribute temporarily or rely on Trainer using it.
        # However, cleaner is to just pass it or let Trainer use Config.
        # Since Config is a class with static attributes, we can modify it carefully.
        original_epochs = config.EPOCHS
        config.EPOCHS = epochs

    trainer = Trainer(config)
    best_score = trainer.fit(debug=debug)

    if epochs is not None:
        config.EPOCHS = original_epochs  # Restore

    return best_score
