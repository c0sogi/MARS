import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import (
    set_seed,
    setup_logger,
    rle_encode_predictions,
    evaluate_predictions,
)
from library.data_loader import get_dataloaders
from library.model import RDKRN
from library.loss import CascadedLoss


class Trainer:
    """
    Manages the training lifecycle of the RD-KRN model.
    """

    def __init__(self, limit_samples=None, num_epochs=Config.NUM_EPOCHS):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs

        # Setup Logger
        self.logger = setup_logger(
            name="Trainer", log_file=os.path.join(Config.WORK_DIR, "training.log")
        )

        # Data Loaders
        self.train_loader, self.val_loader, _ = get_dataloaders(
            batch_size=Config.BATCH_SIZE, limit_samples=limit_samples
        )

        # Model
        self.model = RDKRN().to(self.device)

        # Loss
        self.criterion = CascadedLoss().to(self.device)

        # Optimizer (Adam as per requirements, no AdamW)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Early Stopping State
        self.patience = Config.PATIENCE
        self.best_val_score = float("inf")  # Lower LEV is better
        self.counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        start_time = time.time()

        for batch in self.train_loader:
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass (returns list of logits from 3 stages)
            outputs = self.model(features)

            # Compute Loss
            loss, _ = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Gradient Clipping (Optional but good for RNNs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            # Optimizer Step
            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        duration = time.time() - start_time

        return avg_loss, duration

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        num_batches = 0

        all_preds_seq = []
        all_targets_seq = []

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["features"].to(self.device)
                targets = batch["targets"].to(self.device)

                # Forward pass
                outputs = self.model(features)

                # Compute Loss
                loss, _ = self.criterion(outputs, targets)
                running_loss += loss.item()
                num_batches += 1

                # --- Compute LEV Score Metrics ---
                # Use Stage 3 outputs (index 2) for final prediction
                stage3_logits = outputs[2]

                # Get class indices: (Batch, Time)
                batch_preds = torch.argmax(stage3_logits, dim=2).cpu().numpy()
                batch_targets = targets.cpu().numpy()

                # Convert to sequences (RLE)
                for i in range(batch_preds.shape[0]):
                    pred_seq = rle_encode_predictions(batch_preds[i])
                    target_seq = rle_encode_predictions(batch_targets[i])

                    all_preds_seq.append(pred_seq)
                    all_targets_seq.append(target_seq)

        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

        # Compute Levenshtein Error Rate
        lev_score = evaluate_predictions(all_preds_seq, all_targets_seq)

        return avg_loss, lev_score

    def fit(self):
        self.logger.info(f"Starting training on device: {self.device}")
        self.logger.info(
            f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}"
        )

        for epoch in range(1, self.num_epochs + 1):
            # Train
            train_loss, train_time = self.train_epoch(epoch)

            # Validate
            val_loss, val_lev = self.validate()

            # Log
            self.logger.info(
                f"Epoch {epoch}/{self.num_epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val LEV: {val_lev:.6f} | "
                f"Time: {train_time:.2f}s"
            )

            # Checkpointing & Early Stopping
            # We prioritize LEV score (Error Rate) as the primary metric
            score = val_lev

            if score < self.best_val_score:
                self.best_val_score = score
                self.counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                self.logger.info(f"New best model saved with LEV: {score:.6f}")
            else:
                self.counter += 1
                self.logger.info(
                    f"No improvement. Counter: {self.counter}/{self.patience}"
                )

            if self.counter >= self.patience:
                self.logger.info("Early stopping triggered.")
                break

        self.logger.info(f"Training complete. Best Val LEV: {self.best_val_score:.6f}")


def run_training(limit_samples=None, num_epochs=Config.NUM_EPOCHS):
    """
    Entry point to run the training process.

    Args:
        limit_samples (int, optional): Limit dataset size for debugging.
        num_epochs (int): Maximum number of epochs.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    # Ensure directories exist
    Config.setup_directories()

    # Initialize and run trainer
    trainer = Trainer(limit_samples=limit_samples, num_epochs=num_epochs)
    trainer.fit()
