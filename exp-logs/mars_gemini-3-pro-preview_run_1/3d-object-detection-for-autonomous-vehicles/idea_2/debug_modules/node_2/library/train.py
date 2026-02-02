import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import get_logger, set_seed
from library.dataset import BEVDataset, worker_init_fn
from library.model import DLASeg
from library.loss import CenterNetLoss


class Trainer:
    def __init__(self, sample_size=None):
        """
        Initialize the Trainer.
        Args:
            sample_size (int, optional): Limit the dataset size for debugging purposes.
        """
        # 1. Setup
        set_seed(Config.SEED)
        self.logger = get_logger()
        self.device = torch.device(Config.DEVICE)
        self.logger.info(f"Initializing Trainer on device: {self.device}")

        # 2. Data Loading
        self.logger.info("Loading datasets...")
        self.train_dataset = BEVDataset(
            split="train", load_cached_data=True, sample_size=sample_size
        )
        self.val_dataset = BEVDataset(
            split="val", load_cached_data=True, sample_size=sample_size
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
            drop_last=True,  # Drop last to maintain consistent batch size for OneCycleLR
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            worker_init_fn=worker_init_fn,
            pin_memory=True,
        )

        self.logger.info(
            f"Data loaded. Train steps: {len(self.train_loader)}, Val steps: {len(self.val_loader)}"
        )

        # 3. Model & Loss
        self.logger.info(f"Building model: {Config.BACKBONE}...")
        self.model = DLASeg().to(self.device)
        self.criterion = CenterNetLoss()

        # 4. Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # OneCycleLR Scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LEARNING_RATE,
            steps_per_epoch=len(self.train_loader),
            epochs=Config.NUM_EPOCHS,
            pct_start=0.3,
            div_factor=10,
            final_div_factor=1000,
        )

        # 5. State Management
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train(self):
        """
        Main training loop with validation and early stopping.
        """
        self.logger.info("Starting training...")
        start_time = time.time()

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            epoch_start = time.time()

            # --- Training Step ---
            train_metrics = self.train_one_epoch(epoch)

            # --- Validation Step ---
            val_metrics = self.validate(epoch)
            val_loss = val_metrics["total_loss"]

            # --- Logging ---
            epoch_duration = time.time() - epoch_start
            self.logger.info(
                f"Epoch {epoch}/{Config.NUM_EPOCHS} completed in {epoch_duration:.2f}s"
            )
            self.logger.info(f"Train Metrics: {train_metrics}")
            self.logger.info(f"Val Metrics: {val_metrics}")

            # --- Checkpointing & Early Stopping ---
            if val_loss < self.best_val_loss:
                self.logger.info(
                    f"Validation loss improved from {self.best_val_loss} to {val_loss}. Saving model..."
                )
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model()
            else:
                self.patience_counter += 1
                self.logger.info(
                    f"Validation loss did not improve. Patience: {self.patience_counter}/{Config.PATIENCE}"
                )

            if self.patience_counter >= Config.PATIENCE:
                self.logger.info("Early stopping triggered. Stopping training.")
                break

        total_time = time.time() - start_time
        self.logger.info(f"Training finished in {total_time:.2f}s")
        self.logger.info(f"Best Validation Loss: {self.best_val_loss}")

    def train_one_epoch(self, epoch):
        """
        Trains the model for one epoch.
        """
        self.model.train()
        metrics_accum = {}
        count = 0

        for batch in self.train_loader:
            # Move data to device
            inputs = batch["input"].to(self.device)
            # Filter out non-tensor items if any, and move targets to device
            targets = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k != "input" and isinstance(v, torch.Tensor)
            }

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)

            # Compute loss
            loss, stats = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Accumulate metrics
            count += 1
            for k, v in stats.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + v

        # Average metrics
        avg_metrics = {k: v / count for k, v in metrics_accum.items()}
        return avg_metrics

    def validate(self, epoch):
        """
        Validates the model on the validation set.
        """
        self.model.eval()
        metrics_accum = {}
        count = 0

        with torch.no_grad():
            for batch in self.val_loader:
                inputs = batch["input"].to(self.device)
                targets = {
                    k: v.to(self.device)
                    for k, v in batch.items()
                    if k != "input" and isinstance(v, torch.Tensor)
                }

                # Forward pass
                outputs = self.model(inputs)

                # Compute loss
                loss, stats = self.criterion(outputs, targets)

                # Accumulate metrics
                count += 1
                for k, v in stats.items():
                    metrics_accum[k] = metrics_accum.get(k, 0.0) + v

        # Average metrics
        avg_metrics = {k: v / count for k, v in metrics_accum.items()}
        return avg_metrics

    def save_model(self):
        """
        Saves the model state dictionary.
        """
        os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
        torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
