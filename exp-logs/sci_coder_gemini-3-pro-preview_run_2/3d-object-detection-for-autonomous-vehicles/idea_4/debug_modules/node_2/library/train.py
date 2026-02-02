import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.model import MonoCenterNet
from library.dataset import Mono3DDataset
from library.loss import Mono3DLoss


class Trainer:
    def __init__(
        self, debug=False, load_cached_data=True, batch_size=None, learning_rate=None
    ):
        """
        Initialize the Trainer with model, data, and optimizer.
        """
        # Setup environment
        Config.setup()
        Config.set_seed()
        self.device = Config.DEVICE
        self.debug = debug

        # Hyperparameters
        self.batch_size = batch_size if batch_size is not None else Config.BATCH_SIZE
        self.lr = learning_rate if learning_rate is not None else Config.LEARNING_RATE

        print(
            f"Initializing Trainer (Device: {self.device}, Batch Size: {self.batch_size}, LR: {self.lr})"
        )

        # Data
        self.train_dataset = Mono3DDataset(
            split="train", load_cached_data=load_cached_data, debug=debug
        )
        self.val_dataset = Mono3DDataset(
            split="val", load_cached_data=load_cached_data, debug=debug
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        # Model
        self.model = MonoCenterNet().to(self.device)

        # Loss
        self.criterion = Mono3DLoss()

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.1
        )

    def train_epoch(self, epoch):
        """
        Run one epoch of training.
        """
        self.model.train()
        running_metrics = {}
        count = 0

        for batch_idx, (images, targets, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            # Move targets to device
            for k, v in targets.items():
                targets[k] = v.to(self.device)

            self.optimizer.zero_grad()

            # Forward
            outputs = self.model(images)
            loss, stats = self.criterion(outputs, targets)

            # Backward
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            count += 1
            for k, v in stats.items():
                val = v.item()
                running_metrics[k] = running_metrics.get(k, 0.0) + val

        # Average metrics
        for k in running_metrics:
            running_metrics[k] /= count

        return running_metrics

    def evaluate(self):
        """
        Run evaluation on the validation set.
        """
        self.model.eval()
        running_metrics = {}
        count = 0

        with torch.no_grad():
            for batch_idx, (images, targets, _) in enumerate(self.val_loader):
                images = images.to(self.device)
                for k, v in targets.items():
                    targets[k] = v.to(self.device)

                # Forward
                outputs = self.model(images)
                loss, stats = self.criterion(outputs, targets)

                # Accumulate metrics
                count += 1
                for k, v in stats.items():
                    val = v.item()
                    running_metrics[k] = running_metrics.get(k, 0.0) + val

        # Average metrics
        for k in running_metrics:
            running_metrics[k] /= count

        return running_metrics

    def save_checkpoint(self, filename):
        """
        Save model checkpoint.
        """
        path = os.path.join(Config.CHECKPOINT_DIR, filename)
        torch.save(self.model.state_dict(), path)

    def fit(self, num_epochs=None):
        """
        Main training loop with Early Stopping.
        """
        if num_epochs is None:
            num_epochs = Config.NUM_EPOCHS

        best_val_loss = float("inf")
        patience = 3
        patience_counter = 0

        print(f"\nStarting training for {num_epochs} epochs...")

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # 1. Train
            train_metrics = self.train_epoch(epoch)

            # 2. Validate
            val_metrics = self.evaluate()

            # 3. Scheduler Step
            self.scheduler.step()

            # 4. Logging
            duration = time.time() - start_time
            print(f"Epoch {epoch}/{num_epochs} | Time: {duration:.2f}s")
            print(
                f"  Train Loss: {train_metrics['loss']:.8f} | HM: {train_metrics['hm_loss']:.4f} | Dim: {train_metrics['dim_loss']:.4f} | Depth: {train_metrics['depth_loss']:.4f}"
            )
            print(
                f"  Val Loss:   {val_metrics['loss']:.8f} | HM: {val_metrics['hm_loss']:.4f} | Dim: {val_metrics['dim_loss']:.4f} | Depth: {val_metrics['depth_loss']:.4f}"
            )

            # 5. Early Stopping & Checkpointing
            current_val_loss = val_metrics["loss"]

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                self.save_checkpoint("best_model.pth")
                print("  -> Best model saved.")
            else:
                patience_counter += 1
                print(f"  -> EarlyStopping counter: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

            # Save latest checkpoint every epoch
            self.save_checkpoint("latest_model.pth")


def train_model(debug=False, load_cached_data=True, num_epochs=None, batch_size=None):
    """
    Wrapper function to instantiate Trainer and run training.
    """
    trainer = Trainer(
        debug=debug, load_cached_data=load_cached_data, batch_size=batch_size
    )
    trainer.fit(num_epochs=num_epochs)
