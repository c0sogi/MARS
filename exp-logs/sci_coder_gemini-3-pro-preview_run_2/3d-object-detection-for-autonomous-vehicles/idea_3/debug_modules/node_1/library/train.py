import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import time
from library.config import Config
from library.dataset import LidarDataset
from library.model import BevYolo
from library.loss import YoloLoss


class Trainer:
    """
    Trainer class for the Rasterized BEV-YOLO 3D Object Detection model.
    """

    def __init__(self, learning_rate=None, weight_decay=None, device=None):
        """
        Initialize the Trainer.

        Args:
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to Config.LEARNING_RATE.
            weight_decay (float, optional): Weight decay for the optimizer. Defaults to Config.WEIGHT_DECAY.
            device (torch.device, optional): Device to run training on. Defaults to Config.DEVICE.
        """
        self.device = device if device else Config.DEVICE
        self.lr = learning_rate if learning_rate is not None else Config.LEARNING_RATE
        self.wd = weight_decay if weight_decay is not None else Config.WEIGHT_DECAY

        # Initialize Model
        self.model = BevYolo().to(self.device)

        # Initialize Loss
        self.criterion = YoloLoss().to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd
        )

        # Scheduler (Optional, but good practice)
        # Using a simple StepLR or similar could be added, but relying on Adam is often sufficient for baseline.
        # We will stick to constant LR as per basic config, or implement warmup if needed.
        # For this implementation, we'll keep it simple.

    def train_epoch(self, dataloader, epoch_idx):
        """
        Run a single training epoch.
        """
        self.model.train()
        running_loss = 0.0
        running_metrics = {"loss_obj": 0.0, "loss_reg": 0.0, "loss_cls": 0.0}
        num_batches = 0

        for batch_idx, (bev, targets, _) in enumerate(dataloader):
            # Move to device
            bev = bev.to(self.device)
            targets = targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            predictions = self.model(bev)

            # Compute loss
            loss, metrics = self.criterion(predictions, targets)

            # Backward pass
            loss.backward()

            # Optimizer step
            self.optimizer.step()

            # Accumulate metrics
            running_loss += loss.item()
            for k, v in metrics.items():
                if k in running_metrics:
                    running_metrics[k] += v

            num_batches += 1

        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        avg_metrics = (
            {k: v / num_batches for k, v in running_metrics.items()}
            if num_batches > 0
            else running_metrics
        )

        return avg_loss, avg_metrics

    def validate(self, dataloader):
        """
        Run validation loop.
        """
        self.model.eval()
        running_loss = 0.0
        running_metrics = {"loss_obj": 0.0, "loss_reg": 0.0, "loss_cls": 0.0}
        num_batches = 0

        with torch.no_grad():
            for bev, targets, _ in dataloader:
                bev = bev.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(bev)
                loss, metrics = self.criterion(predictions, targets)

                running_loss += loss.item()
                for k, v in metrics.items():
                    if k in running_metrics:
                        running_metrics[k] += v

                num_batches += 1

        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        avg_metrics = (
            {k: v / num_batches for k, v in running_metrics.items()}
            if num_batches > 0
            else running_metrics
        )

        return avg_loss, avg_metrics

    def fit(self, num_epochs=None, batch_size=None, debug=False, load_cached_data=True):
        """
        Main training loop with Early Stopping.

        Args:
            num_epochs (int, optional): Number of epochs to train. Defaults to Config.NUM_EPOCHS.
            batch_size (int, optional): Batch size. Defaults to Config.BATCH_SIZE.
            debug (bool): If True, use a small subset of data for debugging.
            load_cached_data (bool): Whether to use cached calibration/metadata.
        """
        # Set seeds for reproducibility
        Config.set_seed(Config.SEED)

        epochs = num_epochs if num_epochs is not None else Config.NUM_EPOCHS
        bs = batch_size if batch_size is not None else Config.BATCH_SIZE

        print(f"Starting training on device: {self.device}")
        print(f"Epochs: {epochs}, Batch Size: {bs}, Debug Mode: {debug}")

        # 1. Prepare Datasets
        train_dataset = LidarDataset(split="train", load_cached_data=load_cached_data)
        val_dataset = LidarDataset(split="val", load_cached_data=load_cached_data)

        if debug:
            # Use small subset
            train_indices = list(range(min(len(train_dataset), 50)))
            val_indices = list(range(min(len(val_dataset), 20)))
            train_dataset = Subset(train_dataset, train_indices)
            val_dataset = Subset(val_dataset, val_indices)
            print(
                f"Debug mode: Train size={len(train_dataset)}, Val size={len(val_dataset)}"
            )
        else:
            print(f"Train size={len(train_dataset)}, Val size={len(val_dataset)}")

        # 2. Prepare Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # 3. Training Loop
        best_val_loss = float("inf")
        patience_counter = 0
        patience_limit = Config.PATIENCE

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_loss, train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_metrics = self.validate(val_loader)

            elapsed = time.time() - start_time

            # Print Metrics (Full Precision)
            print(f"Epoch {epoch}/{epochs} | Time: {elapsed:.2f}s")
            print(
                f"  Train Loss: {train_loss} | Obj: {train_metrics['loss_obj']} | Reg: {train_metrics['loss_reg']} | Cls: {train_metrics['loss_cls']}"
            )
            print(
                f"  Val Loss:   {val_loss} | Obj: {val_metrics['loss_obj']} | Reg: {val_metrics['loss_reg']} | Cls: {val_metrics['loss_cls']}"
            )

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save Best Model
                save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"  New best model saved to {save_path}")
            else:
                patience_counter += 1
                print(
                    f"  EarlyStopping counter: {patience_counter} out of {patience_limit}"
                )
                if patience_counter >= patience_limit:
                    print("Early stopping triggered.")
                    break

        print("Training complete.")


def train_model(num_epochs=None, batch_size=None, debug=False):
    """
    Wrapper function to instantiate Trainer and start training.
    """
    trainer = Trainer()
    trainer.fit(num_epochs=num_epochs, batch_size=batch_size, debug=debug)
