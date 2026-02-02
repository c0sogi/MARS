import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import AverageMeter, do_kaggle_metric, get_logger
from library.losses import CombinedLoss


class Trainer:
    """
    Manages the training and validation lifecycle of the Salt Segmentation model.
    Enforces strict FP32 training and handles multi-task outputs.
    """

    def __init__(self, model, train_loader, val_loader, device):
        """
        Args:
            model: The PyTorch model (ResNet34WideLinkNet).
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            device: Compute device (cuda/cpu).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function: Combined BCE + Lovasz + Auxiliary MSE
        self.criterion = CombinedLoss()

        # Optimizer: AdamW with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )

        # Scheduler: Cosine Annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=1e-6
        )

        self.logger = get_logger(Config.WORKING_DIR)
        self.best_map = 0.0

    def train_one_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        losses = AverageMeter()

        for i, (images, masks, depths, ids) in enumerate(self.train_loader):
            # Move inputs to device and ensure FP32
            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.float32)
            depths = depths.to(self.device, dtype=torch.float32)

            # Forward pass
            # Model returns tuple: (logits, aux_pred)
            outputs = self.model(images, depths)

            # Calculate combined loss
            # Criterion handles tuple unpacking internally
            loss = self.criterion(outputs, masks, depths)

            # Backward pass
            # Strict FP32: No GradScaler used
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            losses.update(loss.item(), images.size(0))

        return losses.avg

    def validate(self, epoch):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        losses = AverageMeter()
        maps = AverageMeter()

        with torch.no_grad():
            for images, masks, depths, ids in self.val_loader:
                images = images.to(self.device, dtype=torch.float32)
                masks = masks.to(self.device, dtype=torch.float32)
                depths = depths.to(self.device, dtype=torch.float32)

                # Forward pass
                outputs = self.model(images, depths)

                # Calculate validation loss
                loss = self.criterion(outputs, masks, depths)
                losses.update(loss.item(), images.size(0))

                # Extract logits for metric calculation
                if isinstance(outputs, (tuple, list)):
                    logits = outputs[0]
                else:
                    logits = outputs

                # Apply sigmoid to get probabilities [0, 1]
                probs = torch.sigmoid(logits)

                # Calculate mAP (Mean Average Precision)
                # do_kaggle_metric expects (predictions, truth)
                batch_map = do_kaggle_metric(probs, masks, threshold=0.5)
                maps.update(batch_map, images.size(0))

        return losses.avg, maps.avg

    def fit(self):
        """
        Main training loop with Early Stopping and Best Model Checkpointing.
        """
        self.logger.info(f"Starting training on device: {self.device}")
        self.logger.info(f"Epochs: {Config.EPOCHS} | Batch Size: {Config.BATCH_SIZE}")

        # Early stopping parameters
        patience = 15
        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_map = self.validate(epoch)

            # Update Scheduler
            self.scheduler.step()

            duration = time.time() - start_time

            # Log metrics with full precision
            self.logger.info(
                f"Epoch [{epoch+1}/{Config.EPOCHS}] "
                f"Time: {duration:.1f}s | "
                f"Train Loss: {train_loss:.8f} | "
                f"Val Loss: {val_loss:.8f} | "
                f"Val mAP: {val_map:.10f}"
            )

            # Save Best Model
            if val_map >= self.best_map:
                self.best_map = val_map
                patience_counter = 0
                save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                self.logger.info(f"New best model saved with mAP: {self.best_map:.10f}")
            else:
                patience_counter += 1

            # Early Stopping Check
            if patience_counter >= patience:
                self.logger.info(
                    f"Early stopping triggered at epoch {epoch+1} (No improvement for {patience} epochs)."
                )
                break

        self.logger.info(
            f"Training complete. Best Validation mAP: {self.best_map:.10f}"
        )
