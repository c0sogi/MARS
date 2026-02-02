import os
import torch
import torch.nn as nn
from library.config import Config
from library.utils import MetricMonitor, dice_coeff


class Trainer:
    """
    Trainer class that encapsulates the training and validation loops,
    along with optimization, scheduling, checkpointing, and early stopping logic.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        device,
    ):
        """
        Args:
            model (nn.Module): The neural network model.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            optimizer (Optimizer): PyTorch optimizer.
            scheduler (LRScheduler): PyTorch learning rate scheduler.
            criterion (nn.Module): Loss function.
            device (str): Device to run training on ('cuda' or 'cpu').
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

        # Training State
        self.best_val_dice = -float("inf")
        self.patience_counter = 0
        self.top_k_checkpoints = []  # List of tuples: (dice_score, file_path)

    def train_one_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        metric_monitor = MetricMonitor()

        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images = images.to(self.device)
            masks = masks.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            metric_monitor.update("Loss", loss.item())

        return metric_monitor.metrics

    def validate(self):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        metric_monitor = MetricMonitor()

        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(self.val_loader):
                images = images.to(self.device)
                masks = masks.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

                # Calculate Dice Score
                # Apply sigmoid to convert logits to probabilities
                preds = torch.sigmoid(outputs)
                dice = dice_coeff(preds, masks)

                # Update metrics
                metric_monitor.update("Loss", loss.item())
                metric_monitor.update("Dice", dice.item())

        return metric_monitor.metrics

    def save_checkpoint(self, epoch, dice_score):
        """
        Saves a checkpoint and manages the Top-K checkpoints on disk.
        """
        filename = f"checkpoint_epoch_{epoch}_dice_{dice_score:.6f}.pth"
        save_path = os.path.join(Config.CHECKPOINTS_DIR, filename)

        # Save the current model state
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "dice_score": dice_score,
            },
            save_path,
        )

        # Update Top-K list
        self.top_k_checkpoints.append((dice_score, save_path))
        # Sort by Dice score descending (highest first)
        self.top_k_checkpoints.sort(key=lambda x: x[0], reverse=True)

        # Remove oldest/worst checkpoint if exceeding K
        if len(self.top_k_checkpoints) > Config.SAVE_TOP_K:
            # Pop the last element (lowest score)
            worst_ckpt = self.top_k_checkpoints.pop()
            path_to_remove = worst_ckpt[1]
            if os.path.exists(path_to_remove):
                try:
                    os.remove(path_to_remove)
                except OSError as e:
                    print(f"Error removing checkpoint {path_to_remove}: {e}")

    def fit(self):
        """
        Main training loop handling epochs, logging, checkpointing, and early stopping.
        """
        print(f"Starting training for {Config.EPOCHS} epochs on {self.device}...")

        for epoch in range(1, Config.EPOCHS + 1):
            # 1. Train
            train_metrics = self.train_one_epoch(epoch)
            train_loss = train_metrics["Loss"]["avg"]

            # 2. Validate
            val_metrics = self.validate()
            val_loss = val_metrics["Loss"]["avg"]
            val_dice = val_metrics["Dice"]["avg"]

            # 3. Step Scheduler
            # CosineAnnealingLR usually steps once per epoch
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]

            # 4. Log Metrics (Full Precision)
            print(
                f"Epoch {epoch} | LR: {current_lr:.8f} | "
                f"Train Loss: {train_loss:.8f} | "
                f"Val Loss: {val_loss:.8f} | "
                f"Val Dice: {val_dice:.8f}"
            )

            # 5. Checkpointing (Top-K Strategy)
            # Only start saving after a certain epoch to avoid saving unstable early models
            if epoch >= Config.START_SAVING_EPOCH:
                self.save_checkpoint(epoch, val_dice)

            # 6. Early Stopping
            if val_dice > self.best_val_dice:
                self.best_val_dice = val_dice
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping triggered at epoch {epoch}. Best Dice: {self.best_val_dice:.8f}"
                )
                break
