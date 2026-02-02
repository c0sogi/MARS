import time
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import mixup_data, mixup_criterion, save_checkpoint


class Trainer:
    """
    Trainer class to handle the training loop, validation, and checkpointing
    for the Right Whale Detection model.
    """

    def __init__(self, model, optimizer, scheduler, device):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            optimizer (torch.optim.Optimizer): The optimizer.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            device (str or torch.device): Device to run training on.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # Initialize Loss Function
        # We use BCEWithLogitsLoss which combines Sigmoid + BCE.
        # We apply the positive class weight to handle imbalance.
        pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def train_one_epoch(self, train_loader):
        """
        Runs one epoch of training with Mixup augmentation.

        Args:
            train_loader (DataLoader): DataLoader for training data.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.device)
            target = target.to(self.device)

            # Apply Mixup Augmentation
            data, y_a, y_b, lam = mixup_data(
                data, target, alpha=Config.MIXUP_ALPHA, device=self.device
            )

            self.optimizer.zero_grad()

            # Forward Pass
            output = self.model(data)

            # Squeeze output to match target shape (Batch,)
            # Model output is (B, 1), Target is (B,)
            output = output.squeeze(1)

            # Compute Mixup Loss
            loss = mixup_criterion(self.criterion, output, y_a, y_b, lam)

            # Backward Pass & Optimize
            loss.backward()
            self.optimizer.step()

            # Accumulate Loss
            running_loss += loss.item() * data.size(0)
            count += data.size(0)

        return running_loss / count

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.

        Args:
            val_loader (DataLoader): DataLoader for validation data.

        Returns:
            tuple: (average_loss, auc_score)
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        all_targets = []
        all_preds = []

        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                # Forward Pass
                output = self.model(data)
                output = output.squeeze(1)

                # Compute Loss (Standard BCE)
                loss = self.criterion(output, target)

                running_loss += loss.item() * data.size(0)
                count += data.size(0)

                # Collect predictions for AUC calculation
                # Apply Sigmoid to convert logits to probabilities
                preds = torch.sigmoid(output).cpu().numpy()
                targets = target.cpu().numpy()

                all_preds.extend(preds)
                all_targets.extend(targets)

        avg_loss = running_loss / count

        # Compute ROC-AUC
        # Handle edge case where a batch/set might only have one class
        try:
            auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            auc = 0.5

        return avg_loss, auc

    def train_model(
        self, train_loader, val_loader, num_epochs, save_name="best_model.pth"
    ):
        """
        Runs the full training pipeline with Early Stopping and LR Scheduling.

        Args:
            train_loader (DataLoader): Training data.
            val_loader (DataLoader): Validation data.
            num_epochs (int): Maximum number of epochs.
            save_name (str): Filename to save the best model checkpoint.

        Returns:
            float: The best validation AUC achieved.
        """
        best_auc = 0.0
        patience_counter = 0

        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(train_loader)

            # Validate
            val_loss, val_auc = self.validate(val_loader)

            # Step Scheduler
            # ReduceLROnPlateau usually monitors validation loss
            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            duration = time.time() - start_time

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Time: {duration:.2f}s | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val AUC: {val_auc}"
            )

            # Early Stopping & Checkpointing
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0

                # Save Checkpoint
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_score": best_auc,
                    },
                    is_best=True,
                    filename=save_name,
                )
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(
                        f"Early stopping triggered at epoch {epoch + 1}. Best AUC: {best_auc}"
                    )
                    break

        return best_auc
