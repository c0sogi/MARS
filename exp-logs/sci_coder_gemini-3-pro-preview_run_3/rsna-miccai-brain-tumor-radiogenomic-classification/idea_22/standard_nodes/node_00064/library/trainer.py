import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import LEARNING_RATE, NUM_EPOCHS, CACHE_DIR, DEVICE, seed_everything
from library.utils import AverageMeter, compute_roc_auc, print_metrics, get_device
from library.model import MSSHDNetwork


class Trainer:
    """
    Trainer class to manage the training and validation loops for the MS-SHD Network.
    """

    def __init__(self, train_loader, val_loader):
        """
        Initialize the Trainer.

        Args:
            train_loader: DataLoader for the training set.
            val_loader: DataLoader for the validation set.
        """
        self.device = get_device()
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Initialize Model
        self.model = MSSHDNetwork()
        self.model.to(self.device)

        # Initialize Optimizer
        # Using Adam with learning rate 1e-4 and mild weight decay for regularization
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2
        )

        # Initialize Loss Function
        # BCEWithLogitsLoss is numerically stable for binary classification
        self.criterion = nn.BCEWithLogitsLoss()

        # Checkpointing setup
        self.best_auc = 0.0
        # Ensure cache directory exists
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.save_path = os.path.join(CACHE_DIR, "best_model.pth")

        # Set seeds for reproducibility
        seed_everything()

    def train_one_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        losses = AverageMeter()

        all_targets = []
        all_preds = []

        for batch_idx, batch_data in enumerate(self.train_loader):
            images = batch_data["image"].to(self.device)
            targets = batch_data["target"].to(self.device)

            # Forward pass
            logits = self.model(images)

            # Reshape targets to match logits (B, 1)
            loss = self.criterion(logits, targets.unsqueeze(1))

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)

            # Store predictions for monitoring
            preds = torch.sigmoid(logits).detach().cpu().numpy()
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds)

        # Calculate training AUC for monitoring
        train_auc = compute_roc_auc(np.array(all_targets), np.array(all_preds))

        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] Train Loss: {losses.avg:.6f} | Train AUC: {train_auc}"
        )
        return losses.avg

    def validate(self):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        losses = AverageMeter()

        all_targets = []
        all_preds = []

        with torch.no_grad():
            for batch_data in self.val_loader:
                images = batch_data["image"].to(self.device)
                targets = batch_data["target"].to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, targets.unsqueeze(1))

                batch_size = images.size(0)
                losses.update(loss.item(), batch_size)

                # Apply sigmoid to get probabilities
                preds = torch.sigmoid(logits).cpu().numpy()
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(preds)

        # Compute AUC
        val_auc = compute_roc_auc(np.array(all_targets), np.array(all_preds))

        return {"val_loss": losses.avg, "val_auc": val_auc}

    def run(self):
        """
        Main training loop.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(1, NUM_EPOCHS + 1):
            _ = self.train_one_epoch(epoch)
            metrics = self.validate()

            # Print full precision metrics
            print_metrics(metrics)

            # Checkpointing based on AUC
            current_auc = metrics["val_auc"]
            if current_auc > self.best_auc:
                print(
                    f"Validation AUC improved from {self.best_auc} to {current_auc}. Saving model..."
                )
                self.best_auc = current_auc
                torch.save(self.model.state_dict(), self.save_path)
            else:
                print(f"Validation AUC did not improve (Best: {self.best_auc}).")

        print(f"Training complete. Best AUC: {self.best_auc}")
        print(f"Best model saved to: {self.save_path}")
