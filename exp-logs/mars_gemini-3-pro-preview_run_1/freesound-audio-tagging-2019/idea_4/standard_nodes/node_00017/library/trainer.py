import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, calculate_lwlrap, save_checkpoint
from library.dataset import AudioDataset
from library.model import AudioClassifier


class Trainer:
    """
    Trainer class for the Audio Tagging task.
    Handles model training, validation, mixup augmentation, and checkpointing.
    """

    def __init__(self):
        """
        Initialize the Trainer with model, optimizer, scheduler, and criterion.
        """
        # Ensure reproducibility
        set_seed(Config.SEED)

        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        self.model = AudioClassifier()
        self.model.to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )

        # Initialize Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
        )

        # Loss Function (Multi-label classification)
        self.criterion = nn.BCEWithLogitsLoss()

        # Best score tracking
        self.best_score = -np.inf

    def get_dataloader(self, split, debug=False):
        """
        Creates a DataLoader for the specified split.

        Args:
            split (str): 'train' or 'val'.
            debug (bool): Whether to use a subset of data.

        Returns:
            DataLoader
        """
        dataset = AudioDataset(split=split, debug=debug)

        if split == "train":
            return DataLoader(
                dataset,
                batch_size=Config.BATCH_SIZE,
                shuffle=True,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
                drop_last=True,
            )
        else:
            # Validation uses full length clips which vary in size.
            # Batch size 1 avoids collation issues with variable tensor sizes.
            return DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                pin_memory=True,
            )

    def mixup_data(self, x, y, alpha=0.4):
        """
        Applies Mixup augmentation to the batch.

        Args:
            x (Tensor): Input batch.
            y (Tensor): Target labels.
            alpha (float): Mixup beta distribution parameter.

        Returns:
            mixed_x, mixed_y
        """
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        mixed_y = lam * y + (1 - lam) * y[index, :]

        return mixed_x, mixed_y

    def train_one_epoch(self, train_loader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Apply Mixup (Config says Prob=1.0)
            images, labels = self.mixup_data(images, labels, Config.MIXUP_ALPHA)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self, val_loader):
        """
        Runs validation on the validation set.

        Returns:
            float: LWLRAP score.
        """
        self.model.eval()
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                # Apply sigmoid to convert logits to probabilities
                preds = torch.sigmoid(outputs)

                all_preds.append(preds.cpu())
                all_targets.append(labels.cpu())

        # Concatenate all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Calculate metric
        score = calculate_lwlrap(all_targets, all_preds)
        return score

    def train(self, debug=False):
        """
        Main training loop with Early Stopping.
        """
        train_loader = self.get_dataloader("train", debug=debug)
        val_loader = self.get_dataloader("val", debug=debug)

        print(f"Starting training on device: {self.device}")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

        patience_counter = 0

        for epoch in range(1, Config.EPOCHS + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(train_loader, epoch)

            # Validate
            val_score = self.validate(val_loader)

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            elapsed = time.time() - start_time

            # Print metrics with full precision
            print(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"Time: {elapsed:.2f}s | "
                f"LR: {current_lr:.8f} | "
                f"Train Loss: {train_loss:.16f} | "
                f"Val LWLRAP: {val_score:.16f}"
            )

            # Save Best Model
            if val_score > self.best_score:
                print(
                    f"Validation score improved ({self.best_score:.16f} --> {val_score:.16f}). Saving model..."
                )
                self.best_score = val_score
                save_checkpoint(
                    self.model, self.optimizer, self.scheduler, epoch, val_score
                )
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break

        print(f"Training complete. Best Val LWLRAP: {self.best_score:.16f}")
