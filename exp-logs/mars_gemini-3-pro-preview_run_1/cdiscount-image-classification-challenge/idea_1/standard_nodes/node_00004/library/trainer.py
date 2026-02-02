import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import os

from library.config import Config, seed_everything
from library.model import MultiViewResNet
from library.dataset import CdiscountDataset, collate_fn
from library.utils import (
    AverageMeter,
    calculate_accuracy,
    save_checkpoint,
    get_transforms,
)


class Trainer:
    def __init__(self, config=Config):
        """
        Initializes the Trainer with configuration, model, optimizer, and scaler.
        """
        self.config = config
        self.device = torch.device(self.config.DEVICE)

        # Set random seeds
        seed_everything(self.config.SEED)

        # Initialize Model
        self.model = MultiViewResNet(
            num_classes=self.config.NUM_CLASSES, pretrained=self.config.PRETRAINED
        )
        self.model = self.model.to(self.device)

        # Loss Function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        # Mixed Precision Scaler
        self.scaler = torch.amp.GradScaler("cuda")

        # Scheduler (initialized in fit method once loader is ready)
        self.scheduler = None

    def train_epoch(self, loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        losses = AverageMeter()
        accuracies = AverageMeter()

        for images, indices, targets, _ in loader:
            images = images.to(self.device, non_blocking=True)
            indices = indices.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with torch.amp.autocast("cuda"):
                outputs = self.model(images, indices)
                loss = self.criterion(outputs, targets)

            # Backward Pass with Scaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Step Scheduler
            if self.scheduler:
                self.scheduler.step()

            # Metrics
            acc = calculate_accuracy(outputs, targets)
            losses.update(loss.item(), targets.size(0))
            accuracies.update(acc, targets.size(0))

        return losses.avg, accuracies.avg

    def validate(self, loader):
        """
        Runs validation on the provided loader.
        """
        self.model.eval()
        losses = AverageMeter()
        accuracies = AverageMeter()

        with torch.no_grad():
            for images, indices, targets, _ in loader:
                images = images.to(self.device, non_blocking=True)
                indices = indices.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    outputs = self.model(images, indices)
                    loss = self.criterion(outputs, targets)

                acc = calculate_accuracy(outputs, targets)
                losses.update(loss.item(), targets.size(0))
                accuracies.update(acc, targets.size(0))

        return losses.avg, accuracies.avg

    def fit(self, num_epochs=None, batch_size=None, debug_limit=None):
        """
        Main training loop with Early Stopping.

        Args:
            num_epochs (int, optional): Override default number of epochs.
            batch_size (int, optional): Override default batch size.
            debug_limit (int, optional): Limit dataset size for debugging.
        """
        epochs = num_epochs if num_epochs is not None else self.config.NUM_EPOCHS
        bs = batch_size if batch_size is not None else self.config.BATCH_SIZE

        # Initialize Datasets
        train_dataset = CdiscountDataset(
            metadata_path=self.config.TRAIN_METADATA,
            bson_path=self.config.TRAIN_BSON,
            transform=get_transforms("train"),
            mode="train",
        )

        val_dataset = CdiscountDataset(
            metadata_path=self.config.VAL_METADATA,
            bson_path=self.config.TRAIN_BSON,
            transform=get_transforms("val"),
            mode="val",
        )

        # Apply debug limit if specified
        if debug_limit:
            train_dataset = Subset(
                train_dataset, range(min(len(train_dataset), debug_limit))
            )
            val_dataset = Subset(val_dataset, range(min(len(val_dataset), debug_limit)))

        # Initialize DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=self.config.PIN_MEMORY,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=self.config.PIN_MEMORY,
        )

        # Initialize Scheduler (OneCycleLR requires steps_per_epoch)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
        )

        best_acc = 0.0
        patience_counter = 0

        print(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}: Train Loss {train_loss}, Train Acc {train_acc}, Val Loss {val_loss}, Val Acc {val_acc}"
            )

            # Checkpointing and Early Stopping
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "best_acc": best_acc,
                },
                is_best,
            )

            if patience_counter >= self.config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
