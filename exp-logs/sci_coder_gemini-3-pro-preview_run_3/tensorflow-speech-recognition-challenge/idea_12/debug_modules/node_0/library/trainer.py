import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import os

from library.config import Config
from library.utils import set_seed, MetricMonitor, save_checkpoint
from library.dataset import CachedSpeechDataset, get_class_weights
from library.model import FrequencyPreservingSKResNetCRNN


class Trainer:
    """
    Manages the training lifecycle of the Frequency-Preserving SK-ResNet-CRNN.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)

        # Initialize Model
        self.model = FrequencyPreservingSKResNetCRNN()
        self.model = self.model.to(self.device)

        # Loss Function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

    def get_dataloaders(self):
        """
        Prepares DataLoaders for training and validation.
        Uses WeightedRandomSampler for the training set to handle class imbalance.
        """
        # --- Training Set ---
        train_dataset = CachedSpeechDataset(Config.TRAIN_META, mode="train")

        # Calculate weights for balancing
        sample_weights = get_class_weights(train_dataset)
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            sampler=sampler,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        # --- Validation Set ---
        val_dataset = CachedSpeechDataset(Config.VAL_META, mode="val")

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=False,
        )

        return train_loader, val_loader

    def calculate_accuracy(self, output, target):
        """
        Computes the accuracy for a batch.
        """
        with torch.no_grad():
            batch_size = target.size(0)
            _, pred = torch.max(output, dim=1)
            correct = (pred == target).sum().item()
            return correct / batch_size

    def train_one_epoch(self, train_loader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        loss_monitor = MetricMonitor()
        acc_monitor = MetricMonitor()

        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(self.device)
            targets = targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            acc = self.calculate_accuracy(outputs, targets)
            loss_monitor.update(loss.item(), features.size(0))
            acc_monitor.update(acc, features.size(0))

        return loss_monitor.avg, acc_monitor.avg

    def validate(self, val_loader):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        loss_monitor = MetricMonitor()
        acc_monitor = MetricMonitor()

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, targets)
                acc = self.calculate_accuracy(outputs, targets)

                loss_monitor.update(loss.item(), features.size(0))
                acc_monitor.update(acc, features.size(0))

        return loss_monitor.avg, acc_monitor.avg

    def fit(self):
        """
        Main training loop with Early Stopping and Checkpointing.
        """
        print(f"Starting training on device: {self.device}")

        train_loader, val_loader = self.get_dataloaders()

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(1, Config.EPOCHS + 1):
            # Train
            train_loss, train_acc = self.train_one_epoch(train_loader, epoch)

            # Validate
            val_loss, val_acc = self.validate(val_loader)

            # Scheduler Step
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Logging
            print(f"Epoch: {epoch}")
            print(f"Train Loss: {train_loss}")
            print(f"Train Acc:  {train_acc}")
            print(f"Val Loss:   {val_loss}")
            print(f"Val Acc:    {val_acc}")
            print(f"LR:         {current_lr}")
            print("-" * 30)

            # Checkpointing & Early Stopping
            if val_acc > best_val_acc:
                print(
                    f"Validation Accuracy improved from {best_val_acc} to {val_acc}. Saving model..."
                )
                best_val_acc = val_acc
                patience_counter = 0
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch,
                    val_acc,
                    Config.MODEL_SAVE_PATH,
                )
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Accuracy: {best_val_acc}")
