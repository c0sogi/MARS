import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config, seed_everything
from library.utils import AverageMeter, save_checkpoint, kl_divergence
from library.data import get_dataloaders
from library.model import MultiResNetwork


class Trainer:
    def __init__(self, debug=False):
        self.debug = debug
        self.device = torch.device(Config.DEVICE)

        # Initialize Data
        print("Initializing DataLoaders...")
        self.train_loader, self.val_loader, _ = get_dataloaders(debug=self.debug)

        # Initialize Model
        print("Initializing Model...")
        self.model = MultiResNetwork()
        self.model.to(self.device)

        # Optimization
        self.criterion = nn.KLDivLoss(reduction="batchmean")
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-2
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

        # Tracking
        self.best_score = float("inf")
        self.patience = 4
        self.counter = 0

    def mixup_data(self, x_a, x_b, y, alpha=1.0):
        """Returns mixed inputs, pairs of targets, and lambda"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x_a.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_x_a = lam * x_a + (1 - lam) * x_a[index, :]
        mixed_x_b = lam * x_b + (1 - lam) * x_b[index, :]

        y_a, y_b = y, y[index]
        return mixed_x_a, mixed_x_b, y_a, y_b, lam

    def train_one_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()

        for i, (inputs, targets) in enumerate(self.train_loader):
            # Unpack inputs
            x_a, x_b = inputs
            x_a = x_a.to(self.device)
            x_b = x_b.to(self.device)
            targets = targets.to(self.device)

            # Apply MixUp
            mixed_x_a, mixed_x_b, target_a, target_b, lam = self.mixup_data(
                x_a, x_b, targets, alpha=1.0
            )

            # Forward pass
            # Model outputs Softmax probabilities
            outputs = self.model((mixed_x_a, mixed_x_b))

            # KLDivLoss expects Log-Probabilities as input
            # Add epsilon for numerical stability
            log_outputs = torch.log(outputs + 1e-15)

            # MixUp Loss
            loss = lam * self.criterion(log_outputs, target_a) + (
                1 - lam
            ) * self.criterion(log_outputs, target_b)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), x_a.size(0))

        return losses.avg

    def validate(self):
        self.model.eval()
        losses = AverageMeter()

        # Store predictions and targets for metric calculation
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                x_a, x_b = inputs
                x_a = x_a.to(self.device)
                x_b = x_b.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model((x_a, x_b))

                # Loss calculation (for monitoring)
                log_outputs = torch.log(outputs + 1e-15)
                loss = self.criterion(log_outputs, targets)
                losses.update(loss.item(), x_a.size(0))

                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Calculate competition metric
        kl_score = kl_divergence(all_targets, all_preds)

        return losses.avg, kl_score

    def fit(self):
        print(f"Starting training for {Config.EPOCHS} epochs on {self.device}...")

        for epoch in range(1, Config.EPOCHS + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_score = self.validate()

            # Scheduler Step
            self.scheduler.step()

            duration = time.time() - start_time

            print(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"Time: {duration:.1f}s | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val KL Score: {val_score}"
            )

            # Checkpointing
            is_best = val_score < self.best_score
            if is_best:
                self.best_score = val_score
                self.counter = 0
                print(f"New best model found! Saving...")
            else:
                self.counter += 1

            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "best_score": self.best_score,
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                },
                is_best,
                filename="checkpoint.pth",
                best_filename="best_model.pth",
            )

            # Early Stopping
            if self.counter >= self.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break


def main():
    seed_everything(Config.SEED)

    # Set debug=True to run a quick test with a subset of data
    # Set debug=False for full training
    trainer = Trainer(debug=False)
    trainer.fit()


if __name__ == "__main__":
    main()
