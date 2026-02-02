import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from library.config import Config
from library.utils import MetricMonitor, save_checkpoint, seed_everything
from library.model import DFL_GI_BiLSTM
from library.data import get_data_loaders


class WeightedL1Loss(nn.Module):
    """
    Weighted L1 Loss for Ventilator Pressure Prediction.
    Assigns different weights to inspiratory and expiratory phases.
    """

    def __init__(self, inspiratory_weight=1.0, expiratory_weight=0.1):
        super().__init__()
        self.inspiratory_weight = inspiratory_weight
        self.expiratory_weight = expiratory_weight
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, preds, targets, u_out):
        """
        Args:
            preds: (Batch, Seq_Len)
            targets: (Batch, Seq_Len)
            u_out: (Batch, Seq_Len) - 0 for inspiratory, 1 for expiratory
        """
        # Calculate raw L1 loss per element
        loss = self.l1(preds, targets)

        # Create weight mask
        # u_out is 0 for inspiratory (weight 1.0) and 1 for expiratory (weight 0.1)
        weights = (1 - u_out) * self.inspiratory_weight + u_out * self.expiratory_weight

        # Apply weights
        weighted_loss = loss * weights

        # Return mean loss
        return weighted_loss.mean()


class Trainer:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        self.model = DFL_GI_BiLSTM().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler (Stretched Horizon)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # Loss Function
        self.criterion = WeightedL1Loss(
            inspiratory_weight=Config.INSPIRATORY_WEIGHT,
            expiratory_weight=Config.EXPIRATORY_WEIGHT,
        )

        self.best_score = float("inf")

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        metric_monitor = MetricMonitor()

        for batch_idx, (inputs, targets, u_out) in enumerate(train_loader):
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            u_out = u_out.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            preds = self.model(inputs)

            loss = self.criterion(preds, targets, u_out)

            loss.backward()
            self.optimizer.step()

            metric_monitor.update("Loss", loss.item())

        return metric_monitor.avg["Loss"]

    def evaluate(self, val_loader):
        self.model.eval()
        metric_monitor = MetricMonitor()

        with torch.no_grad():
            for inputs, targets, u_out in val_loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                u_out = u_out.to(self.device, non_blocking=True)

                preds = self.model(inputs)

                # Metric: MAE only on inspiratory phase (u_out == 0)
                # Create boolean mask for inspiratory phase
                mask = u_out == 0

                # Select elements
                preds_insp = torch.masked_select(preds, mask)
                targets_insp = torch.masked_select(targets, mask)

                if len(targets_insp) > 0:
                    mae = torch.abs(preds_insp - targets_insp).mean()
                    metric_monitor.update("MAE", mae.item(), n=len(targets_insp))

        return metric_monitor.avg["MAE"]

    def fit(self, train_loader, val_loader):
        print(f"Starting training for {Config.EPOCHS} epochs on {self.device}...")

        for epoch in range(1, Config.EPOCHS + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(train_loader, epoch)

            # Validate
            val_mae = self.evaluate(val_loader)

            # Step Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            duration = time.time() - start_time

            # Checkpoint
            is_best = val_mae < self.best_score
            if is_best:
                self.best_score = val_mae

            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "best_score": self.best_score,
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                },
                is_best,
                Config.WORKING_DIR,
            )

            # Print Metrics (Full precision as requested)
            print(
                f"Epoch {epoch} | "
                f"Time: {duration:.2f}s | "
                f"LR: {current_lr} | "
                f"Train Loss: {train_loss} | "
                f"Val MAE: {val_mae} | "
                f"Best MAE: {self.best_score}"
            )


def run_training():
    """
    Main execution function to prepare data and run the training pipeline.
    """
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Load Data (Handles caching internally)
    train_loader, val_loader, _ = get_data_loaders(load_cached_data=True)

    # Initialize Trainer
    trainer = Trainer()

    # Run Training
    trainer.fit(train_loader, val_loader)
