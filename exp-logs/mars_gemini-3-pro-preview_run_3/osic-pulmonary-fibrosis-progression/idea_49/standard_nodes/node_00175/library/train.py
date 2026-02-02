import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from library.config import Config
from library.utils import seed_everything, AverageMeter, score_function
from library.loss import StandardizedLaplaceLoss
from library.data import get_dataloaders
from library.model import SCARNet


class Trainer:
    def __init__(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = Config.DEVICE

        # Initialize Model
        self.model = SCARNet()
        self.model.to(self.device)

        # Loss Function
        self.criterion = StandardizedLaplaceLoss()

        # Optimizer with Differential Learning Rates
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        self.optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": Config.BACKBONE_LR},
                {"params": head_params, "lr": Config.HEAD_LR},
            ],
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # Tracking
        self.best_score = -float("inf")
        self.best_loss = float("inf")

    def train_one_epoch(self, epoch):
        self.model.train()
        loss_meter = AverageMeter()

        for batch_idx, (images, tabular, targets) in enumerate(self.train_loader):
            images = images.to(self.device)
            tabular = tabular.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(images, tabular)

            # Calculate Loss (Standardized Space)
            loss = self.criterion(preds, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), images.size(0))

        return loss_meter.avg

    def validate(self):
        self.model.eval()
        loss_meter = AverageMeter()
        metric_meter = AverageMeter()

        with torch.no_grad():
            for images, tabular, targets in self.val_loader:
                images = images.to(self.device)
                tabular = tabular.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                preds = self.model(images, tabular)

                # 1. Calculate Loss in Standardized Space
                loss = self.criterion(preds, targets)
                loss_meter.update(loss.item(), images.size(0))

                # 2. Inverse Transform for Metric Calculation
                # Preds: [mu_scaled, sigma_scaled]
                mu_scaled = preds[:, 0].cpu().numpy()
                sigma_scaled = preds[:, 1].cpu().numpy()

                # Targets: [FVC_Scaled]
                targets_scaled = targets.cpu().numpy().flatten()

                # Inverse Transform
                # mu_final = mu_scaled * sigma_global + mu_global
                mu_final = mu_scaled * Config.TARGET_STD + Config.TARGET_MEAN

                # sigma_final = sigma_scaled * sigma_global
                sigma_final = sigma_scaled * Config.TARGET_STD

                # targets_final = targets_scaled * sigma_global + mu_global
                targets_final = targets_scaled * Config.TARGET_STD + Config.TARGET_MEAN

                # 3. Calculate Metric
                # Note: score_function handles the clipping (sigma > 70, delta < 1000)
                score = score_function(targets_final, mu_final, sigma_final)
                metric_meter.update(score, images.size(0))

        return loss_meter.avg, metric_meter.avg

    def fit(self):
        print(f"Starting training on device: {self.device}")
        Config.print_config()

        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_score = self.validate()

            # Step Scheduler
            self.scheduler.step()

            epoch_time = time.time() - start_time

            # Print Metrics
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Time: {epoch_time:.1f}s | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Metric: {val_score:.8f}"
            )

            # Checkpoint & Early Stopping
            # Metric is negative Laplace Log Likelihood, higher is better (e.g., -6.5 > -6.8)
            if val_score > self.best_score:
                print(
                    f"Validation Score improved ({self.best_score:.6f} --> {val_score:.6f}). Saving model..."
                )
                self.best_score = val_score
                self.best_loss = val_loss
                patience_counter = 0

                save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Metric: {self.best_score:.8f}")


def run_training():
    """
    Main entry point to run the training process.
    """
    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Load Data
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders()

    # Initialize Trainer
    trainer = Trainer(train_loader, val_loader)

    # Start Training
    trainer.fit()
