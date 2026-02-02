import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import AverageMeter, metric_score, InverseScaler, seed_everything
from library.model import TCDSNet
from library.loss import LaplaceNLLLoss
from library.data import get_dataloaders


class Trainer:
    """
    Manages the training lifecycle of the TCDS-Net model.
    """

    def __init__(self, debug=False):
        self.debug = debug
        self.device = torch.device(Config.DEVICE)

        # 1. Data Loading
        # get_dataloaders handles caching and preprocessing internally
        self.train_loader, self.val_loader, _, self.scalers = get_dataloaders(
            debug=self.debug
        )

        # 2. Inverse Scaler
        # Required to convert model predictions (Z-scored) back to ml for metric calculation
        self.inverse_scaler = InverseScaler(
            mean=self.scalers["fvc_mean"], std=self.scalers["fvc_std"]
        )

        # 3. Model Setup
        self.model = TCDSNet().to(self.device)

        # 4. Optimizer with Differential Learning Rates
        # Group 1: Backbone (Pre-trained layers) -> Lower LR
        # Group 2: Head (Projector + MLP) -> Higher LR
        backbone_params = list(self.model.image_encoder.parameters())
        head_params = list(self.model.img_projector.parameters()) + list(
            self.model.mlp.parameters()
        )

        self.optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": Config.LR_BACKBONE},
                {"params": head_params, "lr": Config.LR_HEAD},
            ],
            weight_decay=Config.WEIGHT_DECAY,
        )

        # 5. Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # 6. Loss Function
        self.criterion = LaplaceNLLLoss()

        # 7. Tracking
        self.best_score = -float("inf")

    def train_one_epoch(self):
        """
        Runs one epoch of training.
        """
        self.model.train()
        loss_meter = AverageMeter()
        metric_meter = AverageMeter()

        for batch in self.train_loader:
            # Move data to device
            images = batch["image"].to(self.device)
            tabular = batch["tabular"].to(self.device)
            target = batch["target"].to(self.device)  # Scaled target for optimization
            raw_target = batch["raw_target"].to(
                self.device
            )  # Raw target for metric calc

            self.optimizer.zero_grad()

            # Forward pass
            mu, sigma = self.model(images, tabular)

            # Loss calculation (on scaled values)
            loss = self.criterion(mu, sigma, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Metric Calculation (on original scale)
            # 1. Inverse transform predictions to ml
            mu_orig, sigma_orig = self.inverse_scaler(mu, sigma)
            # 2. Calculate competition metric
            score = metric_score(raw_target, mu_orig, sigma_orig)

            # Update tracking
            loss_meter.update(loss.item(), images.size(0))
            metric_meter.update(score, images.size(0))

        return loss_meter.avg, metric_meter.avg

    def validate(self):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        loss_meter = AverageMeter()
        metric_meter = AverageMeter()

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                tabular = batch["tabular"].to(self.device)
                target = batch["target"].to(self.device)
                raw_target = batch["raw_target"].to(self.device)

                mu, sigma = self.model(images, tabular)

                loss = self.criterion(mu, sigma, target)

                mu_orig, sigma_orig = self.inverse_scaler(mu, sigma)
                score = metric_score(raw_target, mu_orig, sigma_orig)

                loss_meter.update(loss.item(), images.size(0))
                metric_meter.update(score, images.size(0))

        return loss_meter.avg, metric_meter.avg

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        seed_everything(Config.SEED)
        print("Starting training...")

        patience_counter = 0

        for epoch in range(Config.EPOCHS):
            current_epoch = epoch + 1

            # Train
            train_loss, train_score = self.train_one_epoch()

            # Validate
            val_loss, val_score = self.validate()

            # Step Scheduler
            self.scheduler.step()

            # Logging
            print(f"Epoch {current_epoch}/{Config.EPOCHS}")
            print(f"Train Loss: {train_loss} | Train Score: {train_score}")
            print(f"Val Loss: {val_loss} | Val Score: {val_score}")

            # Early Stopping & Checkpointing
            if val_score > self.best_score:
                self.best_score = val_score
                patience_counter = 0
                self.save_checkpoint("best_model.pth")
                print(f"New best model saved with score: {self.best_score}")
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {current_epoch}")
                break

        print(f"Training complete. Best Validation Score: {self.best_score}")

    def save_checkpoint(self, filename):
        """
        Saves the model state dict.
        """
        save_path = os.path.join(Config.CHECKPOINT_DIR, filename)
        torch.save(self.model.state_dict(), save_path)
