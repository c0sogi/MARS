import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import (
    seed_everything,
    LaplaceLogLikelihood,
    calculate_competition_metric,
)
from library.data import get_dataloaders
from library.model import GMARNet


class Trainer:
    """
    Trainer class for the GMAR-Net model.
    Handles training, validation, and model checkpointing.
    """

    def __init__(self, debug=False):
        self.debug = debug
        self.device = torch.device(Config.DEVICE)

        # Ensure reproducibility
        seed_everything(Config.SEED)

        # Create directories
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

        # Load Data
        self.train_loader, self.val_loader, self.scalers = get_dataloaders(
            debug=self.debug
        )
        self.fvc_scaler = self.scalers["fvc_scaler"]

        # Initialize Model
        self.model = GMARNet().to(self.device)

        # Initialize Loss
        self.criterion = LaplaceLogLikelihood().to(self.device)

        # Optimizer with Differential Learning Rates
        # Group 1: Backbone parameters (only those with requires_grad=True)
        backbone_params = [
            p for n, p in self.model.backbone.named_parameters() if p.requires_grad
        ]

        # Group 2: The rest of the network (Clinical Net, Visual Net, Gate, Head)
        # We can filter by checking if the parameter is NOT in the backbone
        backbone_ids = list(map(id, backbone_params))
        head_params = [
            p
            for p in self.model.parameters()
            if id(p) not in backbone_ids and p.requires_grad
        ]

        self.optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": Config.LR_BACKBONE},
                {"params": head_params, "lr": Config.LR_HEAD},
            ],
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # Training State
        self.best_metric = -float("inf")
        self.patience_counter = 0
        self.early_stopping_patience = 10  # Stop if no improvement for 10 epochs

    def train_one_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            images = batch["image"].to(self.device)
            clinical = batch["clinical"].to(self.device)
            targets = batch["target"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            preds = self.model(images, clinical)

            # Compute Loss
            loss = self.criterion(preds, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """
        Runs validation loop and calculates competition metric.
        """
        self.model.eval()
        running_loss = 0.0

        all_true_fvc = []
        all_pred_fvc = []
        all_pred_sigma = []

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["image"].to(self.device)
                clinical = batch["clinical"].to(self.device)
                targets = batch["target"].to(self.device)

                # Forward pass
                preds = self.model(images, clinical)

                # Compute Loss (on scaled data)
                loss = self.criterion(preds, targets)
                running_loss += loss.item()

                # Inverse Transform for Metric Calculation
                # Preds: [mu_scaled, sigma_scaled]
                mu_scaled = preds[:, 0].cpu().numpy()
                sigma_scaled = preds[:, 1].cpu().numpy()
                target_scaled = targets.cpu().numpy()

                # Inverse transform mu: mu * std + mean
                mu_original = self.fvc_scaler.inverse_transform(mu_scaled)

                # Inverse transform sigma: sigma * std (scale only)
                sigma_original = self.fvc_scaler.inverse_transform_sigma(sigma_scaled)

                # Inverse transform target: target * std + mean
                target_original = self.fvc_scaler.inverse_transform(target_scaled)

                all_true_fvc.extend(target_original)
                all_pred_fvc.extend(mu_original)
                all_pred_sigma.extend(sigma_original)

        avg_loss = running_loss / len(self.val_loader)

        # Calculate Competition Metric
        metric_score = calculate_competition_metric(
            np.array(all_true_fvc), np.array(all_pred_fvc), np.array(all_pred_sigma)
        )

        return avg_loss, metric_score

    def train(self):
        """
        Main training loop.
        """
        print(f"Starting training on device: {self.device}")
        print(
            f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}"
        )

        start_time = time.time()

        for epoch in range(1, Config.EPOCHS + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_metric = self.validate()

            # Step Scheduler
            self.scheduler.step()

            # Logging
            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"Time: {epoch_time:.1f}s | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Metric: {val_metric}"
            )

            # Checkpointing & Early Stopping
            if val_metric > self.best_metric:
                self.best_metric = val_metric
                self.patience_counter = 0
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"  >>> New Best Model Saved! Metric: {self.best_metric}")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        total_time = time.time() - start_time
        print(
            f"Training complete in {total_time:.1f}s. Best Metric: {self.best_metric}"
        )


def run_training(debug=False):
    """
    Helper function to instantiate trainer and run training.
    """
    trainer = Trainer(debug=debug)
    trainer.train()
