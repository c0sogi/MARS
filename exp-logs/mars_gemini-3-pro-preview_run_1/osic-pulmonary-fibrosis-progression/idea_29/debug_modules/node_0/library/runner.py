import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import get_logger, AverageMeter, calculate_metric, seed_everything
from library.model import CVRNet
from library.loss import RobustLaplaceLoss
from library.dataset import get_dataloaders


class Trainer:
    """
    Manages the training, validation, and inference lifecycle of the CVR-Net model.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self.logger = get_logger("Trainer")

        # Initialize Model
        self.model = CVRNet()
        self.model.to(self.device)

        # Initialize Loss
        self.criterion = RobustLaplaceLoss()
        self.criterion.to(self.device)

        # Optimizer and Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # State tracking
        self.best_score = -float("inf")
        self.best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    def train_epoch(self, train_loader, epoch):
        """Runs one epoch of training."""
        self.model.train()
        losses = AverageMeter()

        for batch in train_loader:
            # Move data to device
            axial = batch["axial"].to(self.device).float()
            coronal = batch["coronal"].to(self.device).float()
            fusion = batch["fusion"].to(self.device)
            anchor = batch["anchor"].to(self.device)
            meta = batch["meta"].to(self.device)
            targets = batch["target"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            # Returns: alpha, sigma_base, sigma_growth
            preds = self.model(axial, coronal, fusion, anchor)

            # Compute Loss
            loss = self.criterion(preds, targets, meta)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.MAX_GRAD_NORM
            )

            self.optimizer.step()

            losses.update(loss.item(), axial.size(0))

        return losses.avg

    def validate(self, val_loader):
        """Runs validation and calculates the competition metric."""
        self.model.eval()
        losses = AverageMeter()
        metrics = AverageMeter()

        with torch.no_grad():
            for batch in val_loader:
                axial = batch["axial"].to(self.device).float()
                coronal = batch["coronal"].to(self.device).float()
                fusion = batch["fusion"].to(self.device)
                anchor = batch["anchor"].to(self.device)
                meta = batch["meta"].to(self.device)
                targets = batch["target"].to(self.device)

                # Forward pass
                preds = self.model(axial, coronal, fusion, anchor)

                # Compute Loss
                loss = self.criterion(preds, targets, meta)
                losses.update(loss.item(), axial.size(0))

                # --- Metric Calculation ---
                # Unpack predictions
                alpha, sigma_base, sigma_growth = preds

                # Unpack metadata
                base_fvc = meta[:, 0]
                week_diff = meta[:, 1]

                # Reconstruct FVC and Sigma
                # FVC = Base + Slope * Diff
                fvc_pred = base_fvc + alpha * week_diff

                # Sigma = Base + Growth * |Diff|
                sigma_pred = sigma_base + sigma_growth * torch.abs(week_diff)

                # Calculate metric using utility
                score = calculate_metric(targets, fvc_pred, sigma_pred)
                metrics.update(score, axial.size(0))

        return losses.avg, metrics.avg

    def train(self):
        """Main training loop with Early Stopping."""
        self.logger.info("Starting Training...")

        # Get DataLoaders
        train_loader, val_loader, _ = get_dataloaders()

        patience_counter = 0

        for epoch in range(Config.N_EPOCHS):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)

            # Validate
            val_loss, val_score = self.validate(val_loader)

            # Step Scheduler
            self.scheduler.step()

            # Logging
            self.logger.info(
                f"Epoch {epoch+1}/{Config.N_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Metric: {val_score:.10f}"
            )

            # Early Stopping Check
            if val_score > self.best_score:
                self.best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                self.logger.info(
                    f"New Best Score! Model saved to {self.best_model_path}"
                )
            else:
                patience_counter += 1
                self.logger.info(
                    f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                self.logger.info("Early stopping triggered.")
                break

        self.logger.info(
            f"Training Complete. Best Validation Score: {self.best_score:.10f}"
        )

    def predict(self):
        """Generates predictions for the test set and saves the submission file."""
        self.logger.info("Starting Inference...")

        # Load Best Model
        if not os.path.exists(self.best_model_path):
            self.logger.warning(
                "Best model not found. Using current model weights (this may be suboptimal)."
            )
        else:
            self.logger.info(f"Loading model from {self.best_model_path}")
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )

        self.model.eval()

        # Get Test Loader
        _, _, test_loader = get_dataloaders()

        # To store results
        all_fvc = []
        all_conf = []

        with torch.no_grad():
            for batch in test_loader:
                axial = batch["axial"].to(self.device).float()
                coronal = batch["coronal"].to(self.device).float()
                fusion = batch["fusion"].to(self.device)
                anchor = batch["anchor"].to(self.device)
                meta = batch["meta"].to(self.device)

                # Forward pass
                alpha, sigma_base, sigma_growth = self.model(
                    axial, coronal, fusion, anchor
                )

                # Unpack metadata
                base_fvc = meta[:, 0]
                week_diff = meta[:, 1]

                # Reconstruct Predictions
                fvc_pred = base_fvc + alpha * week_diff
                sigma_pred = sigma_base + sigma_growth * torch.abs(week_diff)

                # Clip confidence (min 70) as per submission requirement logic
                # Note: The metric clips at 70, so we should output at least meaningful values.
                # The prompt says "confidence values are clipped at 70 ml to reflect approximate measurement uncertainty".
                # We apply this clipping to the output to be safe.
                sigma_pred = torch.clamp(sigma_pred, min=70)

                all_fvc.extend(fvc_pred.cpu().numpy())
                all_conf.extend(sigma_pred.cpu().numpy())

        # Align with Patient_Week IDs
        # The test_loader dataset dataframe corresponds 1-to-1 with the loader iteration order (shuffle=False)
        test_df = test_loader.dataset.df

        submission = pd.DataFrame(
            {
                "Patient_Week": test_df["Patient_Week"],
                "FVC": all_fvc,
                "Confidence": all_conf,
            }
        )

        # Save Submission
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
        self.logger.info(f"Submission shape: {submission.shape}")
        self.logger.info(f"Sample:\n{submission.head()}")
