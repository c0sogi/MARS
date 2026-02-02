import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, SWALR
import logging
import pandas as pd
from library.utils import AverageMeter, save_checkpoint


class Trainer:
    """
    Trainer class for Iceberg Classification.
    Handles training, validation (with TTA), SWA, and prediction.
    """

    def __init__(self, model, device, logger=None, learning_rate=1e-3):
        self.model = model.to(device)
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

        # Optimization components
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=0.01
        )

        # Scheduler: Gentle Decay (0.5) with High Patience (10)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=10, verbose=False
        )

        self.label_smoothing = 0.05

    def _smooth_labels(self, targets):
        """
        Applies label smoothing to binary targets.
        y_smooth = y(1 - eps) + 0.5 * eps
        """
        return targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

    def train_epoch(self, loader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        losses = AverageMeter()

        for batch_idx, (images, angles, targets, _) in enumerate(loader):
            images = images.to(self.device)
            angles = angles.to(self.device)
            targets = targets.to(self.device).float().view(-1, 1)

            # Apply label smoothing
            smooth_targets = self._smooth_labels(targets)

            self.optimizer.zero_grad()
            outputs = self.model(images, angles)
            loss = self.criterion(outputs, smooth_targets)

            loss.backward()
            self.optimizer.step()

            losses.update(loss.item(), images.size(0))

        return losses.avg

    def validate(self, loader, tta=True):
        """
        Runs validation, optionally with TTA.
        Returns the average Log Loss.
        """
        self.model.eval()
        losses = AverageMeter()
        criterion_val = nn.BCELoss()  # Use BCELoss for probabilities

        with torch.no_grad():
            for i, data in enumerate(loader):
                # Unpack data (img, angle, label, id)
                if len(data) == 4:
                    images, angles, targets, ids = data
                    targets = targets.to(self.device).float().view(-1, 1)
                else:
                    raise ValueError("Validation loader must provide labels.")

                images = images.to(self.device)
                angles = angles.to(self.device)

                if tta:
                    # 1. Original
                    pred_1 = torch.sigmoid(self.model(images, angles))

                    # 2. Horizontal Flip
                    images_h = torch.flip(images, dims=[3])
                    pred_2 = torch.sigmoid(self.model(images_h, angles))

                    # 3. Vertical Flip
                    images_v = torch.flip(images, dims=[2])
                    pred_3 = torch.sigmoid(self.model(images_v, angles))

                    # Average
                    probs = (pred_1 + pred_2 + pred_3) / 3.0
                else:
                    probs = torch.sigmoid(self.model(images, angles))

                # Clamp probabilities to avoid log(0)
                probs = torch.clamp(probs, 1e-7, 1 - 1e-7)

                loss = criterion_val(probs, targets)
                losses.update(loss.item(), images.size(0))

        return losses.avg

    def fit(self, train_loader, val_loader, epochs, checkpoint_dir):
        """
        Standard training phase with Early Stopping.
        """
        best_loss = float("inf")
        patience_counter = 0
        early_stopping_patience = 20

        self.logger.info(f"Starting Standard Training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader, tta=True)

            self.logger.info(
                f"Epoch {epoch}: Train Loss={train_loss:.6f}, Val Loss (TTA)={val_loss:.6f}"
            )

            # Scheduler step
            self.scheduler.step(val_loss)

            # Checkpoint & Early Stopping
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "best_loss": best_loss,
                },
                is_best,
                checkpoint_dir,
            )

            if patience_counter >= early_stopping_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        return best_loss

    def fit_swa(self, train_loader, val_loader, swa_epochs, checkpoint_dir):
        """
        SWA Training phase. Loads best model and refines it.
        """
        # Load best model
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        if not os.path.exists(best_path):
            self.logger.warning(
                "No best model found for SWA. Using current model state."
            )
        else:
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.logger.info("Loaded best model for SWA phase.")

        # Initialize SWA
        swa_model = AveragedModel(self.model)
        swa_scheduler = SWALR(self.optimizer, swa_lr=1e-4)

        self.logger.info(f"Starting SWA Phase for {swa_epochs} epochs...")

        for epoch in range(1, swa_epochs + 1):
            # Train one epoch (Standard training loop)
            self.model.train()
            for images, angles, targets, _ in train_loader:
                images = images.to(self.device)
                angles = angles.to(self.device)
                targets = targets.to(self.device).float().view(-1, 1)
                smooth_targets = self._smooth_labels(targets)

                self.optimizer.zero_grad()
                outputs = self.model(images, angles)
                loss = self.criterion(outputs, smooth_targets)
                loss.backward()
                self.optimizer.step()

            # Update SWA
            swa_model.update_parameters(self.model)
            swa_scheduler.step()

            # Custom BN Update
            self._update_bn(train_loader, swa_model)

            # Validate SWA Model
            # Temporarily swap self.model to use validate() method
            original_model = self.model
            self.model = swa_model
            val_loss = self.validate(val_loader, tta=True)
            self.model = original_model

            self.logger.info(f"SWA Epoch {epoch}: Val Loss (TTA)={val_loss:.6f}")

        # Save Final SWA Model
        save_checkpoint(
            {"state_dict": swa_model.state_dict(), "val_loss": val_loss},
            True,
            checkpoint_dir,
            filename="swa_model.pth",
        )

        # Keep the SWA model as the active model
        self.model = swa_model

    def _update_bn(self, loader, model):
        """
        Custom BN update for dual-input model (image, angle).
        """
        momenta = {}
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.running_mean = torch.zeros_like(module.running_mean)
                module.running_var = torch.ones_like(module.running_var)
                momenta[module] = module.momentum
                module.momentum = None
                module.num_batches_tracked *= 0

        model.train()
        with torch.no_grad():
            for images, angles, _, _ in loader:
                images = images.to(self.device)
                angles = angles.to(self.device)
                model(images, angles)

        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.momentum = momenta[module]

    def predict(self, loader, tta=True):
        """
        Generates predictions for a loader.
        Returns a dict: {id: probability}
        """
        self.model.eval()
        results = {}

        with torch.no_grad():
            for i, data in enumerate(loader):
                # Handle test loader (img, angle, id)
                if len(data) == 3:
                    images, angles, ids = data
                elif len(data) == 4:
                    images, angles, _, ids = data
                else:
                    raise ValueError("Unknown loader format")

                images = images.to(self.device)
                angles = angles.to(self.device)

                if tta:
                    pred_1 = torch.sigmoid(self.model(images, angles))

                    images_h = torch.flip(images, dims=[3])
                    pred_2 = torch.sigmoid(self.model(images_h, angles))

                    images_v = torch.flip(images, dims=[2])
                    pred_3 = torch.sigmoid(self.model(images_v, angles))

                    probs = (pred_1 + pred_2 + pred_3) / 3.0
                else:
                    probs = torch.sigmoid(self.model(images, angles))

                probs = probs.cpu().numpy().flatten()

                for idx, pid in enumerate(ids):
                    results[pid] = float(probs[idx])

        return results

    def generate_submission(self, loader, output_path):
        """
        Generates predictions and saves to CSV.
        """
        self.logger.info("Generating predictions for submission...")
        predictions = self.predict(loader, tta=True)

        df = pd.DataFrame(list(predictions.items()), columns=["id", "is_iceberg"])

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Submission saved to {output_path}")
