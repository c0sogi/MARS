import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import time

from library.config import Config
from torch.cuda.amp import GradScaler, autocast
from library.utils import (
    get_logger,
    calibrate_probabilities,
    pf1_score,
    seed_everything,
)
from library.model import MilTransformerModel

logger = get_logger("trainer")


class Trainer:
    """
    Trainer class for the Multi-View MIL Breast Cancer Detection model.
    Handles training, validation, multi-task loss computation, and checkpointing.
    """

    def __init__(self, model, train_loader, val_loader, device=None):
        """
        Args:
            model (nn.Module): The PyTorch model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            device (torch.device): Device to run training on.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device else Config.DEVICE

        # Move model to device
        self.model.to(self.device)

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Functions
        # Cancer: Binary classification
        self.criterion_cancer = nn.BCEWithLogitsLoss()

        # Density: Multi-class classification (4 classes), ignore -1 (missing)
        self.criterion_density = nn.CrossEntropyLoss(ignore_index=-1)

        # Biopsy: Binary classification
        self.criterion_biopsy = nn.BCEWithLogitsLoss()

        # Best Score Tracking
        self.best_pf1 = -1.0
        self.best_loss = float("inf")
        self.patience_counter = 0

        # Mixed Precision Scaler
        self.scaler = GradScaler()

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        running_cancer_loss = 0.0

        # Lists to store predictions for training metrics (optional, but good for monitoring)
        all_preds = []
        all_targets = []

        start_time = time.time()

        for batch_idx, (images_list, targets) in enumerate(self.train_loader):
            # Move data to device
            # images_list is a list of tensors, need to move each tensor
            images_list = [img.to(self.device) for img in images_list]

            # targets is a dict of tensors
            target_cancer = targets["cancer"].to(self.device).unsqueeze(1)  # (B, 1)
            target_density = targets["density"].to(self.device)  # (B,)
            target_biopsy = targets["biopsy"].to(self.device).unsqueeze(1)  # (B, 1)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with Mixed Precision
            with autocast():
                outputs = self.model(images_list)

                # Compute Losses
                loss_c = self.criterion_cancer(outputs["cancer"], target_cancer)
                loss_d = self.criterion_density(outputs["density"], target_density)
                loss_b = self.criterion_biopsy(outputs["biopsy"], target_biopsy)

                # Weighted Sum
                total_loss = (
                    loss_c
                    + Config.LAMBDA_DENSITY * loss_d
                    + Config.LAMBDA_BIOPSY * loss_b
                )

            # Backward pass with Scaler
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Statistics
            running_loss += total_loss.item()
            running_cancer_loss += loss_c.item()

            # Store for rough training metric
            probs = torch.sigmoid(outputs["cancer"]).detach().cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(target_cancer.detach().cpu().numpy())

        avg_loss = running_loss / len(self.train_loader)
        avg_cancer_loss = running_cancer_loss / len(self.train_loader)

        # Calculate training pF1 (without calibration, just raw performance on balanced set)
        train_pf1 = pf1_score(np.array(all_targets), np.array(all_preds))

        elapsed = time.time() - start_time

        logger.info(
            f"Epoch {epoch_idx+1}/{Config.NUM_EPOCHS} [Train] "
            f"Loss: {avg_loss:.6f} (Cancer: {avg_cancer_loss:.6f}) | "
            f"pF1 (raw): {train_pf1:.6f} | Time: {elapsed:.2f}s"
        )

        return avg_loss

    def validate(self, epoch_idx):
        """
        Runs validation loop.
        """
        self.model.eval()
        running_loss = 0.0

        all_cancer_probs = []
        all_cancer_targets = []

        with torch.no_grad():
            for images_list, targets in self.val_loader:
                # Move data
                images_list = [img.to(self.device) for img in images_list]

                target_cancer = targets["cancer"].to(self.device).unsqueeze(1)
                target_density = targets["density"].to(self.device)
                target_biopsy = targets["biopsy"].to(self.device).unsqueeze(1)

                # Forward
                with autocast():
                    outputs = self.model(images_list)

                # Loss
                loss_c = self.criterion_cancer(outputs["cancer"], target_cancer)
                loss_d = self.criterion_density(outputs["density"], target_density)
                loss_b = self.criterion_biopsy(outputs["biopsy"], target_biopsy)

                total_loss = (
                    loss_c
                    + Config.LAMBDA_DENSITY * loss_d
                    + Config.LAMBDA_BIOPSY * loss_b
                )

                running_loss += total_loss.item()

                # Collect predictions for pF1 calculation
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(outputs["cancer"]).cpu().numpy()
                all_cancer_probs.extend(probs)
                all_cancer_targets.extend(target_cancer.cpu().numpy())

        avg_loss = running_loss / len(self.val_loader)

        # --- Calibration & Metric Calculation ---
        all_cancer_probs = np.array(all_cancer_probs)
        all_cancer_targets = np.array(all_cancer_targets)

        # 1. Calibrate Probabilities
        # The model was trained on a balanced set (approx 50% positive).
        # The validation set (and test set) has approx 2% positive.
        # We must shift the log-odds to account for this prior shift.
        calibrated_probs = calibrate_probabilities(
            all_cancer_probs,
            train_prevalence=Config.TRAIN_PREVALENCE,
            test_prevalence=Config.TEST_PREVALENCE,
        )

        # 2. Compute pF1 Score
        val_pf1 = pf1_score(all_cancer_targets, calibrated_probs)

        logger.info(
            f"Epoch {epoch_idx+1}/{Config.NUM_EPOCHS} [Val]   "
            f"Loss: {avg_loss:.6f} | pF1 (calibrated): {val_pf1:.12f}"
        )

        return avg_loss, val_pf1

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        logger.info(f"Starting training on device: {self.device}")

        for epoch in range(Config.NUM_EPOCHS):
            # Train
            self.train_epoch(epoch)

            # Validate
            val_loss, val_pf1 = self.validate(epoch)

            # Checkpoint & Early Stopping
            # We optimize for pF1 score
            if val_pf1 > self.best_pf1:
                logger.info(
                    f"New best pF1! ({self.best_pf1:.6f} -> {val_pf1:.6f}). Saving model..."
                )
                self.best_pf1 = val_pf1
                self.patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
            else:
                self.patience_counter += 1
                logger.info(
                    f"No improvement. Patience: {self.patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered.")
                break

        logger.info(f"Training complete. Best pF1: {self.best_pf1:.12f}")


def run_training():
    """
    Helper function to initialize datasets, model, and trainer, then run training.
    """
    # 1. Reproducibility
    seed_everything(Config.SEED)

    # 2. DataLoaders
    # We import get_dataloaders inside the function to avoid circular imports if any
    from library.dataset import get_dataloaders

    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # 3. Model
    model = MilTransformerModel()

    # 4. Trainer
    trainer = Trainer(model, train_loader, val_loader)

    # 5. Execute
    trainer.fit()
