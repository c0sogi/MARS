import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import (
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    SEED,
)
from library.utils import setup_logger, seed_everything, ensure_dir
from library.data_loader import get_dataloaders
from library.model import AsymmetricEfficientNet, predict_with_tta

logger = setup_logger("train")


class Trainer:
    """
    Manages the training and validation process for the Asymmetric EfficientNet.
    Includes logic for optimization, metric tracking, early stopping, and data integrity checks.
    """

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimization components
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Training state
        self.best_val_auc = 0.0
        self.patience_counter = 0

        # Circuit Breaker Statistics
        self.total_samples_seen = 0
        self.corrupt_samples_seen = 0
        self.corruption_threshold = 0.01  # 1%

    def train_epoch(self):
        """
        Runs one epoch of training.
        Returns average loss and ROC AUC for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        for inputs, targets in self.train_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).unsqueeze(1)

            # -----------------------------------------------------------------
            # Data Corruption Circuit Breaker
            # -----------------------------------------------------------------
            # Check for zero-filled tensors which indicate loading failure.
            # We sum absolute values across the flattened volume dimensions.
            batch_size = inputs.size(0)
            volume_sums = inputs.view(batch_size, -1).abs().sum(dim=1)
            n_corrupt = (volume_sums == 0).sum().item()

            self.total_samples_seen += batch_size
            self.corrupt_samples_seen += n_corrupt

            if self.total_samples_seen > 0:
                current_rate = self.corrupt_samples_seen / self.total_samples_seen
                if current_rate > self.corruption_threshold:
                    raise RuntimeError(
                        f"Circuit Breaker Triggered: Data corruption rate {current_rate:.4f} "
                        f"exceeds threshold {self.corruption_threshold}."
                    )

            # -----------------------------------------------------------------
            # Optimization Step
            # -----------------------------------------------------------------
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Store predictions for AUC calculation
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(self.train_loader.dataset)

        # Handle cases with single-class batches or other AUC calculation issues
        try:
            epoch_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            epoch_auc = 0.5

        return epoch_loss, epoch_auc

    def validate(self):
        """
        Runs validation on the validation set.
        Returns average loss and ROC AUC.
        """
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)

                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(self.val_loader.dataset)

        try:
            epoch_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            epoch_auc = 0.5

        return epoch_loss, epoch_auc

    def fit(self):
        """
        Executes the full training loop with Early Stopping.
        """
        logger.info("Starting training loop...")

        for epoch in range(EPOCHS):
            train_loss, train_auc = self.train_epoch()
            val_loss, val_auc = self.validate()

            # Log with full precision
            logger.info(
                f"Epoch {epoch+1}/{EPOCHS} - "
                f"Train Loss: {train_loss} - Train AUC: {train_auc} - "
                f"Val Loss: {val_loss} - Val AUC: {val_auc}"
            )

            # Early Stopping & Checkpointing
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.patience_counter = 0
                torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
                logger.info(f"New best model saved with Val AUC: {self.best_val_auc}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= PATIENCE:
                    logger.info(
                        f"Early stopping triggered after {self.patience_counter} epochs without improvement."
                    )
                    break


def run_training_pipeline(debug=False, max_samples=None):
    """
    Main entry point for the training and submission pipeline.
    """
    # 1. Setup
    seed_everything(SEED)
    ensure_dir(os.path.dirname(MODEL_SAVE_PATH))
    ensure_dir(os.path.dirname(SUBMISSION_PATH))

    device = torch.device(DEVICE)
    logger.info(f"Using device: {device}")

    # 2. Data Loading
    logger.info("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=debug, max_samples=max_samples
    )

    # 3. Model Initialization
    logger.info("Initializing AsymmetricEfficientNet...")
    model = AsymmetricEfficientNet().to(device)

    # 4. Training
    trainer = Trainer(model, train_loader, val_loader, device)

    try:
        trainer.fit()
    except RuntimeError as e:
        logger.error(f"Training aborted: {e}")
        return

    # 5. Inference
    logger.info("Loading best model for inference...")
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    else:
        logger.warning("No model file found. Using current model state.")

    logger.info("Generating predictions with Test-Time Augmentation (TTA)...")
    test_ids, preds = predict_with_tta(model, test_loader, device)

    # 6. Submission
    logger.info("Saving submission...")
    submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": preds})
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {SUBMISSION_PATH}")
