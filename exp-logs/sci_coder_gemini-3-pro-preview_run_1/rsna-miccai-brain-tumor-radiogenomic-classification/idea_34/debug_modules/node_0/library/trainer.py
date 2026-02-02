import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import (
    WORK_DIR,
    SUBMISSION_PATH,
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
)
from library.utils import get_logger, seed_everything
from library.network import VRAWIVModel

# Initialize Logger
logger = get_logger("trainer")


class Trainer:
    """
    Trainer class for the V-RAWIV Network.
    Handles training, validation, early stopping, and inference.
    """

    def __init__(self):
        self.device = DEVICE
        self.model = VRAWIVModel().to(self.device)

        # Optimizer with aggressive weight decay as per strategy
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Binary Cross Entropy with Logits
        self.criterion = nn.BCEWithLogitsLoss()

        # Training state
        self.best_auc = 0.0
        self.patience_counter = 0
        self.best_model_path = os.path.join(WORK_DIR, "best_model.pth")

    def train_one_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        for images, targets in train_loader:
            images = images.to(self.device)
            targets = targets.to(self.device).unsqueeze(1)  # Shape (B, 1)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            # Track metrics
            running_loss += loss.item() * images.size(0)

            # Apply sigmoid for AUC calculation
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)

        # Handle edge case where batch might have only one class
        try:
            epoch_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            epoch_auc = 0.5

        return epoch_loss, epoch_auc

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(outputs).detach().cpu().numpy()
                all_preds.extend(probs)
                all_targets.extend(targets.cpu().numpy())

        val_loss = running_loss / len(val_loader.dataset)

        try:
            val_auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            val_auc = 0.5

        return val_loss, val_auc

    def fit(self, train_loader, val_loader):
        """
        Main training loop with Early Stopping.
        """
        logger.info(f"Starting training on device: {self.device}")

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, train_auc = self.train_one_epoch(train_loader)
            val_loss, val_auc = self.validate(val_loader)

            logger.info(
                f"Epoch {epoch}/{NUM_EPOCHS} - "
                f"Train Loss: {train_loss:.6f}, Train AUC: {train_auc:.6f} - "
                f"Val Loss: {val_loss:.6f}, Val AUC: {val_auc:.6f}"
            )

            # Early Stopping Logic (Maximize AUC)
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                logger.info(f"New best model saved with AUC: {self.best_auc:.6f}")
            else:
                self.patience_counter += 1
                logger.info(
                    f"No improvement. Patience: {self.patience_counter}/{EARLY_STOPPING_PATIENCE}"
                )

            if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered.")
                break

        logger.info(f"Training complete. Best Validation AUC: {self.best_auc:.6f}")

    def predict_and_submit(self, test_loader):
        """
        Generates predictions for the test set and saves to CSV.
        """
        logger.info("Loading best model for inference...")

        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
        else:
            logger.warning("Best model not found! Using current model state.")

        self.model.eval()
        all_preds = []

        logger.info("Generating predictions...")
        with torch.no_grad():
            for images in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(probs.flatten())

        # Retrieve BraTS21IDs from the dataset dataframe
        # The DataLoader preserves order (shuffle=False for test)
        test_df = test_loader.dataset.df
        ids = test_df["BraTS21ID"].values

        # Construct Submission DataFrame
        submission = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": all_preds})

        # Save to CSV
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {SUBMISSION_PATH}")


def run_training(train_loader, val_loader, test_loader):
    """
    Entry point to run the full training and submission pipeline.
    """
    seed_everything()

    trainer = Trainer()
    trainer.fit(train_loader, val_loader)
    trainer.predict_and_submit(test_loader)
