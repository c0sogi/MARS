import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import os

from library.config import Config
from library.utils import seed_everything, get_device, log_message
from library.model import WIISNet
from library.dataset import get_dataloader


class Trainer:
    """
    Trainer class for the Weight-Inflated Independent-Slab Network (WIIS-Net).
    Handles the training loop, validation with subject-level aggregation,
    and model checkpointing.
    """

    def __init__(self):
        self.device = get_device()
        self.model = WIISNet().to(self.device)

        # Optimizer and Loss
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.criterion = nn.BCEWithLogitsLoss()

        # State tracking
        self.best_auc = 0.0
        self.patience_counter = 0

    def train_one_epoch(self, train_loader, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

        epoch_loss = running_loss / count if count > 0 else 0.0
        return epoch_loss

    def validate(self, val_loader):
        """
        Runs validation.
        Crucially, this aggregates the 3-slab predictions per subject
        to compute the subject-level ROC AUC.
        """
        self.model.eval()

        all_preds = []
        all_targets = []
        all_ids = []
        running_loss = 0.0
        count = 0

        with torch.no_grad():
            for images, labels, subject_ids in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                count += images.size(0)

                # Convert logits to probabilities
                probs = torch.sigmoid(outputs)

                all_preds.extend(probs.cpu().numpy().flatten())
                all_targets.extend(labels.cpu().numpy().flatten())
                all_ids.extend(subject_ids.numpy().flatten())

        val_loss = running_loss / count if count > 0 else 0.0

        # --- Consensus Aggregation ---
        # Group predictions by Subject ID (BraTS21ID) to handle the 3-slab expansion
        df_val = pd.DataFrame(
            {"BraTS21ID": all_ids, "prob": all_preds, "target": all_targets}
        )

        # Mean prediction per subject
        df_grouped = (
            df_val.groupby("BraTS21ID")
            .agg({"prob": "mean", "target": "first"})  # Target is constant per subject
            .reset_index()
        )

        # Calculate AUC
        try:
            auc = roc_auc_score(df_grouped["target"], df_grouped["prob"])
        except ValueError:
            # Handle edge case with single class in batch
            auc = 0.5

        return val_loss, auc

    def fit(self, load_cached_data=True):
        """
        Main training loop with Early Stopping.
        """
        seed_everything(Config.SEED)

        log_message(f"Initializing training on device: {self.device}")

        # Load Data
        train_loader = get_dataloader("train", load_cached=load_cached_data)
        val_loader = get_dataloader("val", load_cached=load_cached_data)

        log_message(f"Training samples (slabs): {len(train_loader.dataset)}")
        log_message(f"Validation samples (slabs): {len(val_loader.dataset)}")

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss, val_auc = self.validate(val_loader)

            log_message(
                f"Epoch {epoch}/{Config.NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val AUC: {val_auc}"
            )

            # Checkpointing & Early Stopping
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.patience_counter = 0
                log_message(f"New best AUC! Saving model to {Config.MODEL_PATH}")
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
            else:
                self.patience_counter += 1
                if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    log_message(f"Early stopping triggered after {epoch} epochs.")
                    break

        log_message(f"Training complete. Best Validation AUC: {self.best_auc}")

    def predict(self, test_loader):
        """
        Generates predictions for the test set.
        Loads the best model weights before inference.
        """
        # Load best weights
        if os.path.exists(Config.MODEL_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_PATH, map_location=self.device)
            )
            log_message("Loaded best model weights for inference.")
        else:
            log_message("Warning: Best model weights not found. Using current weights.")

        self.model.eval()

        all_preds = []
        all_ids = []

        with torch.no_grad():
            for images, subject_ids in test_loader:
                images = images.to(self.device)

                outputs = self.model(images)
                probs = torch.sigmoid(outputs)

                all_preds.extend(probs.cpu().numpy().flatten())
                all_ids.extend(subject_ids.numpy().flatten())

        # --- Consensus Aggregation ---
        df_test = pd.DataFrame({"BraTS21ID": all_ids, "MGMT_value": all_preds})

        # Mean prediction per subject
        df_submission = (
            df_test.groupby("BraTS21ID").agg({"MGMT_value": "mean"}).reset_index()
        )

        return df_submission
