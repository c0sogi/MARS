import os
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import get_logger, seed_everything
from library.dataset import get_datasets
from library.model import AsymmetricGroupedEfficientNet

logger = get_logger("Trainer")


class Trainer:
    """
    Manages training, validation, and inference for the MGMT promoter methylation prediction task.
    Encapsulates model lifecycle, optimization, metric tracking, and checkpointing.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        self.model = AsymmetricGroupedEfficientNet().to(self.device)

        # Optimizer: AdamW with aggressive weight decay as per idea configuration
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Function: Binary Cross Entropy with Logits
        self.criterion = nn.BCEWithLogitsLoss()

        # Metric Tracking
        self.best_auc = 0.0

    def train_one_epoch(self, train_loader: DataLoader) -> float:
        """
        Runs one epoch of training.

        Args:
            train_loader (DataLoader): The training data loader.

        Returns:
            float: The average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for inputs, labels, _ in train_loader:
            batch_size = inputs.size(0)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0
        return epoch_loss

    def validate(self, val_loader: DataLoader):
        """
        Runs validation loop and computes Loss and ROC AUC.

        Args:
            val_loader (DataLoader): The validation data loader.

        Returns:
            Tuple[float, float]: Average validation loss and validation AUC.
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                batch_size = inputs.size(0)
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                # Apply sigmoid to convert logits to probabilities
                probs = torch.sigmoid(outputs)

                all_labels.extend(labels.cpu().numpy().flatten())
                all_probs.extend(probs.cpu().numpy().flatten())

        epoch_loss = running_loss / dataset_size if dataset_size > 0 else 0.0

        # Compute AUC
        try:
            # AUC requires at least two classes to be present in the set
            if len(np.unique(all_labels)) > 1:
                epoch_auc = roc_auc_score(all_labels, all_probs)
            else:
                # Fallback if batch/set contains only one class
                epoch_auc = 0.5
        except Exception as e:
            logger.warning(f"AUC computation failed: {e}")
            epoch_auc = 0.5

        return epoch_loss, epoch_auc

    def fit(self, load_cached_data: bool = True):
        """
        Orchestrates the training pipeline, including data loading, training loops,
        validation, early stopping, and model checkpointing.

        Args:
            load_cached_data (bool): Whether to attempt loading cached processed data.
        """
        # Ensure reproducibility
        seed_everything(Config.SEED)
        Config.setup_directories()

        logger.info("Initializing datasets...")
        # get_datasets handles the caching logic internally
        train_ds, val_ds, _ = get_datasets(load_cached_data=load_cached_data)

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(self.device.type == "cuda"),
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(self.device.type == "cuda"),
        )

        logger.info(
            f"Starting training on {self.device} for {Config.NUM_EPOCHS} epochs."
        )

        patience_counter = 0

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            start_time = time.time()

            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_auc = self.validate(val_loader)

            elapsed = time.time() - start_time

            # Print metrics with full precision as requested
            logger.info(f"Epoch {epoch}/{Config.NUM_EPOCHS} - Time: {elapsed}s")
            logger.info(f"Train Loss: {train_loss}")
            logger.info(f"Val Loss: {val_loss}")
            logger.info(f"Val AUC: {val_auc}")

            # Checkpointing based on AUC
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                logger.info(f"New best model saved to {Config.MODEL_SAVE_PATH}")
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(
                    f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
                )

            # Early Stopping
            if patience_counter >= Config.PATIENCE:
                logger.info("Early stopping triggered.")
                break

        logger.info(f"Training complete. Best Val AUC: {self.best_auc}")

    def predict(self, load_cached_data: bool = True):
        """
        Generates predictions for the test set using the best saved model.
        Implements Test-Time Augmentation (TTA) and saves results to CSV.

        Args:
            load_cached_data (bool): Whether to attempt loading cached processed data.
        """
        Config.setup_directories()

        # Load best model weights
        if os.path.exists(Config.MODEL_SAVE_PATH):
            logger.info(f"Loading model from {Config.MODEL_SAVE_PATH}")
            self.model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )
        else:
            logger.warning("No checkpoint found! Predicting with untrained model.")

        self.model.eval()

        _, _, test_ds = get_datasets(load_cached_data=load_cached_data)

        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(self.device.type == "cuda"),
        )

        results = []
        logger.info("Starting inference with Test-Time Augmentation (TTA)...")

        with torch.no_grad():
            for inputs, _, subject_ids in test_loader:
                inputs = inputs.to(self.device)

                # TTA 1: Original
                out = self.model(inputs)
                prob = torch.sigmoid(out)

                # TTA 2: Horizontal Flip (Width is dim 3: N, C, H, W)
                inputs_h = torch.flip(inputs, dims=[3])
                out_h = self.model(inputs_h)
                prob_h = torch.sigmoid(out_h)

                # TTA 3: Vertical Flip (Height is dim 2)
                inputs_v = torch.flip(inputs, dims=[2])
                out_v = self.model(inputs_v)
                prob_v = torch.sigmoid(out_v)

                # Average probabilities
                avg_prob = (prob + prob_h + prob_v) / 3.0

                # Convert to numpy list
                avg_prob_np = avg_prob.cpu().numpy().flatten()

                for sid, p in zip(subject_ids, avg_prob_np):
                    results.append({"BraTS21ID": sid, "MGMT_value": p})

        # Save submission
        df_sub = pd.DataFrame(results)
        # Ensure correct column order
        df_sub = df_sub[["BraTS21ID", "MGMT_value"]]
        df_sub.to_csv(Config.SUBMISSION_FILE, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")
