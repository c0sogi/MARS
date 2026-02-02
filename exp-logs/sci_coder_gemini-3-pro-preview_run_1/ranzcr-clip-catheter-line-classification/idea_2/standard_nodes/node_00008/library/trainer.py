import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from library.config import Config
from library.dataset import CatheterDataset, get_transforms
from library.model import CatheterModel
from library.utils import seed_everything, get_device, calculate_metric


def get_pos_weights(df: pd.DataFrame, load_cached_data: bool = True) -> torch.Tensor:
    """
    Calculates or loads positive weights for BCEWithLogitsLoss to handle class imbalance.
    Implements strict caching logic using .npy format.

    Formula: negative_count / positive_count
    """
    cache_dir = Config.WORKING_DIR
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, "pos_weights.npy")

    # 1. IF load_cached_data is True: Try to load the file.
    if load_cached_data and os.path.exists(cache_path):
        try:
            weights_np = np.load(cache_path)
            return torch.tensor(weights_np, dtype=torch.float32)
        except Exception:
            # If loading fails, proceed to calculate
            pass

    # 2. IF loading fails OR load_cached_data is False: Compute from scratch.
    targets = df[Config.TARGET_COLS].values
    # Count positives and negatives
    pos_counts = np.sum(targets, axis=0)
    neg_counts = len(df) - pos_counts

    # Avoid division by zero
    pos_counts = np.maximum(pos_counts, 1)

    weights_np = neg_counts / pos_counts

    # Save the result to cache
    np.save(cache_path, weights_np)

    return torch.tensor(weights_np, dtype=torch.float32)


class Trainer:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.device = get_device()
        seed_everything(Config.SEED)

        # Load Metadata
        self.train_df = pd.read_csv(Config.TRAIN_METADATA)
        self.val_df = pd.read_csv(Config.VAL_METADATA)

        # Calculate/Load Loss Weights
        # We use the training dataframe for calculating weights
        self.pos_weights = get_pos_weights(self.train_df, load_cached_data=True).to(
            self.device
        )

        # Initialize Datasets
        self.train_dataset = CatheterDataset(
            self.train_df,
            transforms=get_transforms("train"),
            mode="train",
            debug=self.debug,
            debug_size=Config.DEBUG_SAMPLE_SIZE,
        )

        self.val_dataset = CatheterDataset(
            self.val_df,
            transforms=get_transforms("valid"),
            mode="valid",
            debug=self.debug,
            debug_size=Config.DEBUG_SAMPLE_SIZE,
        )

        # Initialize Loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        self.model = CatheterModel(
            model_name=Config.MODEL_NAME, pretrained=Config.PRETRAINED
        )
        self.model.to(self.device)

        # Loss Function
        if Config.USE_POS_WEIGHT:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights)
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Mixed Precision Scaler
        self.scaler = GradScaler(enabled=Config.USE_AMP)

        # Best Score Tracker
        self.best_auc = 0.0

    def train_one_epoch(self, epoch_index):
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            batch_size = images.size(0)

            self.optimizer.zero_grad()

            with autocast(enabled=Config.USE_AMP):
                logits = self.model(images)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            # Gradient Clipping
            if Config.MAX_GRAD_NORM > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), Config.MAX_GRAD_NORM
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        preds_list = []
        targets_list = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                batch_size = images.size(0)

                with autocast(enabled=Config.USE_AMP):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                # Apply sigmoid for metric calculation
                probs = torch.sigmoid(logits)

                preds_list.append(probs.cpu().numpy())
                targets_list.append(labels.cpu().numpy())

        val_loss = running_loss / dataset_size

        preds_arr = np.concatenate(preds_list, axis=0)
        targets_arr = np.concatenate(targets_list, axis=0)

        val_auc = calculate_metric(targets_arr, preds_arr)

        return val_loss, val_auc

    def fit(self, epochs=Config.EPOCHS, patience=3):
        print(f"Starting training for {epochs} epochs on device: {self.device}")

        best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(epoch)
            val_loss, val_auc = self.validate()

            # Print full precision metrics
            print(
                f"Epoch {epoch}/{epochs} - "
                f"Train Loss: {train_loss} - "
                f"Val Loss: {val_loss} - "
                f"Val AUC: {val_auc}"
            )

            # Save best model
            if val_auc > self.best_auc:
                print(
                    f"Validation AUC improved from {self.best_auc} to {val_auc}. Saving model..."
                )
                self.best_auc = val_auc
                torch.save(self.model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs of no improvement."
                )
                break

        print(f"Training complete. Best Validation AUC: {self.best_auc}")
