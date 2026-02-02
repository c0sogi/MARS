import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict

# Import from library
from library.config import Config
from library.utils import (
    set_seed,
    get_dice_score,
    get_3d_hausdorff,
    keep_largest_component,
)
from library.loss import BCETverskyLoss
from library.model import SegmentationModel


class Trainer:
    """
    Trainer class to manage the training and validation lifecycle of the
    Stomach and Intestines MRI Segmentation model.
    """

    def __init__(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Initialize the Trainer.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(Config.DEVICE)
        self.num_classes = Config.NUM_CLASSES
        self.classes = Config.CLASSES

        # Initialize Model
        self.model = SegmentationModel()
        self.model.to(self.device)

        # Initialize Loss
        self.criterion = BCETverskyLoss()

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
        )

        # Initialize Scaler for Mixed Precision
        # Using torch.amp.GradScaler for compatibility with newer PyTorch versions
        # or torch.cuda.amp.GradScaler for older ones.
        # Given the environment likely supports torch.amp (PyTorch 2.x), we use that.
        self.scaler = torch.amp.GradScaler("cuda")

        # Metrics tracking
        self.best_score = -float("inf")
        self.patience_counter = 0
        self.early_stopping_patience = 5  # Stop if no improvement for 5 epochs

    def train_one_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        start_time = time.time()

        for batch_idx, data in enumerate(self.train_loader):
            images = data["image"].to(self.device, dtype=torch.float32)
            masks = data["mask"].to(self.device, dtype=torch.float32)
            batch_size = images.size(0)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with torch.amp.autocast("cuda", enabled=Config.MIXED_PRECISION):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)

            # Backward Pass with Scaler
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch + 1}/{Config.EPOCHS} | Train Loss: {epoch_loss:.6f} | Time: {elapsed:.2f}s"
        )
        return epoch_loss

    def valid_one_epoch(self, epoch):
        """
        Runs one epoch of validation.
        Aggregates 2D predictions into 3D volumes to compute the competition metric.
        """
        self.model.eval()

        # Dictionary to store predictions and targets for 3D reconstruction
        # Structure: volume_data[case_day] = list of (slice_num, pred_slice, mask_slice)
        volume_data = defaultdict(list)

        with torch.no_grad():
            for data in self.val_loader:
                images = data["image"].to(self.device, dtype=torch.float32)
                masks = data["mask"].cpu().numpy()  # Keep masks on CPU to save GPU mem
                ids = data["id"]

                # Mixed Precision Inference
                with torch.amp.autocast("cuda", enabled=Config.MIXED_PRECISION):
                    logits = self.model(images)
                    probs = torch.sigmoid(logits)

                # Move predictions to CPU
                probs = probs.cpu().numpy()

                # Collect data
                for i in range(len(ids)):
                    sample_id = ids[i]
                    # Parse ID: caseXXX_dayYY_slice_ZZZZ
                    parts = sample_id.split("_")
                    # case_day key: e.g., "case123_day20"
                    case_day = f"{parts[0]}_{parts[1]}"
                    slice_num = int(parts[3])

                    volume_data[case_day].append(
                        {
                            "slice": slice_num,
                            "pred": probs[i],  # (C, H, W)
                            "mask": masks[i],  # (C, H, W)
                        }
                    )

        # Process volumes and calculate metrics
        dice_scores = []
        hausdorff_scores = []

        # Iterate over each reconstructed volume
        for case_day, slices in volume_data.items():
            # Sort slices by slice number to ensure correct Z-ordering
            slices.sort(key=lambda x: x["slice"])

            # Stack slices to create 3D volumes: (D, C, H, W)
            preds_stacked = np.stack([s["pred"] for s in slices], axis=0)
            masks_stacked = np.stack([s["mask"] for s in slices], axis=0)

            # Permute to (C, D, H, W) to iterate over classes
            preds_vol = np.transpose(preds_stacked, (1, 0, 2, 3))
            masks_vol = np.transpose(masks_stacked, (1, 0, 2, 3))

            case_dice = []
            case_hd = []

            for c in range(self.num_classes):
                # Get volume for specific class
                p_vol = preds_vol[c]
                t_vol = masks_vol[c]

                # Threshold to binary
                p_bin = (p_vol > Config.MASK_THRESHOLD).astype(np.uint8)
                t_bin = (t_vol > 0.5).astype(np.uint8)

                # Post-processing: Keep largest connected component
                # This is crucial for Hausdorff distance to avoid penalizing noise
                p_bin_processed = keep_largest_component(p_bin)

                # Calculate Metrics
                d = get_dice_score(p_bin_processed, t_bin)
                h = get_3d_hausdorff(p_bin_processed, t_bin)

                case_dice.append(d)
                case_hd.append(h)

            # Average over classes for this case
            dice_scores.append(np.mean(case_dice))
            hausdorff_scores.append(np.mean(case_hd))

        # Global averages
        mean_dice = np.mean(dice_scores)
        mean_hd = np.mean(hausdorff_scores)

        # Competition Score: 0.4 * Dice + 0.6 * (1 - Hausdorff)
        # Note: Hausdorff is a distance (lower is better).
        # Normalizing logic implies we want to maximize score.
        score = (Config.METRIC_DICE_WEIGHT * mean_dice) + (
            Config.METRIC_HAUSDORFF_WEIGHT * (1.0 - mean_hd)
        )

        print(
            f"Epoch {epoch + 1}/{Config.EPOCHS} | Val Dice: {mean_dice:.6f} | Val HD: {mean_hd:.6f} | Score: {score:.6f}"
        )

        return score

    def save_model(self, path):
        """
        Saves the model state dictionary.
        """
        torch.save(self.model.state_dict(), path)

    def fit(self):
        """
        Main training loop.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(Config.EPOCHS):
            # Training Step
            self.train_one_epoch(epoch)

            # Validation Step
            val_score = self.valid_one_epoch(epoch)

            # Scheduler Step
            self.scheduler.step()

            # Checkpointing
            self.save_model(Config.LAST_MODEL_PATH)

            if val_score > self.best_score:
                print(
                    f"Score Improved ({self.best_score:.6f} -> {val_score:.6f}). Saving Best Model..."
                )
                self.best_score = val_score
                self.save_model(Config.BEST_MODEL_PATH)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                print(
                    f"Score did not improve. Patience: {self.patience_counter}/{self.early_stopping_patience}"
                )

            # Early Stopping
            if self.patience_counter >= self.early_stopping_patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Score: {self.best_score:.6f}")
