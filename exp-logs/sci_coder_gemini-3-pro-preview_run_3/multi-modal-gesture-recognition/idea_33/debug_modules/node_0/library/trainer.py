import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from itertools import groupby

from library.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    BACKGROUND_CLASS_WEIGHT,
    SMOOTHING_LOSS_WEIGHT,
    MODEL_SAVE_PATH,
    NUM_CLASSES,
    WINDOW_SIZE,
    SEED,
    seed_everything,
)
from library.model import RGHCMN
from library.data_loader import get_data_loaders
from library.utils import (
    LogSpaceSmoothingLoss,
    compute_levenshtein_ratio,
    filter_short_segments,
)


class ModelTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed_everything(SEED)

        # Initialize Model
        self.model = RGHCMN().to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

        # Loss Functions
        # Class weights: 0.2 for background (index 0), 1.0 for gestures
        class_weights = torch.ones(NUM_CLASSES, device=self.device)
        class_weights[0] = BACKGROUND_CLASS_WEIGHT

        self.ce_criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.smooth_criterion = LogSpaceSmoothingLoss()

        # Training State
        self.best_val_metric = float("inf")
        self.patience_counter = 0

    def train(self):
        print(f"Starting training on device: {self.device}")

        # Load Data
        train_loader, val_loader, _ = get_data_loaders(batch_size=BATCH_SIZE)

        for epoch in range(1, NUM_EPOCHS + 1):
            # Training Step
            train_loss = self.train_epoch(train_loader, epoch)

            # Validation Step
            val_metric = self.validate(val_loader)

            print(
                f"Epoch {epoch}/{NUM_EPOCHS} - Train Loss: {train_loss:.6f} - Val Levenshtein: {val_metric:.6f}"
            )

            # Early Stopping & Model Saving
            if val_metric < self.best_val_metric:
                self.best_val_metric = val_metric
                self.patience_counter = 0
                self.save_model()
                print(f"New best model saved with metric: {val_metric:.6f}")
            else:
                self.patience_counter += 1
                print(
                    f"No improvement. Patience: {self.patience_counter}/{EARLY_STOPPING_PATIENCE}"
                )

            if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_x, batch_y, _, _ in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(batch_x)

            # Unpack logits
            logits_1 = outputs["logits_1"]  # (B, T, C)
            logits_2 = outputs["logits_2"]
            logits_3 = outputs["logits_3"]

            # Reshape for CrossEntropy (B*T, C) vs (B*T)
            # Or use (B, C, T) vs (B, T)
            # PyTorch CE expects (N, C, d1...) and Target (N, d1...)

            # Permute to (B, C, T) for CE Loss
            logits_1_t = logits_1.permute(0, 2, 1)
            logits_2_t = logits_2.permute(0, 2, 1)
            logits_3_t = logits_3.permute(0, 2, 1)

            # 1. Classification Losses (Deep Supervision)
            loss_ce_1 = self.ce_criterion(logits_1_t, batch_y)
            loss_ce_2 = self.ce_criterion(logits_2_t, batch_y)
            loss_ce_3 = self.ce_criterion(logits_3_t, batch_y)

            # 2. Smoothing Losses (Log-Space) for Stage 2 & 3
            # LogSoftmax for smoothing input
            log_probs_2 = F.log_softmax(logits_2, dim=2)
            log_probs_3 = F.log_softmax(logits_3, dim=2)

            loss_smooth_2 = self.smooth_criterion(log_probs_2)
            loss_smooth_3 = self.smooth_criterion(log_probs_3)

            # Total Cascaded Loss
            loss = (
                loss_ce_1
                + loss_ce_2
                + SMOOTHING_LOSS_WEIGHT * loss_smooth_2
                + loss_ce_3
                + SMOOTHING_LOSS_WEIGHT * loss_smooth_3
            )

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self, loader):
        """
        Validates by reconstructing full sequences from sliding windows
        and computing the Levenshtein distance against ground truth.
        """
        self.model.eval()
        dataset = loader.dataset

        # Prepare accumulators for sequence reconstruction (CPU)
        # Map sample_idx -> tensor buffer
        sample_probs = {}
        sample_counts = {}

        # Initialize buffers
        for i, sample_id in enumerate(dataset.sample_ids):
            start, end = dataset.sample_boundaries[i]
            length = end - start
            sample_probs[i] = torch.zeros((length, NUM_CLASSES), dtype=torch.float)
            sample_counts[i] = torch.zeros((length, 1), dtype=torch.float)

        with torch.no_grad():
            for batch_x, _, batch_indices, batch_starts in loader:
                batch_x = batch_x.to(self.device)

                # Forward pass - use Stage 3 for final prediction
                outputs = self.model(batch_x)
                logits = outputs["logits_3"]
                probs = F.softmax(logits, dim=2).cpu()

                # Accumulate
                for k in range(len(batch_indices)):
                    s_idx = batch_indices[k].item()
                    r_start = batch_starts[k].item()

                    total_len = sample_probs[s_idx].shape[0]
                    valid_len = min(WINDOW_SIZE, total_len - r_start)

                    sample_probs[s_idx][r_start : r_start + valid_len] += probs[
                        k, :valid_len, :
                    ]
                    sample_counts[s_idx][r_start : r_start + valid_len] += 1.0

        # Generate Predictions and Ground Truths
        all_preds = []
        all_gts = []

        for i in range(len(dataset.sample_ids)):
            # 1. Prediction
            counts = sample_counts[i]
            counts[counts == 0] = 1.0  # Safety
            avg_probs = sample_probs[i] / counts

            frame_preds = torch.argmax(avg_probs, dim=1).numpy()
            pred_seq = filter_short_segments(frame_preds)
            all_preds.append(pred_seq)

            # 2. Ground Truth
            # Extract dense labels from dataset
            global_start, global_end = dataset.sample_boundaries[i]
            dense_labels = dataset.all_labels[global_start:global_end]

            # Convert dense labels to sequence (RLE excluding 0)
            gt_seq = [k for k, g in groupby(dense_labels) if k != 0]
            all_gts.append(gt_seq)

        # Compute Metric
        score = compute_levenshtein_ratio(all_preds, all_gts)
        return score

    def save_model(self):
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        torch.save(self.model.state_dict(), MODEL_SAVE_PATH)
