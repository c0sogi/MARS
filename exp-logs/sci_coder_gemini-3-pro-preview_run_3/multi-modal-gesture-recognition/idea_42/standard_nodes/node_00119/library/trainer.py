import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.dataset import GestureDataset
from library.model import LSMCN
from library.utils import (
    LogSpaceSmoothingLoss,
    decode_predictions,
    compute_levenshtein,
    save_predictions,
)


class Trainer:
    def __init__(self):
        # 1. Setup & Reproducibility
        Config.seed_everything(Config.SEED)
        self.device = torch.device(Config.DEVICE)

        # 2. Initialize Model
        self.model = LSMCN().to(self.device)

        # 3. Define Loss Functions
        # Class Weights: Background (0) gets 0.2, others 1.0
        class_weights = torch.ones(Config.NUM_CLASSES, device=self.device)
        class_weights[Config.BACKGROUND_CLASS_ID] = Config.BACKGROUND_WEIGHT

        # We use NLLLoss because model outputs Softmax probabilities, so we will take log(p)
        self.criterion_cls = nn.NLLLoss(weight=class_weights, reduction="mean")
        self.criterion_smooth = LogSpaceSmoothingLoss(
            threshold=Config.SMOOTHING_THRESHOLD
        )

        # 4. Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # 5. Data Loaders
        # Train: Shuffle=True, Batch Size from Config
        train_dataset = GestureDataset(split="train", load_cached_data=True)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == "cuda" else False,
            drop_last=True,
        )

        # Val: Shuffle=False, Batch Size=1 (Full Sequence Inference)
        val_dataset = GestureDataset(split="val", load_cached_data=True)
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        # Test: Shuffle=False, Batch Size=1
        test_dataset = GestureDataset(split="test", load_cached_data=True)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        self.best_model_path = os.path.join(Config.WORK_DIR, "best_model.pth")

    def train(self, num_epochs=Config.NUM_EPOCHS, patience=10):
        print(f"Starting training on {self.device} for {num_epochs} epochs.")

        best_score = float("inf")
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # Train Step
            train_loss = self.train_epoch()

            # Validation Step
            val_score, val_dist_sum, val_gt_count = self.validate()

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Time: {elapsed:.1f}s | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Score (Lev/GT): {val_score}"
            )

            # Early Stopping & Checkpointing
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  >>> New Best Model Saved (Score: {best_score})")
            else:
                patience_counter += 1
                print(f"  >>> No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training finished. Best Validation Score: {best_score}")

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for x, y, _ in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # Forward Pass: Returns probabilities (Softmax)
            p1, p2, p3 = self.model(x)

            # Convert to Log-Probabilities for NLLLoss and Smoothing
            # Add epsilon for numerical stability
            eps = 1e-10
            log_p1 = torch.log(p1 + eps)
            log_p2 = torch.log(p2 + eps)
            log_p3 = torch.log(p3 + eps)

            # Reshape for NLLLoss: (Batch, Classes, Time)
            # Target is (Batch, Time)
            loss_cls_1 = self.criterion_cls(log_p1.transpose(1, 2), y)
            loss_cls_2 = self.criterion_cls(log_p2.transpose(1, 2), y)
            loss_cls_3 = self.criterion_cls(log_p3.transpose(1, 2), y)

            # Smoothing Loss (only for refinement stages)
            loss_smooth_2 = self.criterion_smooth(log_p2)
            loss_smooth_3 = self.criterion_smooth(log_p3)

            # Cascaded Loss Calculation
            # L = L_cls + lambda * L_smooth
            loss_stage1 = loss_cls_1
            loss_stage2 = loss_cls_2 + Config.SMOOTHING_LAMBDA * loss_smooth_2
            loss_stage3 = loss_cls_3 + Config.SMOOTHING_LAMBDA * loss_smooth_3

            loss = loss_stage1 + loss_stage2 + loss_stage3

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self):
        """
        Performs full-sequence inference on the validation set.
        Computes the Levenshtein distance metric.
        """
        self.model.eval()

        total_distance = 0
        total_gt_gestures = 0

        with torch.no_grad():
            for x, y, _ in self.val_loader:
                x = x.to(self.device)
                # y is (1, Time)

                # Forward
                _, _, p3 = self.model(x)

                # Decode Prediction (Batch size is 1)
                # p3 is (1, Time, Classes) -> squeeze to (Time, Classes)
                pred_seq = decode_predictions(
                    p3.squeeze(0), min_duration=Config.MIN_GESTURE_DURATION
                )

                # Decode Ground Truth
                # y is (1, Time) -> squeeze to (Time,)
                y_np = y.cpu().numpy().squeeze(0)

                # We can use decode_predictions logic on GT labels too,
                # effectively performing RLE and removing background.
                # However, decode_predictions expects probabilities.
                # We can manually do RLE on the labels.

                # Manual RLE for GT
                gt_seq = []
                from itertools import groupby

                for k, g in groupby(y_np):
                    if k != Config.BACKGROUND_CLASS_ID:
                        gt_seq.append(int(k))

                # Compute Metric
                dist = compute_levenshtein(pred_seq, gt_seq)

                total_distance += dist
                total_gt_gestures += len(gt_seq)

        # Avoid division by zero
        if total_gt_gestures == 0:
            score = 0.0
        else:
            score = total_distance / total_gt_gestures

        return score, total_distance, total_gt_gestures

    def predict_test(self):
        """
        Generates predictions for the test set and saves to CSV.
        """
        print("Starting inference on test set...")

        # Load Best Model
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
            print("Loaded best model checkpoint.")
        else:
            print("Warning: No checkpoint found. Using current model weights.")

        self.model.eval()
        predictions = {}

        with torch.no_grad():
            for x, _, sample_ids in self.test_loader:
                x = x.to(self.device)

                # Forward
                _, _, p3 = self.model(x)

                # Decode
                # Batch size is 1, sample_ids is tuple of length 1
                sample_id = sample_ids[0]
                pred_seq = decode_predictions(
                    p3.squeeze(0), min_duration=Config.MIN_GESTURE_DURATION
                )

                predictions[sample_id] = pred_seq

        # Save
        save_predictions(predictions, Config.SUBMISSION_FILE)
        print(f"Predictions saved to {Config.SUBMISSION_FILE}")
