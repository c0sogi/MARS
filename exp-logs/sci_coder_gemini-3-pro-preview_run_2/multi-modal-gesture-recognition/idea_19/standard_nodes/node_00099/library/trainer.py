import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import (
    set_seed,
    compute_levenshtein_distance,
    median_filter_predictions,
    decode_predictions,
)


class Trainer:
    """
    Trainer class for the RSG-CRCN model.
    Handles training, validation, loss computation, and checkpointing.
    """

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Weights
        self.class_weights = torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float32).to(
            self.device
        )

        # Loss Functions
        # We use NLLLoss because we will take log of softmax probabilities manually
        self.criterion_cls = nn.NLLLoss(weight=self.class_weights, reduction="none")
        self.criterion_bnd = nn.BCELoss(reduction="none")

        # Best Metric Tracking (Lower is better for Levenshtein Error Rate)
        self.best_metric = float("inf")
        self.patience_counter = 0

    def compute_loss(self, outputs, labels, boundaries, mask):
        """
        Computes the Multi-Stage Deep Supervision loss.

        Args:
            outputs (dict): Dictionary containing stage outputs.
            labels (torch.Tensor): Ground truth labels (B, T).
            boundaries (torch.Tensor): Ground truth boundaries (B, T).
            mask (torch.Tensor): Sequence mask (B, T).

        Returns:
            torch.Tensor: Total weighted loss.
            dict: Dictionary of loss components for logging.
        """
        total_loss = 0.0
        loss_components = {"cls": 0.0, "bnd": 0.0, "smooth": 0.0}

        # Flatten mask for selection
        # mask: (B, T) -> (B*T) boolean
        mask_flat = mask.view(-1).bool()
        num_valid = mask_flat.sum()

        if num_valid == 0:
            return (
                torch.tensor(0.0, requires_grad=True).to(self.device),
                loss_components,
            )

        # Iterate over stages
        stages = ["stage1", "stage2", "stage3"]

        for stage_name in stages:
            # p_cls: (B, C, T), p_bnd: (B, 1, T)
            p_cls, p_bnd = outputs[stage_name]

            # --- 1. Classification Loss ---
            # Permute p_cls to (B, T, C) for flattening
            p_cls_t = p_cls.permute(0, 2, 1)

            # Flatten: (B*T, C)
            p_cls_flat = p_cls_t.reshape(-1, Config.NUM_CLASSES)
            labels_flat = labels.view(-1)

            # Select valid frames
            valid_p_cls = p_cls_flat[mask_flat]
            valid_labels = labels_flat[mask_flat]

            # Log Softmax for NLLLoss (Add epsilon for stability)
            log_p_cls = torch.log(valid_p_cls + 1e-8)

            loss_cls = self.criterion_cls(log_p_cls, valid_labels).mean()

            # --- 2. Boundary Loss ---
            # p_bnd: (B, 1, T) -> (B, T)
            p_bnd_sq = p_bnd.squeeze(1)
            p_bnd_flat = p_bnd_sq.view(-1)
            boundaries_flat = boundaries.view(-1)

            valid_p_bnd = p_bnd_flat[mask_flat]
            valid_boundaries = boundaries_flat[mask_flat]

            loss_bnd = self.criterion_bnd(valid_p_bnd, valid_boundaries).mean()

            # --- 3. Smoothing Loss (Unclamped MSE) ---
            # Calculate frame-to-frame difference: (P_t - P_{t-1})^2
            # p_cls: (B, C, T)
            # Diff along time axis (dim 2)
            # We compute diff for t=1..T-1
            diff = p_cls[:, :, 1:] - p_cls[:, :, :-1]
            diff_sq = diff**2

            # Mask for smoothing (B, T-1)
            # Valid if both t and t-1 are valid
            mask_smooth = mask[:, 1:] * mask[:, :-1]

            # Expand mask for channels: (B, 1, T-1)
            mask_smooth_exp = mask_smooth.unsqueeze(1)

            # Sum squared diffs over valid regions
            smooth_loss_sum = (diff_sq * mask_smooth_exp).sum()

            # Normalize by number of valid transitions * channels
            num_valid_transitions = mask_smooth.sum()
            if num_valid_transitions > 0:
                loss_smooth = smooth_loss_sum / (
                    num_valid_transitions * Config.NUM_CLASSES
                )
            else:
                loss_smooth = torch.tensor(0.0).to(self.device)

            # --- Aggregate Stage Loss ---
            stage_loss = (
                Config.LAMBDA_CLS * loss_cls
                + Config.LAMBDA_BND * loss_bnd
                + Config.LAMBDA_SMOOTH * loss_smooth
            )

            total_loss += stage_loss

            # Accumulate components for logging (averaged across stages)
            loss_components["cls"] += loss_cls.item()
            loss_components["bnd"] += loss_bnd.item()
            loss_components["smooth"] += loss_smooth.item()

        # Average components for display
        for k in loss_components:
            loss_components[k] /= len(stages)

        return total_loss, loss_components

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (features, labels, boundaries, mask) in enumerate(
            self.train_loader
        ):
            features = features.to(self.device)
            labels = labels.to(self.device)
            boundaries = boundaries.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(features, mask)
            loss, _ = self.compute_loss(outputs, labels, boundaries, mask)

            loss.backward()

            # Gradient clipping to prevent explosion in LSTM
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        print(f"Epoch [{epoch}/{Config.NUM_EPOCHS}] Train Loss: {avg_loss:.6f}")
        return avg_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0

        total_dist = 0
        total_gt_gestures = 0

        with torch.no_grad():
            for features, labels, boundaries, mask in self.val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                boundaries = boundaries.to(self.device)
                mask = mask.to(self.device)

                outputs = self.model(features, mask)
                loss, _ = self.compute_loss(outputs, labels, boundaries, mask)
                running_loss += loss.item()

                # --- Metrics Calculation ---
                # Use Stage 3 output for predictions
                p_cls_s3, _ = outputs["stage3"]  # (B, C, T)

                # Get predictions: (B, T)
                preds = torch.argmax(p_cls_s3, dim=1).cpu().numpy()

                # Get ground truth: (B, T)
                gt_labels = labels.cpu().numpy()

                # Iterate over batch
                batch_size = preds.shape[0]
                lengths = mask.sum(dim=1).long().cpu().numpy()

                for i in range(batch_size):
                    length = lengths[i]

                    # Extract valid sequence
                    pred_seq_raw = preds[i, :length]
                    gt_seq_raw = gt_labels[i, :length]

                    # 1. Smoothing
                    pred_seq_smooth = median_filter_predictions(
                        pred_seq_raw, kernel_size=15
                    )

                    # 2. Decode to Gesture List
                    pred_gestures = decode_predictions(pred_seq_smooth)
                    gt_gestures = decode_predictions(gt_seq_raw)

                    # 3. Levenshtein Distance
                    dist = compute_levenshtein_distance(pred_gestures, gt_gestures)

                    total_dist += dist
                    total_gt_gestures += len(gt_gestures)

        avg_loss = running_loss / len(self.val_loader)

        # Avoid division by zero
        if total_gt_gestures == 0:
            error_rate = 1.0  # Worst case
        else:
            error_rate = total_dist / total_gt_gestures

        print(f"Validation Loss: {avg_loss:.10f}")
        print(f"Validation Levenshtein Error Rate: {error_rate:.10f}")

        return avg_loss, error_rate

    def fit(self):
        print(f"Starting training on device: {self.device}")

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            self.train_epoch(epoch)
            val_loss, val_error_rate = self.validate()

            # Early Stopping Check based on Error Rate
            if val_error_rate < self.best_metric:
                print(
                    f"Metric improved from {self.best_metric:.10f} to {val_error_rate:.10f}. Saving model..."
                )
                self.best_metric = val_error_rate
                self.patience_counter = 0

                # Save Best Model
                save_path = os.path.join(Config.SUBMISSION_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
            else:
                self.patience_counter += 1
                print(
                    f"No improvement. Patience: {self.patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

                if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Error Rate: {self.best_metric:.10f}")
