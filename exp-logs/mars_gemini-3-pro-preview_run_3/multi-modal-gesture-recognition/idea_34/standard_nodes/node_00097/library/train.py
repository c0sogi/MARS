import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import json
from collections import defaultdict

from library import config
from library import model
from library import data_loader
from library import utils

# Set seeds for reproducibility
torch.manual_seed(config.SEED)
np.random.seed(config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(config.SEED)


class Trainer:
    def __init__(self, limit=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.limit = limit

        # Data Loaders
        self.train_loader, self.val_loader = data_loader.get_dataloaders(
            batch_size=config.BATCH_SIZE, num_workers=4, limit=self.limit
        )

        # Model
        self.model = model.PG_HCKN().to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Loss Functions
        class_weights = utils.compute_class_weights(self.device)
        self.criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
        self.criterion_smooth = utils.TruncatedMSELoss(
            threshold=config.TRUNCATION_THRESHOLD
        )

        # Validation Metadata for Sequence Reconstruction
        # We need to map window predictions back to full sequences
        self.val_sample_map = {
            item["sample_id"]: item["num_frames"]
            for item in self.val_loader.dataset.sample_map
        }

        # Load Ground Truth Labels for Validation
        self.val_gt = self._load_ground_truth(config.VAL_METADATA_PATH)

    def _load_ground_truth(self, metadata_path):
        """Loads ground truth label sequences for validation samples."""
        df = pd.read_csv(metadata_path)
        gt_dict = {}
        for _, row in df.iterrows():
            sid = row["sample_id"]
            # Reconstruct label sequence
            # Note: We only need the list of gesture IDs for Levenshtein
            labels_json = row["labels"]
            if isinstance(labels_json, str):
                try:
                    gestures = json.loads(labels_json)
                    # Sort by begin time just in case
                    gestures.sort(key=lambda x: x["begin"])
                    gt_ids = [g["id"] for g in gestures]
                    gt_dict[sid] = gt_ids
                except:
                    gt_dict[sid] = []
            else:
                gt_dict[sid] = []
        return gt_dict

    def compute_loss(self, outputs, targets):
        """
        Computes the cascaded loss:
        L = CE(S1) + CE(S2) + CE(S3) + Smooth(S2) + Smooth(S3)
        """
        # Unpack outputs
        logits_1 = outputs["stage1"]
        logits_2 = outputs["stage2"]
        logits_3 = outputs["stage3"]

        # Flatten for CE Loss: (B*T, C) vs (B*T)
        B, T, C = logits_1.shape
        targets_flat = targets.view(-1)

        loss_ce_1 = self.criterion_ce(logits_1.reshape(-1, C), targets_flat)
        loss_ce_2 = self.criterion_ce(logits_2.reshape(-1, C), targets_flat)
        loss_ce_3 = self.criterion_ce(logits_3.reshape(-1, C), targets_flat)

        # Smoothing Loss (Log-Space Truncated MSE)
        # Apply to Stage 2 and 3
        log_probs_2 = F.log_softmax(logits_2, dim=2)
        log_probs_3 = F.log_softmax(logits_3, dim=2)

        loss_smooth_2 = self.criterion_smooth(log_probs_2)
        loss_smooth_3 = self.criterion_smooth(log_probs_3)

        # Total Loss
        total_loss = (
            loss_ce_1
            + loss_ce_2
            + loss_ce_3
            + config.SMOOTHING_LOSS_WEIGHT * loss_smooth_2
            + config.SMOOTHING_LOSS_WEIGHT * loss_smooth_3
        )

        return total_loss

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (features, labels, _) in enumerate(self.train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(features)
            loss = self.compute_loss(outputs, labels)

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), config.GRADIENT_CLIP_VAL
            )

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate(self):
        """
        Performs validation by reconstructing full sequences from sliding windows
        and calculating the Levenshtein distance.
        """
        self.model.eval()

        # Buffers for full sequence reconstruction
        # sid -> np.array of shape (num_frames, num_classes)
        seq_probs = {
            sid: np.zeros((length, config.NUM_CLASSES), dtype=np.float32)
            for sid, length in self.val_sample_map.items()
        }
        # Count buffer for averaging overlapping windows
        seq_counts = {
            sid: np.zeros((length, 1), dtype=np.float32)
            for sid, length in self.val_sample_map.items()
        }

        # Access window indices to map batch items back to global position
        # Assuming val_loader preserves order (shuffle=False)
        window_indices = self.val_loader.dataset.window_indices
        global_idx = 0

        with torch.no_grad():
            for features, _, sample_ids in self.val_loader:
                features = features.to(self.device)

                # Forward pass
                outputs = self.model(features)
                # Use Stage 3 predictions for final result
                logits = outputs["stage3"]
                probs = F.softmax(logits, dim=2).cpu().numpy()

                batch_size = features.size(0)

                for i in range(batch_size):
                    # Get metadata for this window
                    if global_idx >= len(window_indices):
                        break

                    start, end, sid, needs_padding = window_indices[global_idx]

                    # Current window probabilities
                    window_prob = probs[i]  # (WindowSize, Classes)

                    # Handle padding if it was applied
                    if needs_padding:
                        # Determine actual length of data in this window
                        # The dataset padding logic pads the END of the sequence
                        # We need to slice off the padding
                        actual_len = self.val_sample_map[sid]
                        # If window is larger than sample, we take actual_len
                        # But start is 0 in this case
                        valid_len = min(config.WINDOW_SIZE, actual_len)
                        window_prob = window_prob[:valid_len]

                        # Update indices
                        target_slice = slice(0, valid_len)
                    else:
                        target_slice = slice(start, end)

                    # Accumulate
                    if sid in seq_probs:
                        # Ensure shapes match (handle edge cases)
                        target_len = seq_probs[sid][target_slice].shape[0]
                        source_len = window_prob.shape[0]

                        if target_len == source_len:
                            seq_probs[sid][target_slice] += window_prob
                            seq_counts[sid][target_slice] += 1.0

                    global_idx += 1

        # Decode and Compute Metric
        total_lev_dist = 0
        total_gestures = 0

        for sid, prob_sum in seq_probs.items():
            count = seq_counts[sid]
            # Avoid division by zero (should not happen if covered)
            count[count == 0] = 1.0

            avg_probs = prob_sum / count

            # Frame-wise prediction
            frame_preds = np.argmax(avg_probs, axis=1)

            # Run Length Encoding to get gesture list
            predicted_gestures = utils.run_length_encoding(
                frame_preds, min_duration=config.MIN_GESTURE_DURATION
            )

            # Ground Truth
            target_gestures = self.val_gt.get(sid, [])

            # Levenshtein Distance
            dist = utils.levenshtein_distance(predicted_gestures, target_gestures)

            total_lev_dist += dist
            total_gestures += len(target_gestures)

        # Metric: Error Rate (Total Dist / Total GT Gestures)
        # If total_gestures is 0 (empty validation set?), avoid error
        if total_gestures == 0:
            return 0.0

        score = total_lev_dist / total_gestures
        return score

    def run(self, epochs=config.NUM_EPOCHS):
        best_score = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(config.WORKING_DIR, "idea_34", "best_model.pth")

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_score = self.validate()

            print(
                f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Score (Lev/N): {val_score:.6f}"
            )

            # Checkpoint
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                # print(f"  New best model saved to {best_model_path}")
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        return best_score


def train_model(limit=None, epochs=config.NUM_EPOCHS):
    """
    Main entry point to train the model.

    Args:
        limit (int, optional): Limit dataset size for debugging.
        epochs (int): Number of training epochs.
    """
    trainer = Trainer(limit=limit)
    best_score = trainer.run(epochs=epochs)
    return best_score
