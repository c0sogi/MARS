import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import (
    NUM_CLASSES,
    BACKGROUND_CLASS_ID,
    BACKGROUND_WEIGHT,
    SMOOTHING_LOSS_WEIGHT,
    TRUNCATION_THRESHOLD,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    WINDOW_SIZE,
    STRIDE,
    VAL_METADATA_PATH,
    CACHE_DIR,
    SUBMISSION_FILE,
    DEBUG_DATA_LIMIT,
)
from library.utils import (
    set_seed,
    decode_predictions,
    compute_challenge_metric,
    filter_short_segments,
)
from library.data_loader import get_dataloaders, process_dataset, GestureDataset
from library.model import RMDKN


class SmoothingLoss(nn.Module):
    """
    Truncated MSE Loss applied to log-probabilities of adjacent frames
    to enforce temporal smoothness.
    """

    def __init__(self, threshold=TRUNCATION_THRESHOLD):
        super(SmoothingLoss, self).__init__()
        self.threshold = threshold

    def forward(self, logits):
        """
        Args:
            logits: (Batch, Time, NumClasses)
        Returns:
            Scalar loss
        """
        # Convert to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Calculate difference between adjacent frames: t and t-1
        # Shape: (Batch, Time-1, NumClasses)
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Squared difference
        sq_diff = diff**2

        # Truncate (clamp) the squared difference
        # We clamp the loss value, effectively Huber-like but simple truncation
        truncated_sq_diff = torch.clamp(sq_diff, max=self.threshold**2)

        return truncated_sq_diff.mean()


class Trainer:
    def __init__(self, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        set_seed()

        # Data Loaders
        self.train_loader, self.val_loader, self.test_loader, self.test_samples = (
            get_dataloaders()
        )

        # Load raw validation samples for metric calculation (full sequence reconstruction)
        self.val_samples = process_dataset(
            VAL_METADATA_PATH, "dataset_val", load_cached_data=True
        )

        # Model
        self.model = RMDKN().to(self.device)

        # Optimizer (Adam as requested, no AdamW)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Loss Functions
        # Class weights: 1.0 for gestures, 0.2 for background
        weights = torch.ones(NUM_CLASSES).to(self.device)
        weights[BACKGROUND_CLASS_ID] = BACKGROUND_WEIGHT
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)

        self.smooth_loss = SmoothingLoss(threshold=TRUNCATION_THRESHOLD)

    def compute_loss(self, logits_1, logits_2, logits_3, targets):
        """
        Computes the cascaded loss:
        L = CE(S1) + CE(S2) + Smooth(S2) + CE(S3) + Smooth(S3)
        """
        # Flatten for CrossEntropy: (Batch * Time, Classes) vs (Batch * Time)
        B, T, C = logits_1.shape
        flat_targets = targets.view(-1)

        loss_ce_1 = self.ce_loss(logits_1.reshape(-1, C), flat_targets)
        loss_ce_2 = self.ce_loss(logits_2.reshape(-1, C), flat_targets)
        loss_ce_3 = self.ce_loss(logits_3.reshape(-1, C), flat_targets)

        loss_smooth_2 = self.smooth_loss(logits_2)
        loss_smooth_3 = self.smooth_loss(logits_3)

        total_loss = (
            loss_ce_1
            + loss_ce_2
            + SMOOTHING_LOSS_WEIGHT * loss_smooth_2
            + loss_ce_3
            + SMOOTHING_LOSS_WEIGHT * loss_smooth_3
        )

        return total_loss

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        # Use simple iteration to avoid verbose progress bars
        for batch in self.train_loader:
            features = batch["features"].to(self.device)
            targets = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            l1, l2, l3 = self.model(features)

            loss = self.compute_loss(l1, l2, l3, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate_loss(self):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["features"].to(self.device)
                targets = batch["labels"].to(self.device)

                l1, l2, l3 = self.model(features)
                loss = self.compute_loss(l1, l2, l3, targets)
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def run_inference_on_sample(self, sample, stride=STRIDE // 2):
        """
        Runs sliding window inference on a single sample.
        Aggregates probabilities from overlapping windows.
        """
        self.model.eval()

        # Extract full data
        full_skel = sample["skeleton"]
        full_audio = sample["audio"]
        seq_len = full_skel.shape[0]

        # Prepare accumulators
        prob_sum = np.zeros((seq_len, NUM_CLASSES), dtype=np.float32)
        count_map = np.zeros((seq_len, 1), dtype=np.float32)

        # Create windows
        windows = []
        indices = []

        if seq_len < WINDOW_SIZE:
            # Pad
            pad_len = WINDOW_SIZE - seq_len
            skel_pad = np.pad(
                full_skel, ((0, pad_len), (0, 0), (0, 0)), mode="constant"
            )
            audio_pad = np.pad(full_audio, ((0, pad_len), (0, 0)), mode="constant")
            windows.append(self._prepare_window(skel_pad, audio_pad))
            indices.append((0, seq_len))  # Valid range
        else:
            for start in range(0, seq_len - WINDOW_SIZE + 1, stride):
                end = start + WINDOW_SIZE
                skel_win = full_skel[start:end]
                audio_win = full_audio[start:end]
                windows.append(self._prepare_window(skel_win, audio_win))
                indices.append((start, end))

            # Handle last window if needed
            if (seq_len - WINDOW_SIZE) % stride != 0:
                start = seq_len - WINDOW_SIZE
                end = seq_len
                skel_win = full_skel[start:end]
                audio_win = full_audio[start:end]
                windows.append(self._prepare_window(skel_win, audio_win))
                indices.append((start, end))

        if not windows:
            return np.zeros((seq_len, NUM_CLASSES))

        # Batch inference
        batch_tensor = torch.stack(windows).to(self.device)

        with torch.no_grad():
            # We use Stage 3 output for final prediction
            _, _, logits_3 = self.model(batch_tensor)
            probs_3 = F.softmax(logits_3, dim=-1).cpu().numpy()

        # Aggregate
        for i, (start, end) in enumerate(indices):
            valid_len = end - start
            # If original was padded, we only care about valid part
            # But here windows are fixed size WINDOW_SIZE.
            # If seq_len < WINDOW_SIZE, valid_len is seq_len, but window is WINDOW_SIZE

            p = probs_3[i]  # (WINDOW_SIZE, C)

            if seq_len < WINDOW_SIZE:
                # Crop padding
                p = p[:seq_len]
                prob_sum[0:seq_len] += p
                count_map[0:seq_len] += 1
            else:
                prob_sum[start:end] += p
                count_map[start:end] += 1

        # Avoid division by zero
        count_map[count_map == 0] = 1
        avg_probs = prob_sum / count_map

        return avg_probs

    def _prepare_window(self, skel, audio):
        """Helper to prepare features for model input (no augmentation)"""
        # Calculate derivatives
        vel = np.zeros_like(skel)
        vel[1:] = skel[1:] - skel[:-1]

        acc = np.zeros_like(vel)
        acc[1:] = vel[1:] - vel[:-1]

        # Flatten
        T = skel.shape[0]
        pos_flat = skel.reshape(T, -1)
        vel_flat = vel.reshape(T, -1)
        acc_flat = acc.reshape(T, -1)

        feat = np.concatenate([pos_flat, vel_flat, acc_flat, audio], axis=1)
        return torch.from_numpy(feat).float()

    def validate_metric(self):
        """
        Computes Levenshtein Distance on full validation sequences.
        """
        predictions = []
        ground_truths = []

        for sample in self.val_samples:
            # 1. Run inference
            avg_probs = self.run_inference_on_sample(sample)

            # 2. Decode
            pred_seq = decode_predictions(avg_probs)
            predictions.append(pred_seq)

            # 3. Get Ground Truth
            # sample['labels'] is frame-wise array
            # We need to extract the sequence of gesture IDs
            gt_labels = sample["labels"]
            # Decode GT similarly to get sequence (ignoring background)
            # Or parse from original metadata?
            # Using decode_predictions on one-hot GT is equivalent to parsing RLE of labels
            # But we must ensure duration filter is consistent or just take raw changes

            # Simple RLE on GT labels
            gt_seq = []
            if len(gt_labels) > 0:
                curr = gt_labels[0]
                if curr != BACKGROUND_CLASS_ID:
                    gt_seq.append(curr)

                for x in gt_labels[1:]:
                    if x != curr:
                        curr = x
                        if curr != BACKGROUND_CLASS_ID:
                            gt_seq.append(curr)

            ground_truths.append(gt_seq)

        score = compute_challenge_metric(predictions, ground_truths)
        return score

    def run(self):
        print(f"Starting training on {self.device}...")
        best_score = float("inf")
        patience = 10
        patience_counter = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_loss()
            val_score = self.validate_metric()

            print(
                f"Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Levenshtein: {val_score:.6f}"
            )

            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(
                    self.model.state_dict(), os.path.join(CACHE_DIR, "best_model.pth")
                )
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training finished. Best Val Score: {best_score:.6f}")

    def generate_submission(self):
        print("Generating submission...")
        # Load best model
        model_path = os.path.join(CACHE_DIR, "best_model.pth")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("Loaded best model.")
        else:
            print("Warning: Best model not found, using current weights.")

        results = []

        for sample in self.test_samples:
            sid = sample["sample_id"]
            avg_probs = self.run_inference_on_sample(sample)
            pred_seq = decode_predictions(avg_probs)

            # Format: SessionID,label1,label2,...
            # If empty, just SessionID
            row_str = str(sid)
            if pred_seq:
                row_str += "," + ",".join(map(str, pred_seq))

            results.append(row_str)

        # Save to file
        with open(SUBMISSION_FILE, "w") as f:
            for line in results:
                f.write(line + "\n")

        print(f"Submission saved to {SUBMISSION_FILE}")
