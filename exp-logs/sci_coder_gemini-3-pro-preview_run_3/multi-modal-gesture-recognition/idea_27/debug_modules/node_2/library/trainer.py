import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.model import GHCKRN
from library.loss import CascadedLoss
from library.data_loader import FeatureExtractor, GestureDataset
from library.utils import (
    compute_levenshtein_score,
    decode_predictions_to_labels,
    levenshtein_distance,
)


class Trainer:
    """
    Manages training, validation, and inference for the GHC-KRN model.
    """

    def __init__(self, model, train_loader, val_loader, test_loader=None):
        self.model = model.to(Config.DEVICE)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.criterion = CascadedLoss().to(Config.DEVICE)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.best_score = float("inf")
        self.patience_counter = 0

    def _prepare_sequence_features(self, positions, audio):
        """
        Prepares input features for a full sequence (Validation/Inference).
        Args:
            positions (np.ndarray): (T, 20, 3) in mm.
            audio (np.ndarray): (T, 13).
        Returns:
            torch.Tensor: (T, InputDim)
        """
        # 1. Normalize positions (mm -> m)
        pos_norm = positions / 1000.0

        # 2. Compute Kinematics
        vel, acc = FeatureExtractor.compute_kinematics(pos_norm)

        # 3. Flatten
        T = positions.shape[0]
        pos_flat = pos_norm.reshape(T, -1)
        vel_flat = vel.reshape(T, -1)
        acc_flat = acc.reshape(T, -1)

        # 4. Concatenate
        features = np.concatenate([pos_flat, vel_flat, acc_flat, audio], axis=1)
        return torch.FloatTensor(features)

    def _sliding_window_inference(self, features):
        """
        Performs sliding window inference on a full sequence.
        Args:
            features (torch.Tensor): (T, InputDim)
        Returns:
            np.ndarray: (T, NumClasses) probabilities.
        """
        self.model.eval()
        T = features.shape[0]
        window_size = Config.WINDOW_SIZE
        stride = Config.STRIDE_TEST

        # Prepare buffer for probabilities and counts
        probs_sum = torch.zeros((T, Config.NUM_CLASSES), device=Config.DEVICE)
        counts = torch.zeros((T, 1), device=Config.DEVICE)

        # Generate windows
        windows = []
        indices = []

        # Handle short sequences
        if T < window_size:
            # Pad
            pad_len = window_size - T
            feat_padded = torch.nn.functional.pad(features, (0, 0, 0, pad_len))
            windows.append(feat_padded)
            indices.append((0, T))
        else:
            for start in range(0, T - window_size + 1, stride):
                end = start + window_size
                windows.append(features[start:end])
                indices.append((start, end))

            # Handle last window if not covered perfectly
            if (T - window_size) % stride != 0:
                start = T - window_size
                end = T
                windows.append(features[start:end])
                indices.append((start, end))

        if not windows:
            return np.zeros((T, Config.NUM_CLASSES))

        # Batch processing
        batch_size = Config.BATCH_SIZE
        num_windows = len(windows)

        with torch.no_grad():
            for i in range(0, num_windows, batch_size):
                batch_wins = windows[i : i + batch_size]
                batch_indices = indices[i : i + batch_size]

                # Stack: (B, Window, Dim)
                batch_input = torch.stack(batch_wins).to(Config.DEVICE)

                # Forward
                outputs = self.model(batch_input)

                # Take Stage 3 output (index 2)
                # Output shape: (B, Classes, Window)
                logits = outputs[-1]

                # Softmax -> (B, Classes, Window)
                probs = torch.softmax(logits, dim=1)

                # Permute to (B, Window, Classes) for accumulation
                probs = probs.permute(0, 2, 1)

                # Accumulate
                for j, (start, end) in enumerate(batch_indices):
                    # If padded (short sequence)
                    valid_len = end - start
                    p = probs[j, :valid_len, :]

                    probs_sum[start:end] += p
                    counts[start:end] += 1.0

        # Average
        counts[counts == 0] = 1.0  # Avoid div by zero
        avg_probs = probs_sum / counts

        return avg_probs.cpu().numpy()

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for features, targets in self.train_loader:
            features = features.to(Config.DEVICE)
            targets = targets.to(Config.DEVICE)

            self.optimizer.zero_grad()

            # Forward pass (returns list of outputs from all stages)
            outputs = self.model(features)

            # Compute cascaded loss
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self):
        """
        Validates on the full validation set using Levenshtein distance.
        Reconstructs sequences from the dataset arrays.
        """
        dataset = self.val_loader.dataset
        total_dist = 0
        total_len = 0

        # Iterate over all sequences in the validation set
        num_sequences = len(dataset.seq_lens)

        for i in range(num_sequences):
            # Extract full sequence data
            start_idx = dataset.seq_starts[i]
            length = dataset.seq_lens[i]
            end_idx = start_idx + length

            raw_pos = dataset.positions[start_idx:end_idx]
            raw_audio = dataset.audio[start_idx:end_idx]
            gt_frame_labels = dataset.labels[start_idx:end_idx]

            # Prepare features
            features = self._prepare_sequence_features(raw_pos, raw_audio)

            # Inference
            probs = self._sliding_window_inference(features)

            # Decode predictions (Frame-wise probs -> List of Gesture IDs)
            frame_preds = np.argmax(probs, axis=1)
            pred_labels = decode_predictions_to_labels(frame_preds)

            # Decode Ground Truth (Frame-wise labels -> List of Gesture IDs)
            # Note: dataset.labels contains 0 for background, 1-20 for gestures
            gt_labels = decode_predictions_to_labels(gt_frame_labels)

            # Metric
            dist = levenshtein_distance(pred_labels, gt_labels)
            total_dist += dist
            total_len += len(gt_labels)

        score = total_dist / total_len if total_len > 0 else float("inf")
        return score

    def fit(self, num_epochs=Config.NUM_EPOCHS):
        print(f"Starting training for {num_epochs} epochs...")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_score = self.validate()

            print(
                f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.6f} - Val Score (Levenshtein): {val_score:.6f}"
            )

            # Checkpointing & Early Stopping
            if val_score < self.best_score:
                self.best_score = val_score
                self.patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"  -> New best model saved! Score: {val_score:.6f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Validation Score: {self.best_score:.6f}")

    def generate_submission(self):
        """
        Generates predictions for the test set and saves to CSV.
        """
        if self.test_loader is None:
            print("No test loader provided.")
            return

        print("Generating submission...")
        # Load best model
        try:
            self.model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
            )
            print("Loaded best model for inference.")
        except FileNotFoundError:
            print("Warning: Best model not found, using current weights.")

        dataset = self.test_loader.dataset
        num_sequences = len(dataset.seq_lens)
        results = []

        for i in range(num_sequences):
            sample_id = dataset.ids[i]

            # Extract data
            start_idx = dataset.seq_starts[i]
            length = dataset.seq_lens[i]
            end_idx = start_idx + length

            raw_pos = dataset.positions[start_idx:end_idx]
            raw_audio = dataset.audio[start_idx:end_idx]

            # Prepare & Infer
            features = self._prepare_sequence_features(raw_pos, raw_audio)
            probs = self._sliding_window_inference(features)

            # Decode
            frame_preds = np.argmax(probs, axis=1)
            pred_labels = decode_predictions_to_labels(frame_preds)

            # Format string: "Label1,Label2,Label3"
            label_str = ",".join(map(str, pred_labels))
            results.append(f"{sample_id},{label_str}")

        # Save to file
        with open(Config.SUBMISSION_PATH, "w") as f:
            for line in results:
                f.write(line + "\n")

        print(f"Submission saved to {Config.SUBMISSION_PATH}")
