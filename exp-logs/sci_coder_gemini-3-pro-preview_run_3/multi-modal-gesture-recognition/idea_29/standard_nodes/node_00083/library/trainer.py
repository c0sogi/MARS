import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from library.config import Config
from library.utils import (
    levenshtein_distance,
    rle_encode,
    filter_short_segments,
    TruncatedMSELoss,
)
from library.data_loader import get_dataloaders
from library.model import NGKRN


class Trainer:
    """
    Trainer for the Normalized Gated-Kinematic Refinement Network (NG-KRN).
    Handles training, validation (sliding window inference), and submission generation.
    """

    def __init__(self, debug=False):
        self.debug = debug
        self.device = torch.device(Config.DEVICE)

        # Set seeds for reproducibility
        self._set_seeds(Config.SEED)

        # Initialize Model
        self.model = NGKRN().to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Functions
        # Model outputs Softmax probabilities, so we use NLLLoss with log(probs)
        class_weights = torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float32).to(
            self.device
        )
        self.criterion_cls = nn.NLLLoss(weight=class_weights)
        self.criterion_smooth = TruncatedMSELoss(
            threshold=Config.SMOOTHING_THRESHOLD
        ).to(self.device)

        # Data Loaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            debug=self.debug,
        )

    def _set_seeds(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            features = batch["features"].to(self.device)  # (B, W, D)
            labels = batch["labels"].to(self.device)  # (B, W)

            self.optimizer.zero_grad()

            # Forward pass (Deep Supervision: p1, p2, p3)
            p1, p2, p3 = self.model(features)

            # Compute Loss for each stage
            # Model outputs probs, convert to log-probs for NLLLoss and Smoothing
            loss_stage = 0
            for p in [p1, p2, p3]:
                # Clamp for numerical stability
                p_clamped = torch.clamp(p, min=1e-8)
                log_p = torch.log(p_clamped)

                # Reshape for NLLLoss: (B, C, T)
                log_p_permuted = log_p.permute(0, 2, 1)

                cls_loss = self.criterion_cls(log_p_permuted, labels)
                smooth_loss = self.criterion_smooth(log_p)

                loss_stage += cls_loss + Config.SMOOTHING_LAMBDA * smooth_loss

            loss_stage.backward()
            self.optimizer.step()

            total_loss += loss_stage.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _sliding_window_inference(self, features):
        """
        Performs sliding window inference on a single sequence.
        Aggregates probabilities from overlapping windows.

        Args:
            features: Tensor (T, InputDim)

        Returns:
            avg_probs: Numpy array (T, NumClasses) - Final stage probabilities
        """
        self.model.eval()
        T = features.shape[0]
        W = Config.WINDOW_SIZE
        S = Config.STRIDE_TEST

        # Prepare accumulators
        # We only care about the final stage (p3) for inference
        probs_sum = torch.zeros((T, Config.NUM_CLASSES), device=self.device)
        counts = torch.zeros((T, 1), device=self.device)

        # Handle short sequences by padding
        if T < W:
            pad_len = W - T
            # Pad features: (T, D) -> (W, D)
            # Pad last frame
            last_frame = features[-1:]
            padding = last_frame.repeat(pad_len, 1)
            feat_padded = torch.cat([features, padding], dim=0)

            # Add batch dim
            input_tensor = feat_padded.unsqueeze(0).to(self.device)

            with torch.no_grad():
                _, _, p3 = self.model(input_tensor)  # (1, W, C)

            # Crop back
            probs = p3[0, :T, :]
            return probs.cpu().numpy()

        # Sliding Window Loop
        # Generate start indices
        starts = list(range(0, T - W + 1, S))
        if (T - W) % S != 0:
            starts.append(T - W)

        for start in starts:
            end = start + W
            window = features[start:end]  # (W, D)
            input_tensor = window.unsqueeze(0).to(self.device)

            with torch.no_grad():
                _, _, p3 = self.model(input_tensor)  # (1, W, C)

            probs_window = p3[0]  # (W, C)

            probs_sum[start:end] += probs_window
            counts[start:end] += 1.0

        # Average
        avg_probs = probs_sum / torch.clamp(counts, min=1.0)
        return avg_probs.cpu().numpy()

    def validate(self):
        self.model.eval()
        total_dist = 0
        total_gestures = 0

        # Validation loader returns 1 sequence per batch
        for batch in self.val_loader:
            # Unpack
            features = batch["features"][0]  # (T, D)
            labels = batch["labels"][0]  # (T,)

            # Inference
            probs = self._sliding_window_inference(features)  # (T, C)

            # Decode
            pred_labels = np.argmax(probs, axis=1)

            # Post-processing
            pred_labels_filtered = filter_short_segments(
                pred_labels, min_duration=Config.MIN_DURATION_FRAMES
            )
            pred_sequence = rle_encode(pred_labels_filtered)

            # Ground Truth Sequence
            gt_sequence = rle_encode(labels.numpy())

            # Metric
            dist = levenshtein_distance(pred_sequence, gt_sequence)
            total_dist += dist
            total_gestures += len(gt_sequence)

        # Avoid division by zero
        if total_gestures == 0:
            return 0.0

        return total_dist / total_gestures

    def fit(self):
        print(f"Starting training on device: {self.device}")
        best_score = float("inf")
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = self.train_epoch(epoch)
            val_score = self.validate()

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Train Loss: {train_loss:.6f} - Val Levenshtein: {val_score:.6f}"
            )

            # Checkpoint & Early Stopping
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(
                    self.model.state_dict(),
                    os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
                )
            else:
                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Score: {best_score:.6f}")

    def predict(self):
        print("Generating submission...")
        # Load best model
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            print("Loaded best model.")
        else:
            print("Warning: Best model not found, using current weights.")

        self.model.eval()
        results = []

        for batch in self.test_loader:
            sample_id = batch["sample_id"][0]
            features = batch["features"][0]  # (T, D)

            # Inference
            probs = self._sliding_window_inference(features)

            # Decode
            pred_labels = np.argmax(probs, axis=1)
            pred_labels_filtered = filter_short_segments(
                pred_labels, min_duration=Config.MIN_DURATION_FRAMES
            )
            pred_sequence = rle_encode(pred_labels_filtered)

            # Format: SessionID,Label1,Label2...
            # Labels in submission are 1-20. rle_encode removes 0.
            str_preds = [str(x) for x in pred_sequence]
            line = f"{sample_id}," + ",".join(str_preds)
            results.append(line)

        # Save to file
        with open(Config.SUBMISSION_FILE, "w") as f:
            for line in results:
                f.write(line + "\n")

        print(f"Submission saved to {Config.SUBMISSION_FILE}")
