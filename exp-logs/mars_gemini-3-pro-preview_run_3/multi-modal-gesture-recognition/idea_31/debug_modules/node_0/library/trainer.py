import os
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import (
    set_seed,
    compute_normalized_levenshtein,
    compute_class_weights,
)
from library.model import HNGKN
from library.data_loader import get_dataloaders
from library.inference import (
    preprocess_sequence,
    sliding_window_inference,
    decode_predictions,
)


class TruncatedMSESmoothingLoss(nn.Module):
    """
    Computes Truncated MSE loss on log-probabilities of adjacent frames.
    L = mean( min( (log(p_t) - log(p_{t-1}))^2, threshold^2 ) )
    """

    def __init__(self, threshold=1.0):
        super(TruncatedMSESmoothingLoss, self).__init__()
        self.threshold_sq = threshold**2

    def forward(self, logits):
        # logits: (Batch, Time, Classes)
        # Convert to log probabilities
        log_probs = torch.log_softmax(logits, dim=2)

        # Calculate diff between adjacent frames along time dimension
        # diff: (Batch, Time-1, Classes)
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Squared Error
        mse = diff**2

        # Truncate gradients for large jumps
        truncated_mse = torch.clamp(mse, min=0, max=self.threshold_sq)

        return torch.mean(truncated_mse)


class Trainer:
    def __init__(self, config=Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure reproducibility
        set_seed(self.config.SEED)

        # Initialize Data Loaders
        # load_cached_data=True ensures we use pre-computed numpy files if available
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            self.config, load_cached_data=True
        )

        # Extract normalization stats from the training dataset for consistent inference
        # These are needed for the preprocess_sequence function
        self.stats = {
            "audio_mean": self.train_loader.dataset.audio_mean,
            "audio_std": self.train_loader.dataset.audio_std,
            "skel_pos_std": self.train_loader.dataset.skel_pos_std,
        }

        # Initialize Model
        self.model = HNGKN().to(self.device)

        # Compute Class Weights to handle imbalance
        class_weights = compute_class_weights(
            self.config.TRAIN_METADATA_PATH, load_cached_data=True
        ).to(self.device)

        # Define Loss Functions
        self.criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
        self.criterion_smooth = TruncatedMSESmoothingLoss(
            threshold=self.config.SMOOTHING_THRESHOLD
        )

        # Initialize Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        # Training State
        self.best_lev_dist = float("inf")
        self.patience_counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass (Deep Supervision: outputs from all 3 stages)
            logits1, logits2, logits3 = self.model(features)

            # Loss Calculation
            # Permute logits to (Batch, Classes, Time) for CrossEntropyLoss

            # Stage 1: Encoder (Only Cross Entropy)
            loss1 = self.criterion_cls(logits1.permute(0, 2, 1), labels)

            # Stage 2: Refinement 1 (CE + Smoothing)
            loss2_cls = self.criterion_cls(logits2.permute(0, 2, 1), labels)
            loss2_smooth = self.criterion_smooth(logits2)
            loss2 = loss2_cls + self.config.LOSS_SMOOTHING_WEIGHT * loss2_smooth

            # Stage 3: Refinement 2 (CE + Smoothing)
            loss3_cls = self.criterion_cls(logits3.permute(0, 2, 1), labels)
            loss3_smooth = self.criterion_smooth(logits3)
            loss3 = loss3_cls + self.config.LOSS_SMOOTHING_WEIGHT * loss3_smooth

            # Total Cascaded Loss
            loss = loss1 + loss2 + loss3

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0

    def evaluate(self):
        """
        Evaluates the model on the validation set.
        Returns:
            avg_val_loss: Frame-wise loss (for monitoring).
            lev_dist: Normalized Levenshtein Distance on full sequences (for model selection).
        """
        self.model.eval()

        # 1. Calculate Frame-wise Loss using the loader (windowed data)
        val_loss = 0
        num_batches = 0
        with torch.no_grad():
            for features, labels in self.val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                logits1, logits2, logits3 = self.model(features)

                loss1 = self.criterion_cls(logits1.permute(0, 2, 1), labels)

                loss2_cls = self.criterion_cls(logits2.permute(0, 2, 1), labels)
                loss2_smooth = self.criterion_smooth(logits2)
                loss2 = loss2_cls + self.config.LOSS_SMOOTHING_WEIGHT * loss2_smooth

                loss3_cls = self.criterion_cls(logits3.permute(0, 2, 1), labels)
                loss3_smooth = self.criterion_smooth(logits3)
                loss3 = loss3_cls + self.config.LOSS_SMOOTHING_WEIGHT * loss3_smooth

                loss = loss1 + loss2 + loss3
                val_loss += loss.item()
                num_batches += 1

        avg_val_loss = val_loss / num_batches if num_batches > 0 else 0

        # 2. Calculate Levenshtein Distance on Full Sequences
        # We access the raw data lists directly from the dataset to reconstruct full sequences
        val_dataset = self.val_loader.dataset

        preds = []
        gts = []

        for i in range(len(val_dataset.ids_list)):
            audio = val_dataset.audio_list[i]
            skel = val_dataset.skeleton_list[i]
            gt_labels_frame = val_dataset.labels_list[i]

            # Preprocess full sequence
            features = preprocess_sequence(audio, skel, self.stats)

            # Run Inference (Sliding Window with Overlap)
            probs = sliding_window_inference(self.model, features, self.device)

            # Decode Predictions (RLE + Filtering)
            pred_seq = decode_predictions(probs)
            preds.append(pred_seq)

            # Decode Ground Truth (RLE on frame labels)
            # GT labels: 0=Background, 1-20=Gestures
            gt_rle = [(k, len(list(g))) for k, g in itertools.groupby(gt_labels_frame)]
            gt_seq = [int(k) for k, d in gt_rle if k != self.config.BACKGROUND_LABEL]
            gts.append(gt_seq)

        lev_dist = compute_normalized_levenshtein(preds, gts)

        return avg_val_loss, lev_dist

    def train(self):
        print(f"Starting training on {self.device}...")
        print(
            f"Train batches: {len(self.train_loader)}, Val batches: {len(self.val_loader)}"
        )

        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_lev = self.evaluate()

            print(
                f"Epoch {epoch}/{self.config.NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Levenshtein: {val_lev}"
            )

            # Checkpointing based on Levenshtein Distance (Lower is better)
            if val_lev < self.best_lev_dist:
                self.best_lev_dist = val_lev
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
                print(f"  New best model saved! Score: {val_lev}")
            else:
                self.patience_counter += 1
                print(
                    f"  No improvement. Patience: {self.patience_counter}/{self.config.PATIENCE}"
                )

            # Early Stopping
            if self.patience_counter >= self.config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Levenshtein: {self.best_lev_dist}")

    def generate_submission(self):
        """
        Generates predictions for the test set using the best saved model.
        Saves the result to submission.csv.
        """
        print("Loading best model for submission generation...")
        if os.path.exists(self.config.MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(self.config.MODEL_SAVE_PATH, map_location=self.device)
            )
        else:
            print("Warning: Best model not found. Using current model weights.")

        self.model.eval()
        submission_lines = []

        # Access raw test data
        test_dataset = self.test_loader.dataset

        print("Generating predictions on Test set...")
        for i in range(len(test_dataset.ids_list)):
            sample_id = test_dataset.ids_list[i]
            audio = test_dataset.audio_list[i]
            skel = test_dataset.skeleton_list[i]

            # Preprocess
            features = preprocess_sequence(audio, skel, self.stats)

            # Inference
            probs = sliding_window_inference(self.model, features, self.device)

            # Decode
            pred_seq = decode_predictions(probs)

            # Format: SessionID,Label1,Label2,...
            if len(pred_seq) > 0:
                pred_str = ",".join(map(str, pred_seq))
                line = f"{sample_id},{pred_str}"
            else:
                line = f"{sample_id}"

            submission_lines.append(line)

        os.makedirs(self.config.SUBMISSION_DIR, exist_ok=True)
        with open(self.config.SUBMISSION_PATH, "w") as f:
            for line in submission_lines:
                f.write(line + "\n")

        print(f"Submission saved to {self.config.SUBMISSION_PATH}")
