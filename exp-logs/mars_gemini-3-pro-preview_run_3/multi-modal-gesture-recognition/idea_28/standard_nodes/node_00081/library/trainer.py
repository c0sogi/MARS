import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from library.config import Config
from library.dataset import get_dataloaders
from library.model import RGHC_MN
from library.utils import load_dataset, levenshtein_score
from library.inference import predict_sequence, decode_predictions


class TruncatedSmoothingLoss(nn.Module):
    """
    Computes Truncated MSE loss on log-probabilities of adjacent frames
    to enforce temporal smoothness.
    """

    def __init__(self, threshold=1.0):
        super(TruncatedSmoothingLoss, self).__init__()
        self.threshold = threshold
        self.mse = nn.MSELoss()

    def forward(self, probabilities):
        # probabilities: (Batch, Time, Classes)
        # Convert to log space for numerical stability and standard smoothing
        log_probs = torch.log(probabilities + 1e-8)

        # Calculate difference between t and t-1
        # diff: (Batch, Time-1, Classes)
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Truncate gradients/errors
        diff = torch.clamp(diff, -self.threshold, self.threshold)

        # Compute MSE on the clamped differences
        # We assume target diff is 0 (smoothness)
        zeros = torch.zeros_like(diff)
        return self.mse(diff, zeros)


class Trainer:
    def __init__(self, load_cached_data=True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Trainer initialized on device: {self.device}")

        # 1. Data Setup
        # Get windowed loaders for training loop
        self.train_loader, self.val_loader, _ = get_dataloaders()

        # Load raw validation data for Levenshtein calculation (Full Sequence Inference)
        # We need raw sequences to perform sliding window inference and aggregation
        print("Loading raw validation data for metric evaluation...")
        raw_val_data = load_dataset(
            Config.VAL_METADATA_PATH,
            Config.VAL_CACHE_PATH,
            load_cached_data=load_cached_data,
        )
        self.val_skeletons = raw_val_data["skeletons"]
        self.val_audio = raw_val_data["audio"]
        self.val_gt_sequences = raw_val_data["gt_sequences"]
        self.val_sample_ids = raw_val_data["sample_ids"]

        # 2. Model Setup
        self.model = RGHC_MN().to(self.device)

        # 3. Loss Setup
        # Class weights: 0.2 for background (0), 1.0 for others
        weights = torch.ones(Config.NUM_CLASSES).to(self.device)
        weights[0] = Config.WEIGHT_BACKGROUND

        # Model outputs probabilities, so we use NLLLoss on log(probs)
        self.criterion_cls = nn.NLLLoss(weight=weights)
        self.criterion_smooth = TruncatedSmoothingLoss(
            threshold=Config.SMOOTHING_THRESHOLD
        )

        # 4. Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        for batch_idx, (features, labels, _, _) in enumerate(self.train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)  # (Batch, Time)

            self.optimizer.zero_grad()

            # Forward pass returns dict of outputs for deep supervision
            outputs = self.model(features)

            p1 = outputs["stage1"]
            p2 = outputs["stage2"]
            p3 = outputs["stage3"]

            # Flatten for NLLLoss: (Batch*Time, Classes) vs (Batch*Time)
            # NLLLoss expects (N, C) and target (N) or (N, C, d1...) and target (N, d1...)
            # We'll use (Batch, Classes, Time) convention for NLLLoss or reshape

            # Log probabilities
            log_p1 = torch.log(p1 + 1e-8).transpose(1, 2)  # (B, C, T)
            log_p2 = torch.log(p2 + 1e-8).transpose(1, 2)
            log_p3 = torch.log(p3 + 1e-8).transpose(1, 2)

            # Classification Losses
            loss_1 = self.criterion_cls(log_p1, labels)
            loss_2 = self.criterion_cls(log_p2, labels)
            loss_3 = self.criterion_cls(log_p3, labels)

            # Smoothing Losses (applied to p2 and p3)
            smooth_2 = self.criterion_smooth(p2)
            smooth_3 = self.criterion_smooth(p3)

            # Total Loss
            loss = (loss_1 + loss_2 + loss_3) + Config.WEIGHT_SMOOTHING * (
                smooth_2 + smooth_3
            )

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """
        Performs validation in two steps:
        1. Calculate Loss on windowed validation set.
        2. Calculate Levenshtein Score on full validation sequences.
        """
        self.model.eval()

        # 1. Validation Loss
        val_loss = 0.0
        with torch.no_grad():
            for features, labels, _, _ in self.val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                p3 = outputs["stage3"]
                log_p3 = torch.log(p3 + 1e-8).transpose(1, 2)

                loss = self.criterion_cls(log_p3, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)

        # 2. Levenshtein Score (Model Selection Metric)
        # We iterate over the raw sequences, predict, decode, and compare.
        all_preds = []
        all_targets = self.val_gt_sequences

        # To speed up validation, we can skip if needed, but for best model selection
        # we should run it.
        with torch.no_grad():
            for i, (skel, aud) in enumerate(zip(self.val_skeletons, self.val_audio)):
                # Predict using sliding window inference
                probs = predict_sequence(self.model, skel, aud, self.device)

                # Decode
                pred_seq = decode_predictions(probs)
                all_preds.append(pred_seq)

        lev_score = levenshtein_score(all_preds, all_targets)

        return avg_val_loss, lev_score

    def fit(self, epochs=Config.NUM_EPOCHS):
        # Set seeds for reproducibility
        torch.manual_seed(Config.RANDOM_SEED)
        np.random.seed(Config.RANDOM_SEED)
        random.seed(Config.RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.RANDOM_SEED)

        print(f"Starting training for {epochs} epochs...")

        best_lev_score = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_lev = self.validate()

            print(
                f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Levenshtein: {val_lev:.6f}"
            )

            # Checkpoint
            if val_lev < best_lev_score:
                best_lev_score = val_lev
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"  -> New best model saved! Score: {best_lev_score:.6f}")
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break

        print(f"Training complete. Best Levenshtein Score: {best_lev_score:.6f}")
