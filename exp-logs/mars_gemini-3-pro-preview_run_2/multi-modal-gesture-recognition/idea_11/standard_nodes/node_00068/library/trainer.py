import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import copy

from library.config import Config
from library.utils import (
    set_seed,
    compute_normalized_levenshtein,
    decode_predictions,
    median_filter_1d,
    generate_submission_file,
)
from library.model import NMD_CRCN
from library.data_loader import get_dataloaders


class Trainer:
    def __init__(self):
        set_seed(Config.SEED)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Model
        self.model = NMD_CRCN().to(self.device)

        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-2
        )
        # Reduce LR on plateau to fine-tune convergence
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Loss Weights
        # Convert list to tensor for CrossEntropy
        self.class_weights = torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float32).to(
            self.device
        )

        # Loss Functions
        self.ce_criterion = nn.CrossEntropyLoss(
            weight=self.class_weights, reduction="none"
        )

        # Paths
        self.checkpoint_path = os.path.join(Config.CHECKPOINTS_DIR, "best_model.pth")
        self.submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    def compute_loss(self, outputs, targets, mask):
        """
        Computes the Deep Supervision loss:
        L = L_stage1(CE) + L_stage2(CE + TMSE) + L_stage3(CE + TMSE)
        """
        # Unpack outputs
        logits1 = outputs["stage1"]
        logits2 = outputs["stage2"]
        logits3 = outputs["stage3"]

        # --- Cross Entropy Loss (All Stages) ---
        # Reshape for CE: (Batch, Classes, Time) vs (Batch, Time)
        # Permute logits to (B, C, T)
        loss_ce1 = self.ce_criterion(logits1.permute(0, 2, 1), targets)
        loss_ce2 = self.ce_criterion(logits2.permute(0, 2, 1), targets)
        loss_ce3 = self.ce_criterion(logits3.permute(0, 2, 1), targets)

        # Apply Mask
        # mask is (B, T)
        valid_pixels = torch.sum(mask)

        loss_ce1 = torch.sum(loss_ce1 * mask) / valid_pixels
        loss_ce2 = torch.sum(loss_ce2 * mask) / valid_pixels
        loss_ce3 = torch.sum(loss_ce3 * mask) / valid_pixels

        # --- Truncated MSE (Temporal Smoothness) for Refinement Stages ---
        # Enforce P_t approx P_{t-1}
        # L_tmse = mean( clamp(P_t - P_{t-1})^2 )

        def temporal_smoothness_loss(logits, mask_seq):
            probs = F.softmax(logits, dim=2)  # (B, T, C)
            # Calculate diffs: P[:, 1:, :] - P[:, :-1, :]
            diffs = probs[:, 1:, :] - probs[:, :-1, :]

            # Clamp gradients/values (Truncated part)
            # Using a threshold of 0.15 as a reasonable bound for probability shifts per frame
            diffs = torch.clamp(diffs, min=-0.15, max=0.15)

            # Squared Error
            mse = diffs**2

            # Masking (mask needs to be aligned to diffs, i.e., T-1)
            # We use the mask of the current frame (1:)
            mask_slice = mask_seq[:, 1:].unsqueeze(-1)  # (B, T-1, 1)

            loss_tmse = torch.sum(mse * mask_slice) / torch.sum(
                mask_slice * Config.NUM_CLASSES
            )
            return loss_tmse

        loss_tmse2 = temporal_smoothness_loss(logits2, mask)
        loss_tmse3 = temporal_smoothness_loss(logits3, mask)

        # --- Total Loss ---
        # Weighting: CE is primary. TMSE is regularization (lambda=0.15)
        total_loss = (loss_ce1 + loss_ce2 + loss_ce3) + 0.15 * (loss_tmse2 + loss_tmse3)

        return total_loss

    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0

        for batch in loader:
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(features, mask)
            loss = self.compute_loss(outputs, labels, mask)

            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(loader)

    def validate(self, loader):
        self.model.eval()
        running_loss = 0.0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(self.device)
                labels = batch["labels"].to(self.device)
                mask = batch["mask"].to(self.device)
                lengths = batch["lengths"]

                outputs = self.model(features, mask)
                loss = self.compute_loss(outputs, labels, mask)
                running_loss += loss.item()

                # Inference using Stage 3 outputs
                logits = outputs["stage3"]
                probs = F.softmax(logits, dim=2)
                preds_batch = torch.argmax(probs, dim=2).cpu().numpy()
                targets_batch = labels.cpu().numpy()

                # Decode
                for i in range(len(lengths)):
                    length = lengths[i]
                    # Get valid sequence
                    raw_pred = preds_batch[i, :length]
                    raw_target = targets_batch[i, :length]

                    # Apply Median Filter to smooth predictions
                    filtered_pred = median_filter_1d(raw_pred, kernel_size=7)

                    # Decode to gesture list
                    decoded_pred = decode_predictions(filtered_pred, background_class=0)
                    decoded_target = decode_predictions(raw_target, background_class=0)

                    all_preds.append(decoded_pred)
                    all_targets.append(decoded_target)

        # Compute Metric
        score = compute_normalized_levenshtein(all_preds, all_targets)
        avg_loss = running_loss / len(loader)

        return avg_loss, score

    def fit(self):
        print("Initializing Data Loaders...")
        train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

        print(f"Starting training on {self.device}...")
        best_score = float("inf")
        patience_counter = 0

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_score = self.validate(val_loader)

            self.scheduler.step(val_loss)

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Levenshtein: {val_score:.6f}"
            )

            # Early Stopping based on Levenshtein Score (Lower is better)
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                # print("  -> Saved Best Model")
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training Complete. Best Validation Score: {best_score:.6f}")

    def predict(self):
        print("Loading Best Model for Inference...")
        if not os.path.exists(self.checkpoint_path):
            print(
                "Checkpoint not found! Running inference with untrained model (debug)."
            )
        else:
            self.model.load_state_dict(
                torch.load(self.checkpoint_path, map_location=self.device)
            )

        self.model.eval()

        _, _, test_loader = get_dataloaders(load_cached_data=True)

        predictions_dict = {}

        print("Generating Predictions...")
        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)
                lengths = batch["lengths"]
                sample_ids = batch["sample_ids"]

                outputs = self.model(features, mask)

                # Use Stage 3 for final prediction
                logits = outputs["stage3"]
                probs = F.softmax(logits, dim=2)
                preds_batch = torch.argmax(probs, dim=2).cpu().numpy()

                for i in range(len(lengths)):
                    length = lengths[i]
                    sid = sample_ids[i]

                    raw_pred = preds_batch[i, :length]

                    # Apply Median Filter
                    filtered_pred = median_filter_1d(raw_pred, kernel_size=7)

                    # Decode
                    decoded_gestures = decode_predictions(
                        filtered_pred, background_class=0
                    )

                    predictions_dict[sid] = decoded_gestures

        print(f"Saving submission to {self.submission_path}...")
        generate_submission_file(predictions_dict, self.submission_path)
        print("Submission generated successfully.")
