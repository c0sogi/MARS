import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.ndimage
from library.config import Config
from library.utils import set_seed, compute_levenshtein, save_submission
from library.model import SGCRCN
from library.data_loader import get_loaders


class Trainer:
    def __init__(self, device, train_loader, val_loader, test_loader):
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Initialize Model
        self.model = SGCRCN().to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Class Weights for Imbalance
        self.class_weights = Config.get_class_weights(self.device)

        # Loss Functions
        # Classification: CrossEntropy (reduction='none' to apply mask manually)
        self.cls_criterion = nn.CrossEntropyLoss(
            weight=self.class_weights, reduction="none"
        )

        # Boundary: BCEWithLogits (reduction='none')
        self.bnd_criterion = nn.BCEWithLogitsLoss(reduction="none")

    def compute_tmse_loss(self, probs, mask):
        """
        Mean Squared Error for smoothing (Unbounded).
        probs: (B, T, C) - Softmax probabilities
        mask: (B, T)
        Removed clamping to strictly penalize jitter (Cite Lesson 00087).
        """
        # Calculate diff: P_t - P_{t-1}
        # Shape: (B, T-1, C)
        diff = probs[:, 1:, :] - probs[:, :-1, :]

        # Squared Error
        sq_diff = diff**2

        # Mask alignment (T-1)
        # Only consider transition valid if both t and t-1 are valid
        mask_t = mask[:, 1:] * mask[:, :-1]

        # Mean over valid transitions
        # Sum over features (C), then average over time and batch
        loss = (sq_diff.mean(dim=-1) * mask_t).sum() / (mask_t.sum() + 1e-8)

        return loss

    def compute_combined_loss(self, outputs, targets_cls, targets_bnd, mask):
        """
        Computes the multi-stage deep supervision loss.
        outputs: dict of logits from model
        targets_cls: (B, T)
        targets_bnd: (B, T)
        mask: (B, T)
        """
        total_loss = 0.0

        # Iterate over stages 1, 2, 3
        stages = [1, 2, 3]

        for s in stages:
            # Retrieve logits
            logits_cls = outputs[f"stage{s}_cls"]  # (B, T, C)
            logits_bnd = outputs[f"stage{s}_bnd"].squeeze(-1)  # (B, T)

            # 1. Classification Loss
            # Transpose for CE: (B, C, T)
            ce_loss_raw = self.cls_criterion(logits_cls.transpose(1, 2), targets_cls)
            ce_loss = (ce_loss_raw * mask).sum() / (mask.sum() + 1e-8)

            # 2. Boundary Loss
            bnd_loss_raw = self.bnd_criterion(logits_bnd, targets_bnd)
            bnd_loss = (bnd_loss_raw * mask).sum() / (mask.sum() + 1e-8)

            # 3. Smoothing Loss
            # Apply to softmax probabilities
            probs_cls = F.softmax(logits_cls, dim=2)
            # Use unbounded TMSE (Cite Lesson 00087)
            smooth_loss = self.compute_tmse_loss(probs_cls, mask)

            # Weighted Sum for this stage
            # BND_LOSS_WEIGHT is 0.0, enabling implicit boundary learning (Cite Lesson 00077)
            stage_loss = (
                Config.CLS_LOSS_WEIGHT * ce_loss
                + Config.BND_LOSS_WEIGHT * bnd_loss
                + Config.SMOOTH_LOSS_WEIGHT * smooth_loss
            )

            total_loss += stage_loss

        return total_loss

    def decode_predictions(self, logits_cls, lengths):
        """
        Decodes logits to gesture sequences.
        logits_cls: (B, T, C)
        lengths: (B,)
        Returns: list of list of ints
        """
        preds = []

        # Softmax & Argmax
        probs = F.softmax(logits_cls, dim=2)
        labels = torch.argmax(probs, dim=2).cpu().numpy()  # (B, T)

        for i in range(len(labels)):
            length = lengths[i]
            seq = labels[i, :length]

            # Median Filter to smooth noise
            k = Config.MEDIAN_FILTER_KERNEL
            if k % 2 == 0:
                k += 1
            if len(seq) > k:
                seq = scipy.ndimage.median_filter(seq, size=k, mode="nearest")

            # Collapse repeats and remove background (0)
            decoded_seq = []
            prev = -1
            for lbl in seq:
                if lbl != prev:
                    if lbl != 0:  # 0 is background
                        decoded_seq.append(int(lbl))
                    prev = lbl

            preds.append(decoded_seq)

        return preds

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            # Move to device
            features = batch["features"].to(self.device)
            target_cls = batch["target_cls"].to(self.device)
            target_bnd = batch["target_bnd"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(features, mask)

            # Compute Loss
            loss = self.compute_combined_loss(outputs, target_cls, target_bnd, mask)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["features"].to(self.device)
                target_cls = batch["target_cls"].to(self.device)
                target_bnd = batch["target_bnd"].to(self.device)
                mask = batch["mask"].to(self.device)
                lengths = batch["lengths"]

                # Forward
                outputs = self.model(features, mask)

                # Loss
                loss = self.compute_combined_loss(outputs, target_cls, target_bnd, mask)
                total_loss += loss.item()
                num_batches += 1

                # Decode Predictions (using Stage 3 output for final result)
                stage3_logits = outputs["stage3_cls"]
                batch_preds = self.decode_predictions(stage3_logits, lengths)
                all_preds.extend(batch_preds)

                # Decode Targets (Ground Truth)
                target_cls_np = target_cls.cpu().numpy()
                for i in range(len(target_cls_np)):
                    l = lengths[i]
                    seq = target_cls_np[i, :l]
                    decoded_target = []
                    prev = -1
                    for lbl in seq:
                        if lbl != prev:
                            if lbl != 0:
                                decoded_target.append(int(lbl))
                            prev = lbl
                    all_targets.append(decoded_target)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Compute Levenshtein Error Rate
        error_rate = compute_levenshtein(all_preds, all_targets)

        return avg_loss, error_rate

    def fit(self, num_epochs=Config.NUM_EPOCHS):
        print(f"Starting training on device: {self.device}")

        best_error_rate = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_error = self.validate()

            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss} - "
                f"Val Loss: {val_loss} - "
                f"Val Error Rate: {val_error}"
            )

            # Checkpoint & Early Stopping
            if val_error < best_error_rate:
                best_error_rate = val_error
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Val Error Rate: {best_error_rate}")

    def predict(self):
        """
        Generates predictions for the test set and saves them.
        """
        # Load best model
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            print("Loaded best model for inference.")
        else:
            print("Warning: Best model not found, using current weights.")

        self.model.eval()
        all_sample_ids = []
        all_preds = []

        with torch.no_grad():
            for batch in self.test_loader:
                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)
                lengths = batch["lengths"]
                sample_ids = batch["sample_ids"]

                outputs = self.model(features, mask)
                stage3_logits = outputs["stage3_cls"]

                batch_preds = self.decode_predictions(stage3_logits, lengths)

                all_sample_ids.extend(sample_ids)
                all_preds.extend(batch_preds)

        # Save submission
        save_submission(all_sample_ids, all_preds)
