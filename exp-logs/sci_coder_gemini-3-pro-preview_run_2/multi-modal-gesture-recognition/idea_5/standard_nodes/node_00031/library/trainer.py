import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import (
    TRAIN_PARAMS,
    MODEL_PARAMS,
    WORKING_DIR,
    SUBMISSION_DIR,
    SEED,
)
from library.model import CRCN
from library.utils import compute_levenshtein_score, decode_predictions

# Ensure reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


class Trainer:
    def __init__(self, device):
        self.device = device
        self.model = CRCN().to(device)

        # Define Class Weights
        weights = torch.tensor(TRAIN_PARAMS["class_weights"], dtype=torch.float32).to(
            device
        )
        self.criterion = nn.CrossEntropyLoss(weight=weights, reduction="none")

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=TRAIN_PARAMS["learning_rate"],
            weight_decay=TRAIN_PARAMS["weight_decay"],
        )

        # Scheduler
        # Use Cosine Annealing for smoother convergence
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=TRAIN_PARAMS["num_epochs"], eta_min=1e-6
        )

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.model_path = os.path.join(WORKING_DIR, "best_model.pth")

    def compute_masked_loss(self, logits, targets, lengths):
        """
        Computes CrossEntropyLoss masking out the padding.
        logits: (B, T, C)
        targets: (B, T)
        lengths: (B,)
        """
        batch_size, max_len, num_classes = logits.shape

        # Flatten logits and targets
        logits_flat = logits.reshape(-1, num_classes)  # (B*T, C)
        targets_flat = targets.reshape(-1)  # (B*T)

        # Create mask based on lengths
        # range (0 to max_len-1) < length
        seq_range = (
            torch.arange(max_len, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, max_len)
        )
        mask = seq_range < lengths.unsqueeze(1)
        mask_flat = mask.view(-1)

        # Compute element-wise loss
        loss_raw = self.criterion(logits_flat, targets_flat)

        # Apply mask
        masked_loss = loss_raw * mask_flat.float()

        # Average over valid tokens only
        total_valid_tokens = mask_flat.sum()
        if total_valid_tokens > 0:
            return masked_loss.sum() / total_valid_tokens
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def compute_tmse_loss(self, logits, lengths):
        """
        Computes Truncated MSE loss on probabilities to encourage smoothness.
        Cite Lesson 00013: Smoothing losses can help but require care.
        """
        probs = torch.softmax(logits, dim=2)
        batch_size, max_len, num_classes = probs.shape

        if max_len <= 1:
            return torch.tensor(0.0, device=self.device)

        # Diff: P_t - P_{t-1}
        # shape: (B, T-1, C)
        diff = probs[:, 1:, :] - probs[:, :-1, :]

        # MSE summed over classes
        loss_sq = diff.pow(2).sum(dim=2)  # (B, T-1)

        # Truncated MSE: Clamp the loss to allow sharp transitions (Cite Lesson 00029)
        threshold = TRAIN_PARAMS.get("tmse_threshold", float("inf"))
        loss_sq = torch.clamp(loss_sq, max=threshold)

        # Mask for T-1
        seq_range = (
            torch.arange(max_len - 1, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, max_len - 1)
        )
        mask = seq_range < (lengths.unsqueeze(1) - 1)

        masked_loss = loss_sq * mask.float()
        total_valid = mask.sum()

        if total_valid > 0:
            return masked_loss.sum() / total_valid
        else:
            return torch.tensor(0.0, device=self.device)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            features, labels, lengths, ids = batch
            features = features.to(self.device)
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass returns list of outputs from all stages
            outputs = self.model(features, lengths)

            # Deep Supervision: Sum loss from all stages
            batch_loss = 0

            # 1. Cross Entropy Loss (All Stages)
            for output in outputs:
                batch_loss += self.compute_masked_loss(output, labels, lengths)

            # 2. T-MSE Smoothing Loss (Refinement Stages Only)
            # We apply this to outputs[1:] (the TCN stages)
            tmse_weight = TRAIN_PARAMS.get("tmse_weight", 0.0)
            if tmse_weight > 0 and len(outputs) > 1:
                tmse_loss = 0
                for output in outputs[1:]:
                    tmse_loss += self.compute_tmse_loss(output, lengths)
                batch_loss += tmse_weight * tmse_loss

            batch_loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), TRAIN_PARAMS["grad_clip"]
            )

            self.optimizer.step()

            total_loss += batch_loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                features, labels, lengths, ids = batch
                features = features.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)

                outputs = self.model(features, lengths)

                # Validation loss is sum of all stages too, or just final?
                # Usually we monitor the same objective as training.
                batch_loss = 0
                for output in outputs:
                    batch_loss += self.compute_masked_loss(output, labels, lengths)

                total_loss += batch_loss.item()
                num_batches += 1

                # Use the output of the final stage for metrics
                final_logits = outputs[-1]  # (B, T, C)
                probs = torch.softmax(final_logits, dim=2)

                # Decode predictions for Levenshtein score
                # Iterate over batch to handle variable lengths correctly
                for i in range(len(ids)):
                    length = lengths[i]
                    # Slice valid sequence
                    sample_probs = probs[i, :length, :].cpu().numpy()
                    sample_target = labels[i, :length].cpu().numpy()

                    # Decode
                    pred_seq = decode_predictions(sample_probs)

                    # Target sequence: Remove background (0) and collapse repeats (though GT is usually clean)
                    # The GT provided in metadata is already a list of gesture IDs.
                    # However, the loader provides frame-wise labels including 0.
                    # We need to convert frame-wise GT back to sequence or use the 'labels' from metadata directly?
                    # The loader returns frame-wise labels. Let's process them similarly to predictions
                    # to ensure fair comparison, OR rely on the fact that we can reconstruct the sequence.
                    # Ideally, we should use the raw labels from metadata, but they aren't passed in batch easily.
                    # We will reconstruct from frame-wise labels: remove 0s, collapse repeats.
                    target_seq_raw = [x for x in sample_target if x != 0]
                    # Collapse consecutive duplicates in target (if any exist in frame-wise expansion)
                    target_seq = [
                        x
                        for i, x in enumerate(target_seq_raw)
                        if i == 0 or x != target_seq_raw[i - 1]
                    ]

                    all_preds.append(pred_seq)
                    all_targets.append(target_seq)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        lev_score = compute_levenshtein_score(all_preds, all_targets)

        return avg_loss, lev_score

    def train(self, train_loader, val_loader):
        print(f"Starting training on {self.device}...")

        # Track best Levenshtein score for checkpointing, not just loss
        best_lev_score = float("inf")

        for epoch in range(1, TRAIN_PARAMS["num_epochs"] + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_score = self.validate(val_loader)

            # Cosine Scheduler steps per epoch
            self.scheduler.step()

            print(
                f"Epoch {epoch}/{TRAIN_PARAMS['num_epochs']} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Levenshtein: {val_score:.6f}"
            )

            # Checkpoint based on Levenshtein Score (Cite Lesson 00010, 00008)
            # We trust the metric more than the loss for final model selection
            if val_score < best_lev_score:
                best_lev_score = val_score
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.model_path)
            else:
                self.patience_counter += 1

        print("Training complete.")

    def predict(self, test_loader):
        print("Generating predictions...")
        # Load best model
        if os.path.exists(self.model_path):
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
        else:
            print("Warning: Best model not found, using current weights.")

        self.model.eval()
        results = []
        results.append("Id,Sequence")

        with torch.no_grad():
            for batch in test_loader:
                features, _, lengths, ids = batch
                features = features.to(self.device)
                lengths = lengths.to(self.device)

                outputs = self.model(features, lengths)
                final_logits = outputs[-1]
                probs = torch.softmax(final_logits, dim=2)

                for i in range(len(ids)):
                    sample_id_str = ids[i]
                    try:
                        numeric_id = int(sample_id_str.replace("Sample", ""))
                    except ValueError:
                        numeric_id = sample_id_str

                    length = lengths[i]
                    sample_probs = probs[i, :length, :].cpu().numpy()

                    pred_seq = decode_predictions(sample_probs)

                    # Format: Id,Sequence (space separated)
                    pred_str = " ".join(map(str, pred_seq))
                    results.append(f"{numeric_id},{pred_str}")

        # Save submission
        submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        with open(submission_path, "w") as f:
            for line in results:
                f.write(line + "\n")

        print(f"Submission saved to {submission_path}")
