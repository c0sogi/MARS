import os
import torch
import torch.nn as nn
import numpy as np
from library.config import (
    CACHE_DIR,
    EARLY_STOPPING_PATIENCE,
    GRADIENT_CLIP_VAL,
    LOSS_WEIGHTS,
)
from library.utils import compute_challenge_score, decode_predictions, rle_collapse


class Trainer:
    """
    Trainer class for the Gesture Recognition GRU model.
    Handles training loop, validation, early stopping, and prediction.
    """

    def __init__(self, model, train_loader, val_loader, optimizer, device):
        """
        Args:
            model (nn.Module): The model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            optimizer (Optimizer): PyTorch optimizer.
            device (torch.device): Device to run training on.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device

        # Initialize CrossEntropyLoss with class weights
        # We ignore the padding in the manual masking step, but weights handle class imbalance
        weights = LOSS_WEIGHTS.to(device)
        self.criterion = nn.CrossEntropyLoss(weight=weights, reduction="none")

        self.best_val_loss = float("inf")
        self.checkpoint_dir = CACHE_DIR
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Scheduler (Cite solution_lesson_node_00007)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

    def _compute_masked_loss(self, logits, targets, lengths):
        """
        Computes CrossEntropyLoss masked by sequence lengths to ignore padding.

        Args:
            logits: (B, T, C)
            targets: (B, T)
            lengths: (B,)

        Returns:
            torch.Tensor: Scalar loss
        """
        # Create mask based on lengths
        # shape: (B, T)
        batch_size, max_len = targets.size()
        mask = torch.arange(max_len, device=self.device).expand(
            batch_size, max_len
        ) < lengths.unsqueeze(1)

        # Flatten
        logits_flat = logits.view(-1, logits.size(-1))  # (B*T, C)
        targets_flat = targets.view(-1)  # (B*T)
        mask_flat = mask.view(-1)  # (B*T)

        # Compute element-wise loss
        raw_loss = self.criterion(logits_flat, targets_flat)  # (B*T)

        # Apply mask
        masked_loss = raw_loss * mask_flat.float()

        # Average over valid elements
        loss = masked_loss.sum() / (mask_flat.sum() + 1e-8)

        return loss

    def train_epoch(self):
        """Runs one epoch of training."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (features, targets, lengths, ids) in enumerate(
            self.train_loader
        ):
            features = features.to(self.device)
            targets = targets.to(self.device)
            lengths = lengths.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(features, lengths)

            # Compute Loss
            loss = self._compute_masked_loss(logits, targets, lengths)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP_VAL)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self):
        """Runs validation loop and computes metrics."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_preds_seq = []
        all_targets_seq = []

        with torch.no_grad():
            for features, targets, lengths, ids in self.val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                lengths = lengths.to(self.device)

                # Forward pass
                logits = self.model(features, lengths)

                # Compute Loss
                loss = self._compute_masked_loss(logits, targets, lengths)
                total_loss += loss.item()
                num_batches += 1

                # Decode predictions and targets for metric computation
                # Convert logits to numpy for decoding
                logits_np = logits.cpu().numpy()
                targets_np = targets.cpu().numpy()
                lengths_np = lengths.cpu().numpy()

                for i in range(len(ids)):
                    length = lengths_np[i]
                    # Slice valid frames
                    valid_logits = logits_np[i, :length, :]
                    valid_targets = targets_np[i, :length]

                    # Decode prediction
                    pred_seq = decode_predictions(valid_logits)
                    all_preds_seq.append(pred_seq)

                    # Decode target (RLE collapse)
                    target_seq = rle_collapse(
                        valid_targets, remove_background=True, background_class=0
                    )
                    all_targets_seq.append(target_seq)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Compute Levenshtein Score
        score = compute_challenge_score(all_targets_seq, all_preds_seq)

        return avg_loss, score

    def fit(self, epochs):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {epochs} epochs on {self.device}...")

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_score = self.validate()

            print(
                f"Epoch {epoch}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Score: {val_score}"
            )

            # Step Scheduler
            self.scheduler.step(val_loss)

            # Early Stopping and Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(
                    f"Validation loss improved. Model saved to {self.checkpoint_path}"
                )
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        # Load best model before finishing
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(
                torch.load(self.checkpoint_path, map_location=self.device)
            )
            print("Loaded best model from checkpoint.")

    def predict(self, test_loader):
        """
        Generates predictions for the test set.

        Args:
            test_loader (DataLoader): DataLoader for test data.

        Returns:
            dict: {sample_id: [gesture_id, ...]}
        """
        self.model.eval()
        predictions = {}

        print("Generating predictions...")
        with torch.no_grad():
            for features, _, lengths, ids in test_loader:
                features = features.to(self.device)
                lengths = lengths.to(self.device)

                logits = self.model(features, lengths)
                logits_np = logits.cpu().numpy()
                lengths_np = lengths.cpu().numpy()

                for i, sample_id in enumerate(ids):
                    length = lengths_np[i]
                    valid_logits = logits_np[i, :length, :]

                    pred_seq = decode_predictions(valid_logits)
                    predictions[sample_id] = pred_seq

        return predictions
