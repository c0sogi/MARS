import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import (
    LEARNING_RATE,
    NUM_EPOCHS,
    PATIENCE,
    WEIGHT_DECAY,
    GRADIENT_CLIP,
    CLASS_WEIGHTS,
    BEST_MODEL_PATH,
    NUM_CLASSES,
    WORKING_DIR,
)
from library.utils import levenshtein_score


class Trainer:
    """
    Trainer class for the Hybrid Gesture Recognition Model.
    """

    def __init__(self, model, train_loader, val_loader, device):
        """
        Args:
            model (nn.Module): The hybrid model to train.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            device (torch.device): Device to run training on.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Define Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Define Loss Function with Class Weights
        # Weights: 0.1 for background, 1.0 for gestures
        weights_tensor = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(
            self.device
        )
        self.criterion = nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=-1)

        # Early Stopping variables
        self.patience = PATIENCE
        self.best_val_loss = float("inf")
        self.counter = 0

    def _decode_sequence(self, frame_labels):
        """
        Decodes frame-wise labels into a sequence of gesture IDs.
        Logic: Collapse consecutive duplicates, then remove background (0).
        """
        if len(frame_labels) == 0:
            return []

        # Collapse duplicates
        collapsed = [frame_labels[0]]
        for i in range(1, len(frame_labels)):
            if frame_labels[i] != frame_labels[i - 1]:
                collapsed.append(frame_labels[i])

        # Remove background (0)
        gesture_sequence = [x for x in collapsed if x != 0]
        return gesture_sequence

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, (x, y, lengths, ids) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            # lengths stays on CPU for pack_padded_sequence usually,
            # but model handles moving it if necessary.

            self.optimizer.zero_grad()

            # Forward Pass
            # Returns logits from both stages
            logits1, logits2 = self.model(x, lengths)

            # Flatten outputs for CrossEntropyLoss
            # Shape: (Batch * Time, NumClasses)
            # We mask the loss calculation for padded areas using ignore_index logic implicitly
            # if we had a padding index, but here we rely on the fact that y is padded with 0 (background).
            # However, standard practice with variable lengths is to flatten and compute.
            # Since 0 is a valid class (background) but weighted low, we don't ignore it.
            # We should technically mask out the padding if it affects the loss,
            # but usually 0-padding matches the background class 0.
            # To be precise, we can use the lengths to mask, but standard CE on padded 0s
            # with 0-target is acceptable given the low weight.

            # Reshape
            batch_size, max_len, num_classes = logits1.shape

            flat_logits1 = logits1.reshape(-1, num_classes)
            flat_logits2 = logits2.reshape(-1, num_classes)
            flat_y = y.reshape(-1)

            # Compute Loss
            # We apply supervision to both stages
            loss1 = self.criterion(flat_logits1, flat_y)
            loss2 = self.criterion(flat_logits2, flat_y)

            total_loss = loss1 + loss2

            # Backward Pass
            total_loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRADIENT_CLIP)

            self.optimizer.step()

            running_loss += total_loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate_epoch(self, epoch_idx):
        """
        Runs validation and computes metrics.
        """
        self.model.eval()
        running_loss = 0.0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (x, y, lengths, ids) in enumerate(self.val_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward Pass
                logits1, logits2 = self.model(x, lengths)

                # Compute Loss
                batch_size, max_len, num_classes = logits1.shape
                flat_logits1 = logits1.reshape(-1, num_classes)
                flat_logits2 = logits2.reshape(-1, num_classes)
                flat_y = y.reshape(-1)

                loss1 = self.criterion(flat_logits1, flat_y)
                loss2 = self.criterion(flat_logits2, flat_y)
                total_loss = loss1 + loss2

                running_loss += total_loss.item()

                # Decoding for Levenshtein Score
                # Use Stage 2 logits for final prediction
                probs = torch.softmax(logits2, dim=2)
                preds_frame = torch.argmax(probs, dim=2)  # (Batch, Time)

                # Iterate over batch to decode sequences
                x_cpu = x.cpu()
                y_cpu = y.cpu()
                preds_cpu = preds_frame.cpu()

                for i in range(batch_size):
                    length = lengths[i]
                    # Get valid frames
                    p_seq = preds_cpu[i, :length].tolist()
                    t_seq = y_cpu[i, :length].tolist()

                    # Decode to gesture list
                    decoded_pred = self._decode_sequence(p_seq)
                    decoded_target = self._decode_sequence(t_seq)

                    all_preds.append(decoded_pred)
                    all_targets.append(decoded_target)

        avg_loss = running_loss / len(self.val_loader)

        # Calculate Levenshtein Score
        lev_score = levenshtein_score(all_preds, all_targets)

        return avg_loss, lev_score

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_score = self.validate_epoch(epoch)

            print(
                f"Epoch {epoch}/{NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Score (Lev): {val_score:.6f}"
            )

            # Early Stopping Check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                # Save best model
                torch.save(self.model.state_dict(), BEST_MODEL_PATH)
                print(f"Validation loss improved. Model saved to {BEST_MODEL_PATH}")
            else:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                if self.counter >= self.patience:
                    print("Early stopping triggered.")
                    break
