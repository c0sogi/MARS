import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.ndimage import median_filter
from itertools import groupby

from library.config import PATHS, get_hyperparams
from library.utils import set_seed, evaluate_levenshtein_accuracy, save_checkpoint
from library.losses import ActionSegmentationLoss
from library.data_loader import get_dataloaders
from library.model import MG_CRGN


class Trainer:
    """
    Trainer class for the MG-CRGN model.
    Handles training, validation, and checkpointing.
    """

    def __init__(self):
        self.hp = get_hyperparams()
        self.device = torch.device(self.hp["device"])
        set_seed(self.hp["seed"])

        # Data Loaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            load_cached_data=True
        )

        # Model
        self.model = MG_CRGN().to(self.device)

        # Loss Function
        self.criterion = ActionSegmentationLoss().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.hp["learning_rate"],
            weight_decay=self.hp["weight_decay"],
        )

        # Training State
        self.best_val_score = float("inf")
        self.start_epoch = 0

    def _decode_predictions(self, probs, mask):
        """
        Decodes frame-wise probabilities into gesture sequences.
        Applies Argmax -> Median Filter -> Collapse Duplicates -> Remove Background.

        Args:
            probs (torch.Tensor): (B, NumClasses, T) Class probabilities.
            mask (torch.Tensor): (B, T) Valid frame mask.

        Returns:
            list[list[int]]: List of predicted gesture IDs for each sample in batch.
        """
        # Convert to numpy
        probs_np = probs.detach().cpu().numpy()  # (B, C, T)
        mask_np = mask.detach().cpu().numpy()  # (B, T)

        predictions = []

        # Median filter size (approx 15 frames ~ 1.5 sec based on analysis)
        filter_size = 15

        for i in range(probs_np.shape[0]):
            # Get valid length
            valid_len = int(mask_np[i].sum())
            if valid_len == 0:
                predictions.append([])
                continue

            # Slice valid frames: (C, T_valid) -> (T_valid, C)
            sample_probs = probs_np[i, :21, :valid_len].transpose(1, 0)

            # Argmax
            raw_labels = np.argmax(sample_probs, axis=1)

            # Median Filtering for smoothness
            # mode='nearest' corresponds to "Nearest-Neighbor Padding" for boundary protection
            smooth_labels = median_filter(raw_labels, size=filter_size, mode="nearest")

            # Collapse repetitions and remove background (0)
            gesture_seq = []
            for key, group in groupby(smooth_labels):
                if key != 0:  # 0 is background
                    gesture_seq.append(int(key))

            predictions.append(gesture_seq)

        return predictions

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (features, targets, mask, _) in enumerate(self.train_loader):
            features = features.to(self.device)
            targets = targets.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()

            # Forward Pass: Returns list of outputs [out1, out2, out3]
            outputs = self.model(features, mask)

            # Compute Loss (Deep Supervision)
            loss = self.criterion(outputs, targets, mask)

            # Backward Pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.hp["grad_clip"]
            )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self):
        """
        Runs validation loop and computes Levenshtein error rate.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for features, targets, mask, _ in self.val_loader:
                features = features.to(self.device)
                targets_dev = targets.to(self.device)
                mask = mask.to(self.device)

                # Forward Pass
                outputs = self.model(features, mask)

                # Compute Loss (for monitoring)
                loss = self.criterion(outputs, targets_dev, mask)
                total_loss += loss.item()
                num_batches += 1

                # Decode Predictions using Stage 3 output (index 2)
                # Output shape: (B, NumClasses+1, T). We need first 21 channels.
                stage3_out = outputs[2]
                cls_probs = stage3_out[:, :21, :]  # (B, 21, T)

                batch_preds = self._decode_predictions(cls_probs, mask)

                # Process Targets
                # Convert padded tensor targets back to list of lists without padding/background
                targets_np = targets.numpy()
                mask_np = mask.cpu().numpy()

                for i in range(targets_np.shape[0]):
                    valid_len = int(mask_np[i].sum())
                    raw_t = targets_np[i, :valid_len]
                    # Collapse targets to sequence (ground truth is already frame-wise in loader)
                    # Ground truth labels provided in metadata are sequences, but loader expands them.
                    # To evaluate Levenshtein correctly against the original sequence logic:
                    # We collapse the frame-wise targets.
                    t_seq = []
                    for key, group in groupby(raw_t):
                        if key != 0:
                            t_seq.append(int(key))
                    all_targets.append(t_seq)

                all_preds.extend(batch_preds)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Compute Metric
        error_rate = evaluate_levenshtein_accuracy(all_preds, all_targets)

        return avg_loss, error_rate

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on {self.device}...")
        print(f"Hyperparameters: {self.hp}")

        patience = self.hp["patience"]
        patience_counter = 0

        for epoch in range(1, self.hp["epochs"] + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_score = self.validate()

            print(
                f"Epoch {epoch}/{self.hp['epochs']} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Error Rate: {val_score}"
            )

            # Checkpoint & Early Stopping
            if val_score < self.best_val_score:
                self.best_val_score = val_score
                patience_counter = 0
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    PATHS["model_save_path"],
                )
                print(f"New best model saved with Error Rate: {val_score}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        print(f"Training complete. Best Validation Error Rate: {self.best_val_score}")
