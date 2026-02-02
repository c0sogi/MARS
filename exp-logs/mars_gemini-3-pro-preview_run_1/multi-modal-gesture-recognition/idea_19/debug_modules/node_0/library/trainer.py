import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import (
    HYPERPARAMS,
    TOTAL_CLASSES,
    BACKGROUND_CLASS_ID,
    CHECKPOINT_DIR,
    WORKING_DIR,
)
from library.utils import set_seed, levenshtein_distance, decode_predictions
from library.model import GCINet


class Trainer:
    """
    Trainer class for the GCI-Net model.
    Encapsulates training, validation, and checkpointing logic.
    """

    def __init__(self, train_loader, val_loader, device=None):
        """
        Args:
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
            device (torch.device, optional): Device to run on.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Initialize Model
        self.model = GCINet().to(self.device)

        # Loss Configuration
        # Background weight = 0.5, others = 1.0
        weights = torch.ones(TOTAL_CLASSES, device=self.device)
        weights[BACKGROUND_CLASS_ID] = HYPERPARAMS["bg_weight"]

        # CrossEntropyLoss with label smoothing and specific weights
        # We do NOT mask padding (padding is 0, which is background),
        # so the model learns to predict background in padded regions.
        self.criterion = nn.CrossEntropyLoss(
            weight=weights,
            label_smoothing=HYPERPARAMS["label_smoothing"],
            reduction="mean",
        )

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=HYPERPARAMS["learning_rate"],
            weight_decay=HYPERPARAMS["weight_decay"],
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=HYPERPARAMS["num_epochs"],
            eta_min=HYPERPARAMS["scheduler_min_lr"],
        )

        # State
        self.best_val_score = float("inf")
        self.early_stopping_counter = 0

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch_idx, batch in enumerate(self.train_loader):
            # Unpack batch
            # skeletons: (B, T, 60), audios: (B, T, 13), lengths: (B), labels: (B, T)
            skeletons, audios, lengths, labels = batch

            skeletons = skeletons.to(self.device)
            audios = audios.to(self.device)
            lengths = lengths.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward Pass
            # logits: (B, T, NumClasses)
            logits = self.model(skeletons, audios, lengths)

            # Reshape for Loss
            # CELoss expects (N, C, T) for inputs and (N, T) for targets
            logits_permuted = logits.permute(0, 2, 1)  # (B, C, T)

            loss = self.criterion(logits_permuted, labels)

            # Backward
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * skeletons.size(0)
            count += skeletons.size(0)

        avg_loss = running_loss / count if count > 0 else 0.0
        return avg_loss

    def validate(self):
        """
        Runs validation loop.
        Computes Loss and Levenshtein Error Rate.
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        total_dist = 0
        total_ref_gestures = 0

        with torch.no_grad():
            for batch in self.val_loader:
                skeletons, audios, lengths, labels = batch

                skeletons = skeletons.to(self.device)
                audios = audios.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)

                # Forward
                logits = self.model(skeletons, audios, lengths)  # (B, T, C)

                # Loss
                logits_permuted = logits.permute(0, 2, 1)
                loss = self.criterion(logits_permuted, labels)
                running_loss += loss.item() * skeletons.size(0)
                count += skeletons.size(0)

                # Metric Calculation (Levenshtein)
                # Get predictions
                probs = torch.softmax(logits, dim=2)
                preds = torch.argmax(probs, dim=2)  # (B, T)

                preds_np = preds.cpu().numpy()
                labels_np = labels.cpu().numpy()
                lengths_np = lengths.cpu().numpy()

                for i in range(len(preds_np)):
                    # Slice by actual length to ignore padding in metric calc if desired,
                    # but decode_predictions handles background/noise.
                    # However, passing the full padded sequence might introduce trailing background.
                    # It's safer to slice using lengths.
                    curr_len = lengths_np[i]
                    p_seq = preds_np[i][:curr_len]
                    t_seq = labels_np[i][:curr_len]

                    # Decode to gesture list
                    pred_gestures = decode_predictions(
                        p_seq,
                        background_id=BACKGROUND_CLASS_ID,
                        min_len=HYPERPARAMS["min_gesture_length"],
                        median_filter_size=HYPERPARAMS["median_filter_size"],
                    )

                    target_gestures = decode_predictions(
                        t_seq,
                        background_id=BACKGROUND_CLASS_ID,
                        min_len=1,  # Ground truth shouldn't be filtered aggressively
                        median_filter_size=1,
                    )

                    dist = levenshtein_distance(pred_gestures, target_gestures)
                    total_dist += dist
                    total_ref_gestures += len(target_gestures)

        avg_loss = running_loss / count if count > 0 else 0.0

        # Error Rate = Total Distance / Total Reference Gestures
        # If total_ref_gestures is 0 (empty val set or empty labels), avoid div by zero
        if total_ref_gestures == 0:
            error_rate = 0.0 if total_dist == 0 else float("inf")
        else:
            error_rate = total_dist / total_ref_gestures

        return avg_loss, error_rate

    def fit(self):
        """
        Main training loop with Early Stopping and Checkpointing.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(1, HYPERPARAMS["num_epochs"] + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_error_rate = self.validate()

            # Scheduler Step
            self.scheduler.step()

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{HYPERPARAMS['num_epochs']} | "
                f"Time: {elapsed:.2f}s | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val Error Rate: {val_error_rate}"
            )

            # Checkpointing and Early Stopping
            if val_error_rate < self.best_val_score:
                self.best_val_score = val_error_rate
                self.early_stopping_counter = 0

                save_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved with Error Rate: {val_error_rate}")
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_counter >= HYPERPARAMS["early_stopping_patience"]:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        print(f"Training complete. Best Validation Error Rate: {self.best_val_score}")
