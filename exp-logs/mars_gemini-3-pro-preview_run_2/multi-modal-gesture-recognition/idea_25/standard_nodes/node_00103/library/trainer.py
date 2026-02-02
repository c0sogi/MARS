import os
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from library.config import (
    WORKING_DIR,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    SEED,
    NUM_CLASSES,
)
from library.utils import set_seed, compute_levenshtein, save_checkpoint
from library.data_loader import GestureDataset, collate_fn
from library.model import GSG_CRCN
from library.loss import DeepSupervisionLoss


class Trainer:
    """
    Manages the training lifecycle of the GSG-CRCN model.
    """

    def __init__(self, subset_size=None):
        """
        Initialize the trainer.

        Args:
            subset_size (int, optional): If provided, limits the dataset size for debugging.
        """
        set_seed(SEED)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Trainer initialized on device: {self.device}")

        # 1. Prepare Data
        print("Initializing Datasets...")
        self.train_dataset = GestureDataset(
            TRAIN_METADATA_PATH, is_train=True, augment=True, subset_size=subset_size
        )
        self.val_dataset = GestureDataset(
            VAL_METADATA_PATH, is_train=True, augment=False, subset_size=subset_size
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        # 2. Initialize Model
        self.model = GSG_CRCN().to(self.device)

        # 3. Initialize Loss and Optimizer
        self.criterion = DeepSupervisionLoss().to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # 4. Training State
        self.best_val_score = float("inf")  # Lower is better (Levenshtein Error Rate)
        self.patience_counter = 0
        self.start_epoch = 0
        self.checkpoint_path = os.path.join(WORKING_DIR, "best_model.pth")

    def decode_batch(self, logits, lengths):
        """
        Decodes batch logits into gesture sequences.
        Strategy: Argmax -> Collapse Repeats -> Remove Background (0).

        Args:
            logits: (N, C, L) Tensor of class logits (Stage 3 output).
            lengths: (N,) List or Tensor of sequence lengths.

        Returns:
            list[list[int]]: Predicted gesture sequences.
        """
        preds = []
        # Permute to (N, L, C) for easier processing
        probs = torch.softmax(logits, dim=1)
        classes = torch.argmax(probs, dim=1).cpu().numpy()  # (N, L)

        for i, length in enumerate(lengths):
            # Get valid sequence
            seq = classes[i, :length]

            # Collapse repeats
            collapsed = []
            prev = -1
            for label in seq:
                if label != prev:
                    collapsed.append(label)
                    prev = label

            # Remove background (0)
            final_seq = [x for x in collapsed if x != 0]
            preds.append(final_seq)

        return preds

    def get_truth_sequences(self, targets, lengths):
        """
        Extracts ground truth sequences from tensor targets.

        Args:
            targets: (N, L) Tensor of class indices.
            lengths: (N,) List of lengths.

        Returns:
            list[list[int]]: Ground truth sequences.
        """
        truths = []
        targets_np = targets.cpu().numpy()

        for i, length in enumerate(lengths):
            seq = targets_np[i, :length]

            # Collapse repeats and remove background to get the event list
            # Note: The target tensor is frame-wise. To get the list of gestures,
            # we collapse repeats and remove 0.
            collapsed = []
            prev = -1
            for label in seq:
                if label != prev:
                    collapsed.append(label)
                    prev = label

            final_seq = [x for x in collapsed if x != 0]
            truths.append(final_seq)

        return truths

    def train_epoch(self, epoch_idx):
        self.model.train()
        total_loss = 0.0
        metrics_accum = {}

        start_time = time.time()

        for batch_idx, batch_data in enumerate(self.train_loader):
            # Unpack batch
            features, cls_targets, bnd_targets, lengths, mask, _ = batch_data

            # Move to device
            features = features.to(self.device)
            cls_targets = cls_targets.to(self.device)
            bnd_targets = bnd_targets.to(self.device)
            mask = mask.to(self.device)

            # Forward Pass
            self.optimizer.zero_grad()
            outputs = self.model(features, mask)

            # Compute Loss
            loss, metrics = self.criterion(outputs, cls_targets, bnd_targets, mask)

            # Backward Pass
            loss.backward()

            # Gradient Clipping (Optional but recommended for LSTMs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            # Accumulate
            total_loss += loss.item()
            for k, v in metrics.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + v

        # Average metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_metrics = {k: v / len(self.train_loader) for k, v in metrics_accum.items()}

        duration = time.time() - start_time

        print(f"Epoch {epoch_idx} [Train] Loss: {avg_loss:.6f} | Time: {duration:.2f}s")
        # Print breakdown of Stage 3 losses as they are most relevant
        print(
            f"    S3_Cls: {avg_metrics.get('stage3_loss_cls', 0):.4f} | "
            f"S3_Bnd: {avg_metrics.get('stage3_loss_bnd', 0):.4f} | "
            f"S3_Smooth: {avg_metrics.get('stage3_loss_smooth', 0):.4f}"
        )

        return avg_loss

    def validate_epoch(self, epoch_idx):
        self.model.eval()
        total_loss = 0.0

        all_preds = []
        all_truths = []

        with torch.no_grad():
            for batch_data in self.val_loader:
                features, cls_targets, bnd_targets, lengths, mask, _ = batch_data

                features = features.to(self.device)
                cls_targets = cls_targets.to(self.device)
                bnd_targets = bnd_targets.to(self.device)
                mask = mask.to(self.device)

                outputs = self.model(features, mask)

                # Compute Loss
                loss, _ = self.criterion(outputs, cls_targets, bnd_targets, mask)
                total_loss += loss.item()

                # Decode Predictions (Using Stage 3 Class Logits)
                # Stage 3 output key: 'stage3' -> 'cls' is (N, C, L) based on model.py output
                # Wait, model.py output for stage 3:
                # "cls": s3_cls_logits.permute(0, 2, 1) which is (N, L, 21)
                # Let's check model.py return again.
                # model.py: s3_cls_logits is output of conv1d (N, C, L).
                # model.py return: "cls": s3_cls_logits.permute(0, 2, 1) -> (N, L, C)
                # My decode_batch expects (N, C, L) or I adapt it.
                # Let's adapt decode_batch to take (N, L, C) or permute here.

                s3_logits = outputs["stage3"]["cls"]  # (N, L, C)
                # Permute back to (N, C, L) for decode_batch logic or change decode_batch
                s3_logits_permuted = s3_logits.permute(0, 2, 1)  # (N, C, L)

                batch_preds = self.decode_batch(s3_logits_permuted, lengths)
                batch_truths = self.get_truth_sequences(cls_targets, lengths)

                all_preds.extend(batch_preds)
                all_truths.extend(batch_truths)

        avg_loss = total_loss / len(self.val_loader)

        # Compute Levenshtein Score
        score = compute_levenshtein(all_preds, all_truths)

        print(
            f"Epoch {epoch_idx} [Valid] Loss: {avg_loss:.6f} | Score (Levenshtein): {score}"
        )

        return avg_loss, score

    def train(self):
        print(f"Starting training for {NUM_EPOCHS} epochs...")

        for epoch in range(1, NUM_EPOCHS + 1):
            _ = self.train_epoch(epoch)
            val_loss, val_score = self.validate_epoch(epoch)

            # Early Stopping and Checkpointing based on Levenshtein Score
            # (The competition metric is the error rate, so lower is better)
            if val_score < self.best_val_score:
                print(
                    f"    New best score! ({self.best_val_score} -> {val_score}) Saving model..."
                )
                self.best_val_score = val_score
                self.patience_counter = 0

                # Save Checkpoint
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_score": self.best_val_score,
                    },
                    self.checkpoint_path,
                )
            else:
                self.patience_counter += 1
                print(
                    f"    No improvement. Patience: {self.patience_counter}/{EARLY_STOPPING_PATIENCE}"
                )

            if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Score: {self.best_val_score}")
