import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import set_seed, get_device, ensure_dir
from library.model import MultiStreamGRU
from library.losses import SequenceLoss
from library.data_loader import GestureDataset, collate_fn
from library.metrics import levenshtein_distance, decode_sequence
import pandas as pd


class Trainer:
    """
    Manages the training, validation, and checkpointing of the MSRN model.
    """

    def __init__(self, limit=None, load_cached_data=True):
        """
        Args:
            limit (int, optional): Limit the number of samples for debugging.
            load_cached_data (bool): Whether to use cached data in the dataset.
                                     (Note: GestureDataset handles caching logic internally,
                                     this flag is primarily for consistency with requirements).
        """
        # 1. Setup
        set_seed(Config.SEED)
        self.device = get_device()
        ensure_dir(Config.MODEL_SAVE_PATH)

        # 2. Data Loaders
        # We pass the limit to the dataset for debugging purposes
        self.train_dataset = GestureDataset(split="train", augment=True, limit=limit)
        self.val_dataset = GestureDataset(split="val", augment=False, limit=limit)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        # 3. Model
        self.model = MultiStreamGRU().to(self.device)

        # 4. Loss, Optimizer, Scheduler
        self.criterion = SequenceLoss(ignore_index=-100).to(self.device)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=Config.LR_FACTOR,
            patience=Config.LR_PATIENCE,
            min_lr=Config.MIN_LR,
        )

        # 5. Training State
        self.best_val_loss = float("inf")
        self.best_val_metric = float("inf")  # Cite Lesson 00025
        self.early_stopping_counter = 0

    def _calculate_accuracy(self, logits, targets):
        """
        Calculates accuracy ignoring the padded values (-100).
        Args:
            logits: (B, T, C)
            targets: (B, T)
        Returns:
            accuracy (float)
        """
        preds = torch.argmax(logits, dim=2)  # (B, T)

        # Create mask for valid targets (not -100)
        mask = targets != -100

        correct = (preds == targets) & mask

        total_valid = mask.sum().item()
        total_correct = correct.sum().item()

        if total_valid == 0:
            return 0.0

        return total_correct / total_valid

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        for batch_idx, (skeleton, audio, labels, lengths, indices) in enumerate(
            self.train_loader
        ):
            # Move to device
            skeleton = skeleton.to(self.device)
            audio = audio.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(skeleton, audio)

            # Compute Loss
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimizer Step
            self.optimizer.step()

            # Metrics
            total_loss += loss.item()
            total_acc += self._calculate_accuracy(logits, labels)
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_acc = total_acc / num_batches if num_batches > 0 else 0.0
        return avg_loss, avg_acc

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_dist = 0.0
        total_gestures = 0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, (skeleton, audio, labels, lengths, indices) in enumerate(
                self.val_loader
            ):
                skeleton = skeleton.to(self.device)
                audio = audio.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(skeleton, audio)

                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                total_acc += self._calculate_accuracy(logits, labels)
                num_batches += 1

                # Calculate Levenshtein Metric (Cite Lesson 00025)
                preds = torch.argmax(logits, dim=2).cpu().numpy()
                batch_size = skeleton.shape[0]

                for i in range(batch_size):
                    idx = indices[i]
                    # Retrieve Ground Truth from Metadata
                    row = self.val_dataset.df.iloc[idx]
                    raw_val = row["labels"]
                    if pd.isna(raw_val) or str(raw_val).strip() == "":
                        gt_list = []
                    else:
                        gt_list = [int(x) for x in str(raw_val).split(",")]

                    # Decode Prediction
                    valid_len = lengths[i]
                    pred_seq = preds[i, :valid_len]
                    # Apply median filter and decode
                    pred_list = decode_sequence(pred_seq, apply_median_filter=True)

                    dist = levenshtein_distance(pred_list, gt_list)
                    total_dist += dist
                    total_gestures += len(gt_list)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_acc = total_acc / num_batches if num_batches > 0 else 0.0
        avg_metric = total_dist / total_gestures if total_gestures > 0 else 0.0
        return avg_loss, avg_acc, avg_metric

    def fit(self, epochs=Config.NUM_EPOCHS):
        """
        Runs the full training loop with early stopping based on Levenshtein metric.
        Args:
            epochs (int): Maximum number of epochs to train.
        """
        print(f"Starting training on device: {self.device}")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_metric = self.validate()

            # Print full precision metrics
            print(
                f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, "
                f"Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}, Val Metric {val_metric:.4f}"
            )

            # Scheduler Step (Keep using Loss for scheduler stability)
            self.scheduler.step(val_loss)

            # Checkpointing and Early Stopping based on Metric (Cite Lesson 00025)
            if val_metric < self.best_val_metric:
                self.best_val_metric = val_metric
                self.early_stopping_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"  New best model saved! Metric: {val_metric:.4f}")
            else:
                self.early_stopping_counter += 1

            if self.early_stopping_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

        print("Training complete.")
