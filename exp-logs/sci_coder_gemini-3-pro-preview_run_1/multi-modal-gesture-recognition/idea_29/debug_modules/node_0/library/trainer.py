import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config
from library.utils import (
    set_seed,
    compute_levenshtein_score,
    median_filter,
    rle_decode,
    levenshtein_distance,
)
from library.model import BS_MPII
from library.data_loader import get_dataloaders


class Trainer:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)

        # Data Loaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders()

        # Model
        self.model = BS_MPII().to(self.device)

        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.COSINE_T_MAX
        )

        # Loss Functions
        # 1. Classification Loss: CrossEntropy with Label Smoothing & Class Weights
        # Weight background class (0) by 0.5, others by 1.0
        class_weights = torch.ones(Config.NUM_CLASSES).to(self.device)
        class_weights[0] = Config.BACKGROUND_CLASS_WEIGHT

        self.criterion_class = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=Config.LABEL_SMOOTHING
        )

        # 2. Boundary Loss: BCEWithLogitsLoss
        self.criterion_boundary = nn.BCEWithLogitsLoss()

        # Training State
        self.best_val_score = float("inf")
        self.patience_counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_class_loss = 0.0
        total_boundary_loss = 0.0

        for batch in self.train_loader:
            if batch is None:
                continue

            # Move data to device
            skeleton = batch["skeleton"].to(self.device)
            audio = batch["audio"].to(self.device)
            labels = batch["labels"].to(self.device)
            boundaries = batch["boundaries"].to(self.device)
            lengths = batch["lengths"].to(self.device)

            # Forward Pass
            self.optimizer.zero_grad()
            outputs = self.model(skeleton, audio, lengths)

            class_logits = outputs["class_logits"]  # (B, T, NumClasses)
            boundary_logits = outputs["boundary_logits"]  # (B, T)

            # Flatten for Loss Calculation
            # We include padding in the loss calculation as "Background" (Class 0)
            # This is "Supervised Padding"
            flat_class_logits = class_logits.view(-1, Config.NUM_CLASSES)
            flat_labels = labels.view(-1)

            flat_boundary_logits = boundary_logits.view(-1)
            flat_boundaries = boundaries.view(-1)

            # Calculate Losses
            loss_cls = self.criterion_class(flat_class_logits, flat_labels)
            loss_bnd = self.criterion_boundary(flat_boundary_logits, flat_boundaries)

            # Multi-Task Loss Combination
            loss = loss_cls + Config.BOUNDARY_LOSS_WEIGHT * loss_bnd

            # Backward
            loss.backward()
            self.optimizer.step()

            # Stats
            total_loss += loss.item()
            total_class_loss += loss_cls.item()
            total_boundary_loss += loss_bnd.item()

        avg_loss = total_loss / len(self.train_loader)
        avg_cls = total_class_loss / len(self.train_loader)
        avg_bnd = total_boundary_loss / len(self.train_loader)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {avg_loss:.4f} (Cls: {avg_cls:.4f}, Bnd: {avg_bnd:.4f})"
        )

        return avg_loss

    def validate(self):
        self.model.eval()
        total_dist = 0
        total_gestures = 0

        with torch.no_grad():
            for batch in self.val_loader:
                if batch is None:
                    continue

                skeleton = batch["skeleton"].to(self.device)
                audio = batch["audio"].to(self.device)
                labels = batch["labels"].to(self.device)
                lengths = batch["lengths"].to(self.device)

                # Forward
                outputs = self.model(skeleton, audio, lengths)
                class_logits = outputs["class_logits"]  # (B, T, C)

                # Predictions
                preds = torch.argmax(class_logits, dim=2)  # (B, T)

                # Decode batch
                batch_size = preds.size(0)
                for i in range(batch_size):
                    length = lengths[i].item()

                    # Extract valid sequence
                    p_seq = preds[i, :length].cpu().numpy()
                    t_seq = labels[i, :length].cpu().numpy()

                    # 1. Median Filter Smoothing
                    p_smooth = median_filter(
                        p_seq, window_size=Config.MEDIAN_FILTER_SIZE
                    )

                    # 2. RLE Decode to get gesture list
                    p_gestures = rle_decode(
                        p_smooth, min_length=Config.MIN_GESTURE_LENGTH
                    )
                    t_gestures = rle_decode(
                        t_seq, min_length=1
                    )  # GT usually clean, but decode to be safe

                    # 3. Levenshtein Distance
                    dist = levenshtein_distance(p_gestures, t_gestures)

                    total_dist += dist
                    total_gestures += len(t_gestures)

        # Compute Metric
        if total_gestures == 0:
            score = 0.0
        else:
            score = total_dist / total_gestures

        print(f"Validation Levenshtein Error Rate: {score}")
        return score

    def fit(self):
        print("Starting training...")
        Config.print_config()

        for epoch in range(Config.NUM_EPOCHS):
            start_time = time.time()

            # Train
            self.train_epoch(epoch)

            # Validate
            val_score = self.validate()

            # Scheduler Step
            self.scheduler.step()

            epoch_time = time.time() - start_time
            print(f"Epoch Time: {epoch_time:.2f}s")

            # Checkpointing & Early Stopping
            if val_score < self.best_val_score:
                print(
                    f"Score improved from {self.best_val_score} to {val_score}. Saving model..."
                )
                self.best_val_score = val_score
                self.patience_counter = 0
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
            else:
                self.patience_counter += 1
                print(
                    f"Score did not improve. Patience: {self.patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Score: {self.best_val_score}")


def run_training():
    trainer = Trainer()
    trainer.fit()
