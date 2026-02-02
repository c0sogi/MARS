import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.model import RCGRNet
from library.data_loader import get_data_loaders
from library.utils import (
    set_seed,
    compute_levenshtein_score,
    post_process_sequence,
    decode_predictions_rle,
)


class Trainer:
    def __init__(self, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        set_seed(Config.SEED)

        # Initialize Model
        self.model = RCGRNet().to(self.device)

        # Loss Function with Class Balancing
        # Index 0 is background
        class_weights = torch.ones(Config.NUM_CLASSES, device=self.device)
        class_weights[0] = Config.BG_WEIGHT

        # We use reduction='none' to handle masking manually for variable lengths
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=Config.LABEL_SMOOTHING,
            reduction="none",
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_idx, batch_data in enumerate(loader):
            skels, audios, labels, lengths = batch_data

            # Skip empty batches
            if skels is None:
                continue

            # Move to device
            skels = skels.to(self.device)
            audios = audios.to(self.device)
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)

            self.optimizer.zero_grad()

            # Forward Pass
            # logits: (B, T, NumClasses)
            logits = self.model(skels, audios)

            # Create Mask for variable lengths
            # (B, T)
            B, T, C = logits.shape
            mask = torch.arange(T, device=self.device).expand(B, T) < lengths.unsqueeze(
                1
            )

            # Flatten for Loss
            logits_flat = logits.view(-1, C)
            labels_flat = labels.view(-1)
            mask_flat = mask.view(-1)

            # Compute Loss
            raw_loss = self.criterion(logits_flat, labels_flat)

            # Apply Mask
            masked_loss = (raw_loss * mask_flat).sum() / (mask_flat.sum() + 1e-6)

            # Backward
            masked_loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += masked_loss.item() * B
            total_samples += B

        return total_loss / max(1, total_samples)

    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_data in loader:
                skels, audios, labels, lengths = batch_data

                if skels is None:
                    continue

                skels = skels.to(self.device)
                audios = audios.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)

                # Forward
                logits = self.model(skels, audios)

                # Loss Calculation
                B, T, C = logits.shape
                mask = torch.arange(T, device=self.device).expand(
                    B, T
                ) < lengths.unsqueeze(1)

                logits_flat = logits.view(-1, C)
                labels_flat = labels.view(-1)
                mask_flat = mask.view(-1)

                raw_loss = self.criterion(logits_flat, labels_flat)
                masked_loss = (raw_loss * mask_flat).sum() / (mask_flat.sum() + 1e-6)

                total_loss += masked_loss.item() * B
                total_samples += B

                # Metric Calculation: Levenshtein
                probs = torch.softmax(logits, dim=2)

                for i in range(B):
                    length = lengths[i].item()

                    # Get valid sequence for this sample
                    sample_probs = probs[i, :length]  # (L, C)
                    sample_labels = labels[i, :length]  # (L,)

                    # Decode Predictions
                    pred_seq = post_process_sequence(sample_probs)
                    all_preds.append(pred_seq)

                    # Decode Targets (Ground Truth)
                    # We use the same RLE logic to extract the gesture sequence from frame labels
                    target_seq = decode_predictions_rle(sample_labels.cpu().numpy())
                    all_targets.append(target_seq)

        avg_loss = total_loss / max(1, total_samples)
        lev_score = compute_levenshtein_score(all_preds, all_targets)

        return avg_loss, lev_score

    def fit(
        self, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    ):
        print(f"Starting training on device: {self.device}")

        train_loader, val_loader, _ = get_data_loaders(
            batch_size=batch_size, debug=debug
        )

        best_lev_score = float("inf")
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_loss, val_lev = self.validate(val_loader)

            self.scheduler.step()

            duration = time.time() - start_time

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Time: {duration:.1f}s | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Levenshtein: {val_lev}"
            )

            # Checkpoint based on Levenshtein Score (Lower is better)
            if val_lev < best_lev_score:
                print(
                    f"Validation score improved ({best_lev_score} -> {val_lev}). Saving model..."
                )
                best_lev_score = val_lev
                torch.save(self.model.state_dict(), best_model_path)

        print(f"Training complete. Best Validation Levenshtein Score: {best_lev_score}")


def main():
    trainer = Trainer()
    trainer.fit()


# Note: The __name__ == "__main__" block is omitted as per instructions.
