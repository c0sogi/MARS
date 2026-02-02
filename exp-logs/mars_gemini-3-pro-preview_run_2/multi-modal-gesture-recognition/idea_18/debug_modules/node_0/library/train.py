import os
import random
import numpy as np
import torch
import torch.optim as optim
from itertools import groupby

from library.config import Config
from library.model import DSL_CRCN
from library.loss import DeepSupervisionLoss
from library.data_loader import get_dataloaders
from library.utils import decode_predictions, compute_levenshtein_score


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self):
        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        self.model = DSL_CRCN().to(self.device)

        # Initialize Loss
        self.criterion = DeepSupervisionLoss().to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Checkpoint path
        self.best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        metrics_sum = {}
        num_batches = 0

        for features, targets, lengths in loader:
            features = features.to(self.device)
            targets = targets.to(self.device)
            lengths = lengths.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(
                features, mask=None
            )  # Mask handled in loss via lengths/targets logic if needed,
            # but model accepts mask. Let's generate mask from lengths.

            # Generate boolean mask for the model/loss
            # Shape (B, T)
            max_len = features.size(1)
            batch_size = features.size(0)
            idx_range = (
                torch.arange(max_len, device=self.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            mask = idx_range < lengths.unsqueeze(1)

            # Re-run forward with mask if the model supports it to be safe (DSL_CRCN does)
            # Although we ran it without mask above, let's run it correctly with mask or just pass mask to loss.
            # The model code applies mask internally between stages if provided.
            outputs = self.model(features, mask=mask)

            # Compute Loss
            loss, batch_metrics = self.criterion(outputs, targets, lengths)

            # Backward
            loss.backward()
            self.optimizer.step()

            # Logging
            total_loss += loss.item()
            for k, v in batch_metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v

            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}

        return avg_loss, avg_metrics

    def validate_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0
        metrics_sum = {}
        num_batches = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for features, targets, lengths in loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                lengths = lengths.to(self.device)

                # Generate mask
                max_len = features.size(1)
                batch_size = features.size(0)
                idx_range = (
                    torch.arange(max_len, device=self.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
                mask = idx_range < lengths.unsqueeze(1)

                outputs = self.model(features, mask=mask)
                loss, batch_metrics = self.criterion(outputs, targets, lengths)

                total_loss += loss.item()
                for k, v in batch_metrics.items():
                    metrics_sum[k] = metrics_sum.get(k, 0.0) + v
                num_batches += 1

                # Decode Predictions for LER
                # Stage 3 output is the last element of the tuple: (B, T, NumClasses)
                stage3_logits = outputs[2]
                stage3_probs = torch.softmax(stage3_logits, dim=2).cpu().numpy()

                # Targets to numpy
                targets_np = targets.cpu().numpy()
                lengths_np = lengths.cpu().numpy()

                for i in range(batch_size):
                    length = lengths_np[i]

                    # 1. Get valid sequence (remove padding)
                    # Probs: (T, C)
                    valid_probs = stage3_probs[i, :length, :]

                    # Decode predicted sequence
                    pred_seq = decode_predictions(valid_probs)
                    all_preds.append(pred_seq)

                    # 2. Get valid target sequence
                    # Target: (T,)
                    valid_target = targets_np[i, :length]

                    # Collapse repeats and remove background (0)
                    # groupby returns iterator, we explicitly filter 0
                    target_seq = [
                        k
                        for k, g in groupby(valid_target)
                        if k != Config.BACKGROUND_CLASS_IDX
                    ]
                    all_targets.append(target_seq)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}

        # Compute Levenshtein Error Rate
        ler = compute_levenshtein_score(all_preds, all_targets)

        return avg_loss, avg_metrics, ler

    def fit(self):
        set_seed(Config.SEED)

        # Load Data
        train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

        best_ler = float("inf")
        patience_counter = 0

        print("Starting training...")

        for epoch in range(1, Config.MAX_EPOCHS + 1):
            train_loss, train_metrics = self.train_epoch(train_loader)
            val_loss, val_metrics, val_ler = self.validate_epoch(val_loader)

            print(f"Epoch {epoch}/{Config.MAX_EPOCHS}")
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")
            print(f"Val LER: {val_ler}")
            # Print detailed metrics if needed, but keeping output clean as per instructions

            # Early Stopping Check
            if val_ler < best_ler:
                best_ler = val_ler
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved with LER: {best_ler}")
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Val LER: {best_ler}")
