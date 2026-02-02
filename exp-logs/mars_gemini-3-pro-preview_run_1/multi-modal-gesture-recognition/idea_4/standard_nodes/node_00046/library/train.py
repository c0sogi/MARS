import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.utils import set_seed, compute_levenshtein, decode_predictions
from library.data_loader import GestureDataset, CollateFn
from library.model import CGR_GRU


class Trainer:
    """
    Trainer class for the Context-Gated Residual GRU model.
    Handles training, validation, checkpointing, and early stopping.
    """

    def __init__(self):
        self.device = Config.get_device()
        set_seed(Config.SEED)

        # 1. Data Loaders
        print("Initializing DataLoaders...")
        self.train_dataset = GestureDataset(split="train", debug=False)
        self.val_dataset = GestureDataset(split="val", debug=False)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            collate_fn=CollateFn(mode="train"),
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=CollateFn(mode="val"),
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False,
        )

        # 2. Model
        print("Initializing Model...")
        self.model = CGR_GRU().to(self.device)

        # 3. Loss Function
        # Class weights: 0.5 for background (0), 1.0 for gestures (1-20)
        weights = torch.ones(Config.NUM_CLASSES)
        weights[0] = Config.BG_CLASS_WEIGHT
        self.criterion = nn.CrossEntropyLoss(
            weight=weights.to(self.device), label_smoothing=Config.LABEL_SMOOTHING
        )

        # 4. Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )  # Cite solution_lesson_node_00040
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.NUM_EPOCHS
        )  # Cite solution_lesson_node_00040

    def _get_gt_sequence(self, tensor_labels):
        """
        Extracts sequence of gesture IDs from frame-wise label tensor.
        Collapses duplicates and removes background (0).
        Args:
            tensor_labels: (T,) tensor
        Returns:
            List[int]
        """
        seq = []
        prev = -1
        for x in tensor_labels:
            val = x.item()
            if val != prev:
                if val != 0:
                    seq.append(int(val))
                prev = val
        return seq

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            if batch is None:
                continue

            # Move to device
            skeleton = batch["skeleton"].to(self.device)
            audio = batch["audio"].to(self.device)
            labels = batch["labels"].to(self.device)
            lengths = batch[
                "lengths"
            ]  # CPU tensor usually fine for pack_padded, but model handles it

            # Forward
            self.optimizer.zero_grad()
            logits = self.model(skeleton, audio, lengths=lengths)  # (B, T, C)

            # Flatten for CrossEntropy: (B*T, C) vs (B*T)
            # We must mask padding if necessary, but CE loss ignore_index defaults to -100.
            # However, our padding value in loader is 0 (Background).
            # The model predicts for padded frames too.
            # Ideally we should mask loss for padded regions, but since padding is 0 and 0 is background,
            # the model learns to predict background for padding, which is acceptable.

            logits_flat = logits.view(-1, Config.NUM_CLASSES)
            labels_flat = labels.view(-1)

            loss = self.criterion(logits_flat, labels_flat)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_dist = 0.0
        total_gt_gestures = 0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                if batch is None:
                    continue

                skeleton = batch["skeleton"].to(self.device)
                audio = batch["audio"].to(self.device)
                labels = batch["labels"].to(self.device)
                lengths = batch["lengths"]

                # Forward
                logits = self.model(skeleton, audio, lengths=lengths)

                # Loss
                logits_flat = logits.view(-1, Config.NUM_CLASSES)
                labels_flat = labels.view(-1)
                loss = self.criterion(logits_flat, labels_flat)
                total_loss += loss.item()

                # Decode & Metric
                probs = torch.softmax(logits, dim=2)

                # Iterate over batch
                for i in range(logits.size(0)):
                    # Get sequence length to ignore padding in prediction
                    seq_len = lengths[i]

                    # Slice valid probabilities
                    valid_probs = probs[i, :seq_len, :]

                    # Decode
                    pred_seq = decode_predictions(valid_probs)

                    # Get GT sequence
                    gt_seq = self._get_gt_sequence(labels[i, :seq_len])

                    # Compute Levenshtein
                    dist = compute_levenshtein(pred_seq, gt_seq)

                    total_dist += dist
                    total_gt_gestures += len(gt_seq)

                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        # Avoid division by zero
        ler = total_dist / total_gt_gestures if total_gt_gestures > 0 else 1.0

        return avg_loss, ler

    def fit(self):
        print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
        best_ler = float("inf")
        patience_counter = 0

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_ler = self.validate()

            # Scheduler step (CosineAnnealing steps per epoch without metric)
            self.scheduler.step()

            print(
                f"Epoch {epoch}/{Config.NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val LER: {val_ler:.6f}"
            )

            # Checkpoint & Early Stopping
            if val_ler < best_ler:
                best_ler = val_ler
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.CHECKPOINT_PATH)
                print(f"New best model saved with LER: {best_ler:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print(
                        f"Early stopping triggered after {patience_counter} epochs without improvement."
                    )
                    break

        print(f"Training complete. Best LER: {best_ler:.6f}")
