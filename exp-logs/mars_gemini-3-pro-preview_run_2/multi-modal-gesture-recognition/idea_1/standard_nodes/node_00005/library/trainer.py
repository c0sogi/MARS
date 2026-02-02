import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config, set_seed
from library.utils import compute_levenshtein, decode_predictions


class Trainer:
    """
    Trainer class for the Bi-LSTM Gesture Recognition model.
    Handles training loops, validation, metric calculation, and checkpointing.
    """

    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.best_val_score = float("inf")  # Levenshtein distance (lower is better)

    def _process_batch_loss(self, logits, labels, lengths):
        """
        Computes the masked cross-entropy loss, ignoring padded frames.

        Args:
            logits: (Batch, Time, Classes)
            labels: (Batch, Time)
            lengths: (Batch,)

        Returns:
            torch.Tensor: Scalar loss
        """
        # Create a boolean mask for valid frames [Batch, Time]
        max_len = logits.size(1)
        mask = torch.arange(max_len, device=self.device)[None, :] < lengths[:, None]

        # Flatten and mask
        active_logits = logits[mask]
        active_labels = labels[mask]

        loss = self.criterion(active_logits, active_labels)
        return loss

    def _get_truth_sequences(self, labels, lengths):
        """
        Converts frame-wise label tensors into list of gesture sequences
        for Levenshtein comparison.

        Logic:
        1. Slice by length.
        2. Collapse repeats.
        3. Remove background (0).
        """
        labels_np = labels.cpu().numpy()
        lengths_np = lengths.cpu().numpy()
        truth_seqs = []

        for i in range(len(labels_np)):
            l = lengths_np[i]
            seq = labels_np[i, :l]

            # Collapse repeats and remove background (0)
            if len(seq) == 0:
                truth_seqs.append([])
                continue

            collapsed = [seq[0]]
            for k in range(1, len(seq)):
                if seq[k] != seq[k - 1]:
                    collapsed.append(seq[k])

            final_seq = [int(x) for x in collapsed if x != 0]
            truth_seqs.append(final_seq)

        return truth_seqs

    def train_epoch(self, epoch_idx):
        self.model.train()
        running_loss = 0.0
        total_batches = 0

        for batch in self.train_loader:
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)
            lengths = batch["lengths"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(features, lengths)

            # Compute Loss
            loss = self._process_batch_loss(logits, labels, lengths)

            # Backward pass
            loss.backward()

            # Gradient clipping (optional but recommended for LSTM)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        avg_loss = running_loss / total_batches if total_batches > 0 else 0.0
        return avg_loss

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        total_batches = 0

        all_preds = []
        all_truth = []

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["features"].to(self.device)
                labels = batch["labels"].to(self.device)
                lengths = batch["lengths"].to(self.device)

                # Forward pass
                logits = self.model(features, lengths)

                # Compute Loss
                loss = self._process_batch_loss(logits, labels, lengths)
                running_loss += loss.item()
                total_batches += 1

                # Decode Predictions
                batch_preds = decode_predictions(logits)

                # Decode Ground Truth
                batch_truth = self._get_truth_sequences(labels, lengths)

                all_preds.extend(batch_preds)
                all_truth.extend(batch_truth)

        avg_loss = running_loss / total_batches if total_batches > 0 else 0.0

        # Calculate Levenshtein Error Rate
        lev_score = compute_levenshtein(all_preds, all_truth)

        return avg_loss, lev_score

    def fit(self, num_epochs, patience):
        print(f"Starting training on device: {self.device}")

        # Initialize Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )

        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_score = self.validate()

            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Train Loss: {train_loss:.10f} | "
                f"Val Loss: {val_loss:.10f} | "
                f"Val Levenshtein: {val_score:.10f}"
            )

            # Step Scheduler based on validation loss
            scheduler.step(val_loss)

            # Checkpoint
            if val_score < self.best_val_score:
                self.best_val_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_CHECKPOINT)
                print(f"New best model saved with score: {val_score:.10f}")
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        print(f"Training complete. Best Validation Score: {self.best_val_score:.10f}")


def run_training(model, train_loader, val_loader):
    """
    Helper function to instantiate Trainer and run the training loop.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    model = model.to(device)

    # Define Weighted Cross Entropy Loss
    # Weights: Lower weight for background (0), higher for gestures (1-20)
    weights = torch.tensor(Config.CLASS_WEIGHTS, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    trainer.fit(num_epochs=Config.NUM_EPOCHS, patience=Config.PATIENCE)

    return trainer
