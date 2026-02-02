import os
import time
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import json

from library.config import Config
from library.utils import (
    set_seeds,
    compute_levenshtein_score,
    decode_predictions_to_sequence,
    run_length_encoding,
    filter_segments,
)
from library.data_loader import get_dataloaders
from library.model import DGC_KN
from library.loss import CascadedLoss


class Trainer:
    """
    Manages the training lifecycle of the DGC-KN model.
    """

    def __init__(self, debug=False, epochs=None):
        self.device = Config.DEVICE
        self.debug = debug

        # Override config if arguments provided
        if epochs is not None:
            Config.NUM_EPOCHS = epochs
        if debug:
            Config.set_debug_mode()

        # Initialize Data Loaders
        print("Initializing Data Loaders...")
        self.train_loader, self.val_loader, _ = get_dataloaders(debug=self.debug)

        # Initialize Model
        print("Initializing Model...")
        self.model = DGC_KN().to(self.device)

        # Initialize Loss
        self.criterion = CascadedLoss().to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Early Stopping Variables
        self.best_score = float("inf")
        self.patience_counter = 0
        self.early_stopping_patience = Config.EARLY_STOPPING_PATIENCE

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        metrics_accum = {}

        start_time = time.time()

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(features)

            # Compute loss
            loss, batch_metrics = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate stats
            running_loss += loss.item()
            for k, v in batch_metrics.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + v

        avg_loss = running_loss / len(self.train_loader)
        avg_metrics = {k: v / len(self.train_loader) for k, v in metrics_accum.items()}

        duration = time.time() - start_time

        print(
            f"Epoch {epoch_idx+1}/{Config.NUM_EPOCHS} [Train] "
            f"Loss: {avg_loss:.6f} | Time: {duration:.2f}s"
        )

        return avg_loss

    def validate(self):
        """
        Runs validation on the full sequences and computes Levenshtein score.
        """
        self.model.eval()

        predictions = {}
        ground_truths = {}

        with torch.no_grad():
            for features, dense_labels, sample_ids in self.val_loader:
                features = features.to(self.device)
                # dense_labels is (Batch, Time), Batch=1
                # sample_ids is tuple of size 1

                sample_id = sample_ids[0]

                # Forward pass
                outputs = self.model(features)

                # Get final stage probabilities (Stage 3)
                probs = outputs["probs_3"]  # (1, Time, Classes)
                probs = probs.squeeze(0).cpu().numpy()  # (Time, Classes)

                # Decode prediction to sequence of gesture IDs
                pred_seq = decode_predictions_to_sequence(probs)
                predictions[sample_id] = pred_seq

                # Decode ground truth dense labels to sequence of gesture IDs
                # dense_labels[0] is (Time,) tensor
                gt_dense = dense_labels[0].numpy()

                # We use the same logic: RLE -> Filter Background -> Filter Short (optional, but GT shouldn't have short/noise ideally)
                # However, for GT, we trust the dense labels derived from metadata.
                # Just RLE and remove background (0).
                gt_segments = run_length_encoding(gt_dense)

                # Note: We don't filter GT by duration, only by background class
                gt_seq = []
                for cls_id, _, _ in gt_segments:
                    if cls_id != Config.BACKGROUND_CLASS_ID:
                        gt_seq.append(int(cls_id))

                ground_truths[sample_id] = gt_seq

        # Compute Levenshtein Score (Error Rate)
        score = compute_levenshtein_score(predictions, ground_truths)

        print(f"Epoch Validation Score (Levenshtein Error): {score}")
        return score

    def fit(self):
        """
        Main training loop.
        """
        print(f"Starting training on {self.device}...")
        set_seeds()

        for epoch in range(Config.NUM_EPOCHS):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_score = self.validate()

            # Checkpoint & Early Stopping
            if val_score < self.best_score:
                print(
                    f"Score improved from {self.best_score} to {val_score}. Saving model..."
                )
                self.best_score = val_score
                self.patience_counter = 0
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
            else:
                self.patience_counter += 1
                print(
                    f"Score did not improve. Patience: {self.patience_counter}/{self.early_stopping_patience}"
                )

            if self.patience_counter >= self.early_stopping_patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Score: {self.best_score}")


def train_model(debug=False, epochs=None):
    """
    Wrapper function to instantiate Trainer and run fit.
    """
    trainer = Trainer(debug=debug, epochs=epochs)
    trainer.fit()
