import os
import time
import torch
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import (
    set_seed,
    compute_normalized_levenshtein,
    smooth_predictions,
    decode_predictions,
)
from library.data import load_data, GestureDataset, collate_fn
from library.model import SG_CRCN
from library.loss import TotalLoss


class Trainer:
    """
    Trainer class for the SG-CRCN model.
    Handles the training loop, validation, metric calculation, and checkpointing.
    """

    def __init__(self, device=None):
        """
        Initialize the Trainer.

        Args:
            device (torch.device, optional): Device to run training on.
                                             Defaults to CUDA if available.
        """
        # Set reproducibility
        set_seed(Config.SEED)

        # Device configuration
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Using device: {self.device}")

        # Initialize Model
        self.model = SG_CRCN().to(self.device)

        # Initialize Loss
        self.criterion = TotalLoss().to(self.device)

        # Initialize Optimizer
        # Using AdamW as specified in the idea
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Training State
        self.start_epoch = 0
        self.best_metric = float("inf")  # Lower Levenshtein distance is better

    def load_datasets(self, batch_size=None):
        """
        Loads training and validation datasets using the library functions.

        Args:
            batch_size (int, optional): Batch size. Defaults to Config.BATCH_SIZE.
        """
        bs = batch_size if batch_size is not None else Config.BATCH_SIZE

        # Load raw data dictionaries (cached if available)
        train_data_dict = load_data(mode="train", load_cached_data=True)
        val_data_dict = load_data(mode="val", load_cached_data=True)

        # Create Datasets
        # Enable augmentation for training
        self.train_dataset = GestureDataset(train_data_dict, augment=True)
        # Disable augmentation for validation
        self.val_dataset = GestureDataset(val_data_dict, augment=False)

        # Create DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=bs,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        metrics_accum = {}

        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)
            boundaries = batch["boundaries"].to(self.device)
            mask = batch["mask"].to(self.device)

            batch_targets = {"labels": labels, "boundaries": boundaries, "mask": mask}

            # Forward pass
            outputs = self.model(features, mask)

            # Compute Loss (Deep Supervision)
            loss, batch_metrics = self.criterion(outputs, batch_targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.GRADIENT_CLIP
            )

            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            for k, v in batch_metrics.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + v

        # Average metrics
        avg_loss = total_loss / len(self.train_loader)
        avg_sub_metrics = {
            k: v / len(self.train_loader) for k, v in metrics_accum.items()
        }

        return avg_loss, avg_sub_metrics

    def validate(self):
        """
        Runs validation loop.
        Computes Loss and Normalized Levenshtein Distance.
        """
        self.model.eval()
        total_loss = 0.0

        all_preds_seq = []
        all_truth_seq = []

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["features"].to(self.device)
                labels = batch["labels"].to(self.device)
                boundaries = batch["boundaries"].to(self.device)
                mask = batch["mask"].to(self.device)
                lengths = batch["lengths"]

                batch_targets = {
                    "labels": labels,
                    "boundaries": boundaries,
                    "mask": mask,
                }

                # Forward pass
                outputs = self.model(features, mask)

                # Compute Loss
                loss, _ = self.criterion(outputs, batch_targets)
                total_loss += loss.item()

                # --- Decode Predictions for Levenshtein Metric ---
                # Use Stage 3 Class Probabilities
                s3_probs = outputs["stage3_cls"]  # (B, T, C)

                # Get frame-wise predictions
                frame_preds = torch.argmax(s3_probs, dim=2).cpu().numpy()  # (B, T)

                # Get ground truth frame labels
                frame_targets = labels.cpu().numpy()  # (B, T)

                # Process each sample in the batch
                for i in range(len(features)):
                    length = lengths[i].item()

                    # 1. Extract valid frames based on length
                    raw_pred = frame_preds[i, :length]
                    raw_target = frame_targets[i, :length]

                    # 2. Smooth predictions (Median Filter)
                    smoothed_pred = smooth_predictions(
                        raw_pred, window_size=Config.MEDIAN_WINDOW
                    )

                    # 3. Decode to sequence (Collapse & Remove Background)
                    decoded_pred = decode_predictions(smoothed_pred, background_class=0)
                    decoded_target = decode_predictions(raw_target, background_class=0)

                    all_preds_seq.append(decoded_pred)
                    all_truth_seq.append(decoded_target)

        avg_loss = total_loss / len(self.val_loader)

        # Compute Metric
        levenshtein_score = compute_normalized_levenshtein(all_preds_seq, all_truth_seq)

        return avg_loss, levenshtein_score

    def fit(self, num_epochs=None):
        """
        Main training loop with Early Stopping.
        """
        epochs = num_epochs if num_epochs is not None else Config.NUM_EPOCHS
        print(f"Starting training for {epochs} epochs...")

        patience_counter = 0

        for epoch in range(self.start_epoch, epochs):
            start_time = time.time()

            # Train
            train_loss, train_metrics = self.train_epoch(epoch)

            # Validate
            val_loss, val_score = self.validate()

            duration = time.time() - start_time

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch+1}/{epochs} [{duration:.1f}s] "
                f"Train Loss: {train_loss:.6f} "
                f"Val Loss: {val_loss:.6f} "
                f"Val Levenshtein: {val_score:.10f}"
            )

            # Checkpointing & Early Stopping
            if val_score < self.best_metric:
                print(
                    f"Validation score improved ({self.best_metric:.6f} --> {val_score:.6f}). Saving model..."
                )
                self.best_metric = val_score
                torch.save(self.model.state_dict(), Config.MODEL_CHECKPOINT)
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break

        print(f"Training complete. Best Validation Score: {self.best_metric:.10f}")


def run_training():
    """
    Helper function to instantiate Trainer and run fit.
    """
    trainer = Trainer()
    trainer.load_datasets()
    trainer.fit()
