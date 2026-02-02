import torch
import torch.optim as optim
import numpy as np
import os
from typing import List, Tuple, Optional

from library.config import Config
from library.model import MSTCN
from library.loss import ActionSegmentationLoss
from library.dataset import create_dataloaders
from library.utils import (
    save_checkpoint,
    load_checkpoint,
    seed_everything,
    write_submission_file,
)


class Trainer:
    """
    Trainer class for the MS-TCN Gesture Recognition model.
    Handles training, validation, early stopping, and prediction generation.
    """

    def __init__(self, config=Config):
        self.config = config
        self.device = config.DEVICE

        # Ensure reproducibility
        seed_everything(config.RANDOM_SEED)

        # Initialize Model
        self.model = MSTCN().to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

        # Initialize Loss Function
        self.criterion = ActionSegmentationLoss(ignore_index=-100, mse_threshold=4.0)

    def train_epoch(self, loader) -> float:
        """
        Runs one epoch of training.
        Returns average loss.
        """
        self.model.train()
        total_loss = 0.0
        count = 0

        for features, labels, mask, lengths, ids in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass: returns list of outputs (one per stage)
            outputs = self.model(features, mask)

            # Calculate loss (sum of CE + Smoothness for all stages)
            loss = self.criterion(outputs, labels, mask)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.GRADIENT_CLIP
            )

            self.optimizer.step()

            total_loss += loss.item()
            count += 1

        return total_loss / count if count > 0 else 0.0

    def validate(self, loader) -> float:
        """
        Runs validation.
        Returns average loss.
        """
        self.model.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for features, labels, mask, lengths, ids in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)

                outputs = self.model(features, mask)
                loss = self.criterion(outputs, labels, mask)

                total_loss += loss.item()
                count += 1

        return total_loss / count if count > 0 else 0.0

    def fit(
        self, debug_subset_size: Optional[int] = None, epochs: Optional[int] = None
    ):
        """
        Main training loop with Early Stopping.
        """
        # Create DataLoaders
        train_loader, val_loader, _ = create_dataloaders(
            batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS,
            debug_subset_size=debug_subset_size,
        )

        num_epochs = epochs if epochs is not None else self.config.NUM_EPOCHS
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training on device: {self.device}")
        print(f"Epochs: {num_epochs}, Batch Size: {self.config.BATCH_SIZE}")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            print(f"Epoch {epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}")

            # Early Stopping and Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    self.config.MODEL_SAVE_PATH,
                )
            else:
                patience_counter += 1

            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(
                    f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_val_loss}"
                )
                break

        print("Training complete.")

    def _decode_predictions(self, frame_preds: np.ndarray) -> List[int]:
        """
        Decodes frame-wise predictions into a list of gesture IDs.
        Applies Run-Length Encoding, removes background class (0),
        and filters out segments shorter than 5 frames.
        """
        if len(frame_preds) == 0:
            return []

        # 1. Run-Length Encoding
        segments = []
        current_label = frame_preds[0]
        current_len = 1

        for label in frame_preds[1:]:
            if label == current_label:
                current_len += 1
            else:
                segments.append((current_label, current_len))
                current_label = label
                current_len = 1
        segments.append((current_label, current_len))

        # 2. Filter Background and Short Segments
        final_gestures = []
        for label, length in segments:
            # Remove background class (0)
            if label != self.config.BACKGROUND_CLASS_ID:
                # Remove extremely short segments (< 5 frames)
                if length >= 5:
                    final_gestures.append(int(label))

        return final_gestures

    def predict(self, debug_subset_size: Optional[int] = None):
        """
        Generates predictions for the test set using the best saved model.
        Writes the result to the submission file.
        """
        # Load the best model
        try:
            self.model, _, _, _ = load_checkpoint(
                self.model, None, self.config.MODEL_SAVE_PATH
            )
            print(f"Loaded best model from {self.config.MODEL_SAVE_PATH}")
        except FileNotFoundError:
            print("Warning: No checkpoint found. Predicting with current model state.")

        self.model.eval()

        # Get Test Loader
        _, _, test_loader = create_dataloaders(
            batch_size=self.config.BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS,
            debug_subset_size=debug_subset_size,
        )

        all_predictions = []

        print("Generating predictions...")
        with torch.no_grad():
            for features, _, mask, lengths, ids in test_loader:
                features = features.to(self.device)
                mask = mask.to(self.device)

                # Forward pass
                outputs = self.model(features, mask)

                # Use the output of the final refinement stage
                final_stage_logits = outputs[-1]  # (Batch, Classes, Time)

                # Get predicted class indices
                pred_classes = torch.argmax(final_stage_logits, dim=1)  # (Batch, Time)

                # Process batch
                for i in range(len(ids)):
                    sample_id = ids[i]
                    length = lengths[i]

                    # Extract valid frames (ignore padding)
                    valid_preds = pred_classes[i, :length].cpu().numpy()

                    # Decode to gesture list
                    decoded_gestures = self._decode_predictions(valid_preds)

                    all_predictions.append((sample_id, decoded_gestures))

        # Write submission file
        write_submission_file(all_predictions, self.config.SUBMISSION_PATH)
        print(f"Submission file generated at: {self.config.SUBMISSION_PATH}")
