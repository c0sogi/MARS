import os
import time
import csv
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from typing import Optional

from library.config import Config
from library.utils import calculate_f1_score


class Trainer:
    """
    Manages the training, validation, and inference lifecycle for the Stack Exchange Tag Prediction model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[object] = None,
        tokenizer=None,
        device: str = Config.DEVICE,
    ):
        """
        Args:
            model: The PyTorch model to train.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            optimizer: Optimizer instance.
            scheduler: Learning rate scheduler (optional).
            tokenizer: TextProcessor instance (required for submission generation).
            device: Compute device ('cpu' or 'cuda').
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.device = device

        # Multi-label classification loss
        self.criterion = nn.BCEWithLogitsLoss()
        self.best_model_path = Config.BEST_MODEL_PATH

        # Ensure model is on the correct device
        self.model.to(self.device)

    def train_epoch(self) -> float:
        """
        Runs one epoch of training.
        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        n_batches = len(self.train_loader)

        for t_text, t_offsets, b_text, b_offsets, targets, _ in self.train_loader:
            t_text = t_text.to(self.device)
            t_offsets = t_offsets.to(self.device)
            b_text = b_text.to(self.device)
            b_offsets = b_offsets.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(t_text, t_offsets, b_text, b_offsets)
            loss = self.criterion(logits, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / n_batches if n_batches > 0 else 0.0

    def validate(self) -> tuple:
        """
        Evaluates the model on the validation set.
        Returns:
            tuple: (average_loss, f1_score)
        """
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        n_batches = len(self.val_loader)

        total_f1 = 0.0
        total_samples = 0

        with torch.no_grad():
            for t_text, t_offsets, b_text, b_offsets, targets, _ in self.val_loader:
                t_text = t_text.to(self.device)
                t_offsets = t_offsets.to(self.device)
                b_text = b_text.to(self.device)
                b_offsets = b_offsets.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(t_text, t_offsets, b_text, b_offsets)
                loss = self.criterion(logits, targets)
                running_loss += loss.item()

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)

                # Threshold probabilities to get binary predictions
                # Using Config.TAG_THRESHOLD defined in config
                preds = (probs > Config.TAG_THRESHOLD).float()

                # Calculate F1 for this batch to avoid storing full dataset in memory
                batch_f1 = calculate_f1_score(targets, preds, average="samples")
                batch_size = targets.size(0)

                total_f1 += batch_f1 * batch_size
                total_samples += batch_size

        avg_loss = running_loss / n_batches if n_batches > 0 else 0.0
        avg_f1 = total_f1 / total_samples if total_samples > 0 else 0.0

        return avg_loss, avg_f1

    def fit(self, epochs: int = Config.EPOCHS, patience: int = Config.PATIENCE):
        """
        Runs the full training loop with early stopping and model checkpointing.
        """
        best_f1 = -1.0
        patience_counter = 0

        print(f"Starting training on {self.device}...")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss = self.train_epoch()
            val_loss, val_f1 = self.validate()

            # Update scheduler if provided
            if self.scheduler:
                # Assuming ReduceLROnPlateau which needs a metric (val_f1 or val_loss)
                # We usually want to maximize F1
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_f1)
                else:
                    self.scheduler.step()

            elapsed = time.time() - start_time

            # Print metrics with full precision
            print(
                f"Epoch {epoch} | Time: {elapsed}s | Train Loss: {train_loss} | Val Loss: {val_loss} | Val F1: {val_f1}"
            )

            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved to {self.best_model_path}")
            else:
                patience_counter += 1

            # Early stopping check
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break

        print(f"Training complete. Best Validation F1: {best_f1}")

    def generate_submission(
        self, test_loader: DataLoader, output_path: str = Config.SUBMISSION_PATH
    ):
        """
        Generates predictions for the test set and saves them to a CSV file.
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for submission generation.")

        # Load the best model weights
        if os.path.exists(self.best_model_path):
            print(f"Loading best model from {self.best_model_path}...")
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
        else:
            print("Warning: Best model file not found. Using current model weights.")

        self.model.eval()
        results = []

        print("Generating predictions for test set...")
        with torch.no_grad():
            for t_text, t_offsets, b_text, b_offsets, _, ids in test_loader:
                t_text = t_text.to(self.device)
                t_offsets = t_offsets.to(self.device)
                b_text = b_text.to(self.device)
                b_offsets = b_offsets.to(self.device)

                logits = self.model(t_text, t_offsets, b_text, b_offsets)
                probs = torch.sigmoid(logits)

                # Decode tags using the tokenizer
                # decode_tags returns a list of space-separated strings
                # It handles thresholding internally if probs are passed
                predicted_tags = self.tokenizer.decode_tags(
                    probs, threshold=Config.TAG_THRESHOLD
                )

                # Collect results
                batch_ids = ids.cpu().numpy()
                for q_id, tags in zip(batch_ids, predicted_tags):
                    results.append({"Id": int(q_id), "Tags": tags})

        # Create DataFrame
        df_submission = pd.DataFrame(results)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        # quoting=csv.QUOTE_NONNUMERIC ensures that non-numeric fields (Tags) are quoted,
        # while numeric fields (Id) are not, matching the submission format: 1,"tag1 tag2"
        df_submission.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        print(f"Submission saved to {output_path}")
