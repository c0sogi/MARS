import os
import torch
import torch.optim as optim
import numpy as np
from library.config import Config, set_seed
from library.model import HCRGCN
from library.loss import CombinedLoss
from library.data_loader import get_dataloaders
from library.utils import (
    compute_levenshtein,
    median_filter_predictions,
    decode_predictions,
)


class Trainer:
    def __init__(self, device=None):
        """
        Trainer for the HCRGCN model handling training, validation, and prediction.
        """
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Initializing Trainer on device: {self.device}")

        # Set seed for reproducibility
        set_seed(Config.SEED)

        # Data Loaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders()

        # Model
        self.model = HCRGCN().to(self.device)

        # Loss Function
        self.criterion = CombinedLoss().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Paths
        self.checkpoint_path = Config.BEST_MODEL_PATH
        self.submission_path = Config.SUBMISSION_FILE

        # Tracking
        self.best_val_score = float("inf")

    def train_epoch(self):
        """
        Executes one epoch of training.
        Returns:
            float: Average training loss.
        """
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            # Unpack batch
            features, labels, boundaries, mask, lengths = batch

            # Move to device
            features = features.to(self.device)
            labels = labels.to(self.device)
            boundaries = boundaries.to(self.device)
            mask = mask.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(features, mask)

            # Compute loss (Deep Supervision)
            loss = self.criterion(predictions, labels, boundaries, mask)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        """
        Evaluates the model on the validation set.
        Returns:
            tuple: (Average Validation Loss, Levenshtein Error Rate)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_truths = []

        with torch.no_grad():
            for batch in self.val_loader:
                features, labels, boundaries, mask, lengths = batch

                features = features.to(self.device)
                labels = labels.to(self.device)
                boundaries = boundaries.to(self.device)
                mask = mask.to(self.device)

                # Forward pass
                predictions = self.model(features, mask)

                # Compute loss
                loss = self.criterion(predictions, labels, boundaries, mask)
                total_loss += loss.item()

                # --- Decoding for Metric Calculation ---
                # Use Stage 3 Class Probabilities for final decision
                s3_cls_probs, _ = predictions["stage3"]  # (B, T, C)

                # Get frame-wise class indices
                predicted_indices = (
                    torch.argmax(s3_cls_probs, dim=2).cpu().numpy()
                )  # (B, T)

                # Apply Median Filter to smooth predictions
                filtered_indices = median_filter_predictions(predicted_indices)

                # Convert tensors to numpy for decoding
                batch_labels = labels.cpu().numpy()
                batch_lengths = lengths.cpu().numpy()

                for i in range(len(features)):
                    length = batch_lengths[i]

                    # Extract valid sequence (remove padding)
                    pred_seq_indices = filtered_indices[i, :length]
                    true_seq_indices = batch_labels[i, :length]

                    # Decode to gesture IDs (collapse duplicates, remove background)
                    pred_gestures = decode_predictions(pred_seq_indices)
                    true_gestures = decode_predictions(true_seq_indices)

                    all_preds.append(pred_gestures)
                    all_truths.append(true_gestures)

        avg_loss = total_loss / len(self.val_loader)
        levenshtein_score = compute_levenshtein(all_preds, all_truths)

        return avg_loss, levenshtein_score

    def fit(self, epochs=None, early_stopping_patience=10):
        """
        Main training loop with checkpointing and early stopping.
        Args:
            epochs (int, optional): Number of epochs. Defaults to Config.NUM_EPOCHS.
            early_stopping_patience (int): Epochs to wait for improvement before stopping.
        """
        if epochs is None:
            epochs = Config.NUM_EPOCHS

        print(f"Starting training for {epochs} epochs...")

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_score = self.validate()

            # Print metrics with full precision
            print(
                f"Epoch {epoch}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Levenshtein: {val_score}"
            )

            # Checkpoint logic (Minimize Levenshtein Score)
            if val_score < self.best_val_score:
                print(
                    f"Validation score improved ({self.best_val_score} -> {val_score}). Saving model..."
                )
                self.best_val_score = val_score
                torch.save(self.model.state_dict(), self.checkpoint_path)
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= early_stopping_patience:
                print(
                    f"Early stopping triggered after {early_stopping_patience} epochs without improvement."
                )
                break

        print(f"Training complete. Best Validation Score: {self.best_val_score}")

    def predict(self):
        """
        Generates predictions for the test set using the best model and saves to CSV.
        """
        print("Generating predictions for test set...")

        # Load best model weights
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(
                torch.load(self.checkpoint_path, map_location=self.device)
            )
            print(f"Loaded best model from {self.checkpoint_path}")
        else:
            print("Warning: No checkpoint found. Using current model weights.")

        self.model.eval()
        results = []

        # Access sample IDs from the dataset (order is preserved in test loader)
        test_ids = self.test_loader.dataset.ids
        current_idx = 0

        with torch.no_grad():
            for batch in self.test_loader:
                features, labels, boundaries, mask, lengths = batch

                features = features.to(self.device)
                mask = mask.to(self.device)

                # Forward pass
                predictions = self.model(features, mask)

                # Use Stage 3 output
                s3_cls_probs, _ = predictions["stage3"]
                predicted_indices = torch.argmax(s3_cls_probs, dim=2).cpu().numpy()

                # Apply Median Filter
                filtered_indices = median_filter_predictions(predicted_indices)

                batch_lengths = lengths.cpu().numpy()

                for i in range(len(features)):
                    length = batch_lengths[i]

                    # Get sequence
                    pred_seq_indices = filtered_indices[i, :length]

                    # Decode to gesture IDs
                    pred_gestures = decode_predictions(pred_seq_indices)

                    # Format as required: "ID1,ID2,ID3"
                    pred_str = ",".join(map(str, pred_gestures))

                    # Retrieve Sample ID
                    sample_id = test_ids[current_idx]

                    # Store result line
                    results.append(f"{sample_id},{pred_str}")
                    current_idx += 1

        # Save to CSV
        with open(self.submission_path, "w") as f:
            for line in results:
                f.write(f"{line}\n")

        print(f"Submission saved to {self.submission_path}")
