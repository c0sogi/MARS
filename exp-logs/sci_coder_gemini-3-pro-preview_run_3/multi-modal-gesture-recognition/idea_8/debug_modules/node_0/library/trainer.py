import sys
import os
import time
import json
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from collections import defaultdict

# Add library path to access provided modules
sys.path.append(os.path.abspath("./library"))
from config import Config
from utils import set_seed, run_length_encoding, compute_levenshtein_score
from model import BAKC_IRN
from loss import MultiTaskCascadedLoss
from data_loader import get_dataloaders


class Trainer:
    """
    Manages the training, validation, and checkpointing of the BA-KC-IRN model.
    """

    def __init__(self):
        # 1. Setup Configuration and Device
        self.config = Config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(self.config.SEED)

        # Ensure directories exist
        self.config.setup_directories()

        print(f"Initializing Trainer on device: {self.device}")

        # 2. Initialize Data Loaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders()

        # 3. Initialize Model
        self.model = BAKC_IRN().to(self.device)

        # 4. Initialize Loss and Optimizer
        self.criterion = MultiTaskCascadedLoss().to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        # 5. Load Validation Ground Truth for Levenshtein Calculation
        self.val_ground_truth = self._load_ground_truth(self.config.VAL_METADATA_PATH)

    def _load_ground_truth(self, metadata_path):
        """
        Parses the metadata CSV to extract the sequence of gesture IDs for validation.
        Returns: dict {sample_id: [gesture_id, ...]}
        """
        df = pd.read_csv(metadata_path)
        ground_truth = {}
        for _, row in df.iterrows():
            sample_id = row["sample_id"]
            labels_json = row["labels"]
            if isinstance(labels_json, str):
                labels = json.loads(labels_json)
                # Extract sequence of IDs
                seq = [l["id"] for l in labels]
                ground_truth[sample_id] = seq
            else:
                ground_truth[sample_id] = []
        return ground_truth

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (features, cls_targets, bnd_targets, _, _) in enumerate(
            self.train_loader
        ):
            # Move data to device
            features = features.to(self.device)
            cls_targets = cls_targets.to(self.device)
            bnd_targets = bnd_targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            # outputs is a list of dicts from each stage
            outputs = self.model(features)

            # Compute Loss
            loss = self.criterion(outputs, cls_targets, bnd_targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self):
        """
        Runs validation on the validation set.
        Aggregates sliding window predictions to compute the Levenshtein score.
        """
        self.model.eval()

        # Dictionary to aggregate logits: sample_id -> (max_len, num_classes)
        # We use a dictionary of tensors or numpy arrays.
        # Since we don't know total length upfront easily, we'll store (start, end, logits) and reconstruct.
        sample_predictions = defaultdict(list)

        with torch.no_grad():
            for features, _, _, sample_ids, starts in self.val_loader:
                features = features.to(self.device)

                # Forward pass
                outputs = self.model(features)

                # We only care about the output of the final stage for inference
                final_output = outputs[-1]
                cls_logits = final_output["cls"]  # (Batch, NumClasses, Time)

                # Move to CPU
                cls_logits = cls_logits.cpu().numpy()
                starts = starts.numpy()

                batch_size = cls_logits.shape[0]

                for i in range(batch_size):
                    sid = sample_ids[i]
                    start_frame = starts[i]
                    logits = cls_logits[i]  # (NumClasses, Time)
                    # Transpose to (Time, NumClasses) for easier handling
                    logits = logits.transpose(1, 0)

                    sample_predictions[sid].append((start_frame, logits))

        # Reconstruct sequences and decode
        final_sequences = {}

        for sid, fragments in sample_predictions.items():
            # Find total length
            max_len = 0
            for start, logits in fragments:
                end = start + logits.shape[0]
                if end > max_len:
                    max_len = end

            # Aggregate logits (Summing overlapping windows)
            full_logits = np.zeros((max_len, self.config.NUM_CLASSES), dtype=np.float32)
            # Optional: Keep count for averaging, though sum is usually sufficient for argmax
            # counts = np.zeros((max_len, 1), dtype=np.float32)

            for start, logits in fragments:
                length = logits.shape[0]
                full_logits[start : start + length] += logits
                # counts[start : start + length] += 1

            # Argmax to get frame labels
            frame_preds = np.argmax(full_logits, axis=1)

            # Run Length Encoding to get gesture sequence
            sequence = run_length_encoding(frame_preds)
            final_sequences[sid] = sequence

        # Compute Metric
        score = compute_levenshtein_score(final_sequences, self.val_ground_truth)

        return score

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {self.config.NUM_EPOCHS} epochs...")

        best_score = float("inf")
        patience_counter = 0

        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_score = self.validate()

            duration = time.time() - start_time

            print(
                f"Epoch {epoch}/{self.config.NUM_EPOCHS} | "
                f"Time: {duration:.2f}s | "
                f"Train Loss: {train_loss} | "
                f"Val Levenshtein Score: {val_score}"
            )

            # Checkpoint & Early Stopping
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), self.config.BEST_MODEL_PATH)
                print(f"  >>> New Best Model Saved (Score: {best_score})")
            else:
                patience_counter += 1
                print(
                    f"  >>> No improvement. Patience: {patience_counter}/{self.config.PATIENCE}"
                )

            if patience_counter >= self.config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Score: {best_score}")

    def predict_test(self):
        """
        Generates predictions for the test set using the best model.
        Saves results to submission.csv.
        """
        print("Generating test predictions...")

        # Load best model
        if os.path.exists(self.config.BEST_MODEL_PATH):
            self.model.load_state_dict(
                torch.load(self.config.BEST_MODEL_PATH, map_location=self.device)
            )
            print("Loaded best model checkpoint.")
        else:
            print(
                "Warning: Best model checkpoint not found. Using current model weights."
            )

        self.model.eval()
        sample_predictions = defaultdict(list)

        with torch.no_grad():
            for features, _, _, sample_ids, starts in self.test_loader:
                features = features.to(self.device)
                outputs = self.model(features)
                final_output = outputs[-1]
                cls_logits = final_output["cls"].cpu().numpy()
                starts = starts.numpy()

                batch_size = cls_logits.shape[0]
                for i in range(batch_size):
                    sid = sample_ids[i]
                    start_frame = starts[i]
                    logits = cls_logits[i].transpose(1, 0)
                    sample_predictions[sid].append((start_frame, logits))

        # Reconstruct and Decode
        final_sequences = {}
        for sid, fragments in sample_predictions.items():
            max_len = 0
            for start, logits in fragments:
                end = start + logits.shape[0]
                if end > max_len:
                    max_len = end

            full_logits = np.zeros((max_len, self.config.NUM_CLASSES), dtype=np.float32)
            for start, logits in fragments:
                length = logits.shape[0]
                full_logits[start : start + length] += logits

            frame_preds = np.argmax(full_logits, axis=1)
            sequence = run_length_encoding(frame_preds)
            final_sequences[sid] = sequence

        # Save Submission
        from utils import save_submission

        save_submission(final_sequences, self.config.SUBMISSION_PATH)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")
