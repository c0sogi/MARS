import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.model import PAM_CN
from library.utils import (
    compute_levenshtein_distance,
    decode_predictions,
    LogSpaceSmoothingLoss,
)
from library.data_loader import get_dataloaders


class Trainer:
    """
    Manages the training, validation, and model selection for the PAM-CN model.
    Implements Cascaded Deep Supervision and Full-Sequence Validation.
    """

    def __init__(self):
        # Setup Device
        self.device = Config.DEVICE

        # Initialize Model
        self.model = PAM_CN().to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Functions
        # 1. Weighted Cross Entropy (for classification)
        # We use ignore_index if necessary, but here we have a background class (0)
        # which is weighted down in Config.CLASS_WEIGHTS.
        weight_tensor = Config.CLASS_WEIGHTS.to(self.device)
        self.ce_criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        # 2. Log-Space Smoothing (for temporal consistency)
        self.smooth_criterion = LogSpaceSmoothingLoss(
            smoothing_lambda=Config.SMOOTHING_LAMBDA,
            threshold=Config.SMOOTHING_THRESHOLD,
        ).to(self.device)

        # Checkpoints
        self.checkpoint_dir = Config.CHECKPOINT_DIR
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")

        # Metrics
        self.best_val_score = float("inf")  # Levenshtein Error Rate (lower is better)

    def calculate_loss(self, logits_tuple, targets):
        """
        Computes the Cascaded Loss:
        L_total = CE(P1) + CE(P2) + CE(P3) + Smooth(P2) + Smooth(P3)
        """
        logits1, logits2, logits3 = logits_tuple

        # Reshape for CrossEntropyLoss: (Batch*Time, Classes) vs (Batch*Time)
        # Or (Batch, Classes, Time) vs (Batch, Time).
        # PyTorch CE expects (N, C, ...) or (N, C)
        # Our logits are (Batch, Time, Classes).
        # We permute to (Batch, Classes, Time) for CE Loss

        l1_perm = logits1.permute(0, 2, 1)
        l2_perm = logits2.permute(0, 2, 1)
        l3_perm = logits3.permute(0, 2, 1)

        # Classification Losses (Deep Supervision)
        loss_ce1 = self.ce_criterion(l1_perm, targets)
        loss_ce2 = self.ce_criterion(l2_perm, targets)
        loss_ce3 = self.ce_criterion(l3_perm, targets)

        # Smoothing Losses (Only for refinement stages 2 and 3)
        # Input to smoothing loss should be log_probs of shape (Batch, Time, Classes)
        log_probs2 = F.log_softmax(logits2, dim=2)
        log_probs3 = F.log_softmax(logits3, dim=2)

        loss_smooth2 = self.smooth_criterion(log_probs2)
        loss_smooth3 = self.smooth_criterion(log_probs3)

        total_loss = loss_ce1 + loss_ce2 + loss_ce3 + loss_smooth2 + loss_smooth3

        return total_loss, (loss_ce1.item(), loss_ce2.item(), loss_ce3.item())

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        # For logging individual components if needed, but we focus on total loss

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(self.device)
            batch_labels = batch_labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits_tuple = self.model(batch_features)

            # Loss calculation
            loss, _ = self.calculate_loss(logits_tuple, batch_labels)

            # Backward
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_loader)

    def validate(self, val_loader):
        """
        Performs Full Sequence Inference on the validation set.
        Calculates the Levenshtein Error Rate.
        """
        self.model.eval()

        total_dist = 0
        total_gestures = 0

        with torch.no_grad():
            for features, labels, sample_id in val_loader:
                features = features.to(self.device)
                # labels are (1, T)
                labels_np = labels.squeeze(0).numpy()

                # Forward pass
                _, _, logits3 = self.model(features)

                # Decode Predictions (Stage 3 is final output)
                # logits3: (1, Time, Classes)
                probs3 = F.softmax(logits3, dim=2)
                preds_frame = torch.argmax(probs3, dim=2).squeeze(0).cpu().numpy()

                # Convert frame predictions to gesture list
                predicted_gestures = decode_predictions(preds_frame)

                # Convert ground truth frames to gesture list
                # We use the same decoding logic to ensure consistency (RLE + filtering)
                # This treats the frame-wise annotation as the source of truth.
                true_gestures = decode_predictions(labels_np)

                # Compute Metric
                dist = compute_levenshtein_distance(predicted_gestures, true_gestures)

                total_dist += dist
                total_gestures += len(true_gestures)

        # Avoid division by zero
        if total_gestures == 0:
            return 0.0

        error_rate = total_dist / total_gestures
        return error_rate

    def fit(self, epochs=Config.NUM_EPOCHS, patience=Config.EARLY_STOPPING_PATIENCE):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")

        # Get DataLoaders
        train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_score = self.validate(val_loader)

            print(
                f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Error Rate: {val_score}"
            )

            # Model Selection
            if val_score < self.best_val_score:
                self.best_val_score = val_score
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  -> New best model saved! Score: {val_score}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        print(f"Training complete. Best Validation Score: {self.best_val_score}")

    def load_best_model(self):
        """Loads the best saved checkpoint."""
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
            print("Best model loaded.")
        else:
            print("Warning: No best model checkpoint found.")

    def predict_test_set(self):
        """
        Generates predictions for the test set using the best model.
        Saves to submission.csv.
        """
        self.load_best_model()
        self.model.eval()

        _, _, test_loader = get_dataloaders(load_cached_data=True)

        results = []

        print("Generating predictions for test set...")
        with torch.no_grad():
            for features, _, sample_ids in test_loader:
                features = features.to(self.device)
                sample_id = sample_ids[0]  # batch size is 1

                # Forward
                _, _, logits3 = self.model(features)

                # Decode
                probs3 = F.softmax(logits3, dim=2)
                preds_frame = torch.argmax(probs3, dim=2).squeeze(0).cpu().numpy()

                predicted_gestures = decode_predictions(preds_frame)

                # Format: Id,Sequence
                # Sanitize ID: Sample00300 -> 300
                try:
                    sid_int = int(sample_id.replace("Sample", ""))
                except ValueError:
                    sid_int = sample_id

                # Sequence: Space-separated string (e.g., "2 12 3")
                pred_str = " ".join(map(str, predicted_gestures))

                results.append(f"{sid_int},{pred_str}")

        # Save to submission file
        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        with open(submission_path, "w") as f:
            f.write("Id,Sequence\n")
            for line in results:
                f.write(line + "\n")

        print(f"Submission saved to {submission_path}")
