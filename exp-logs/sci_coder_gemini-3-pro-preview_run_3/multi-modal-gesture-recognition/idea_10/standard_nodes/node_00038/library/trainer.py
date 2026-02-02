import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import nltk
import random

from library.config import Config
from library.utils import (
    TruncatedMSELoss,
    rle_decode,
    save_submission,
    compute_kinematics,
)
from library.dataset import GestureDataset
from library.model import VIARN


class Trainer:
    """
    Trainer class for the View-Invariant Attentive Refinement Network (VI-ARN).
    Handles training, validation, early stopping, and inference.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        self._set_seed(Config.SEED)

        # Initialize Model
        self.model = VIARN().to(self.device)

        # Loss Functions
        self.cls_criterion = nn.CrossEntropyLoss(weight=Config.get_class_weights()).to(
            self.device
        )

        self.smooth_criterion = TruncatedMSELoss(threshold=1.0).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _compute_loss(self, outputs, targets):
        """
        Computes the cascaded loss for Deep Supervision.
        Loss = Sum over stages (Classification Loss + Lambda * Smoothing Loss)
        """
        total_loss = 0.0

        # outputs is a tuple (out1, out2, out3)
        for stage_out in outputs:
            # Classification Loss
            # stage_out: (Batch, Classes, Time)
            # targets: (Batch, Time)
            total_loss += self.cls_criterion(stage_out, targets)

            # Smoothing Loss
            # Apply to Log Softmax probabilities
            log_probs = F.log_softmax(stage_out, dim=1)
            smooth_loss = self.smooth_criterion(log_probs)

            total_loss += Config.LAMBDA_SMOOTH * smooth_loss

        return total_loss

    def train(self, num_epochs=None):
        """
        Main training loop with validation and early stopping.
        """
        if num_epochs is None:
            num_epochs = Config.NUM_EPOCHS

        # Load Datasets
        train_dataset = GestureDataset(split="train", load_cached_data=True)
        val_dataset = GestureDataset(split="val", load_cached_data=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        best_val_loss = float("inf")
        patience = 10
        patience_counter = 0

        print(f"Starting training on device: {self.device}")

        for epoch in range(1, num_epochs + 1):
            # --- Training Step ---
            self.model.train()
            train_loss = 0.0

            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass (returns tuple of outputs from 3 stages)
                outputs = self.model(features)

                loss = self._compute_loss(outputs, labels)
                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), Config.GRADIENT_CLIP
                )

                self.optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # --- Validation Step ---
            val_loss, val_edit_dist = self.evaluate(val_loader)

            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Edit Dist: {val_edit_dist:.6f}"
            )

            # --- Early Stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                # print(f"  New best model saved to {Config.MODEL_SAVE_PATH}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    def evaluate(self, loader):
        """
        Evaluates the model on the validation set.
        Returns average loss and average Levenshtein distance on windows.
        """
        self.model.eval()
        total_loss = 0.0
        total_edit_dist = 0.0
        num_samples = 0

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                loss = self._compute_loss(outputs, labels)
                total_loss += loss.item()

                # Calculate Levenshtein Distance on the final stage output
                # Output 3 is the final refined prediction
                final_logits = outputs[-1]
                predictions = torch.argmax(final_logits, dim=1)  # (Batch, Time)

                # Convert to CPU numpy
                preds_np = predictions.cpu().numpy()
                labels_np = labels.cpu().numpy()

                for i in range(preds_np.shape[0]):
                    # Decode sequences (RLE + remove background)
                    pred_seq = rle_decode(preds_np[i])
                    true_seq = rle_decode(labels_np[i])

                    # Compute Levenshtein distance
                    d = nltk.edit_distance(pred_seq, true_seq)

                    # Normalize by length of true sequence (metric definition: sum(dist)/total_gestures)
                    # Here we approximate by averaging per window
                    total_edit_dist += d
                    num_samples += 1

        avg_loss = total_loss / len(loader)
        avg_edit_dist = total_edit_dist / num_samples if num_samples > 0 else 0.0

        return avg_loss, avg_edit_dist

    def predict_test(self):
        """
        Generates predictions for the test set using full sequences.
        Saves the result to submission.csv.
        """
        print("Generating predictions for test set...")

        # Load best model
        if os.path.exists(Config.MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )
        else:
            print("Warning: No best model found. Using current weights.")

        self.model.eval()

        # Load Test Dataset (Raw sequences)
        # We access the internal list of sequences to process them fully
        test_dataset = GestureDataset(split="test", load_cached_data=True)

        predictions_dict = {}

        with torch.no_grad():
            for seq in test_dataset.sequences:
                sample_id = seq["id"]

                # 1. Prepare Data
                # Skeleton: (T, 20, 3)
                skel = seq["skeleton"]
                # Audio: (T, 13)
                audio = seq["audio"]

                # 2. Feature Engineering (Kinematics)
                # Compute Kinematics -> (T, 20, 9)
                skel_features = compute_kinematics(skel)

                # Flatten -> (T, 180)
                T = skel_features.shape[0]
                skel_flat = skel_features.reshape(T, -1)

                # Concatenate Audio -> (T, 193)
                features = np.concatenate([skel_flat, audio], axis=1)

                # Convert to Tensor -> (1, 193, T)
                features_tensor = torch.from_numpy(features).float()
                features_tensor = (
                    features_tensor.unsqueeze(0).permute(0, 2, 1).to(self.device)
                )

                # 3. Inference
                # Model handles variable length due to fully convolutional/recurrent nature
                outputs = self.model(features_tensor)
                final_logits = outputs[-1]  # (1, Classes, T)

                # 4. Decode
                pred_indices = (
                    torch.argmax(final_logits, dim=1).squeeze(0).cpu().numpy()
                )
                decoded_gestures = rle_decode(pred_indices)

                predictions_dict[sample_id] = decoded_gestures

        # Save Submission
        save_submission(predictions_dict, Config.SUBMISSION_PATH)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
