import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import Config
from library.utils import setup_logger, compute_levenshtein, collapse_predictions
from library.model import BA_AKN
from library.loss import BoundaryAwareLoss
from library.data_loader import get_dataloaders


class Trainer:
    """
    Manages the training, validation, and prediction lifecycle for the BA-AKN model.
    """

    def __init__(self):
        self.logger = setup_logger()
        self.device = Config.DEVICE

        # Initialize Model
        self.model = BA_AKN().to(self.device)

        # Initialize Loss
        # BoundaryAwareLoss handles class weights internally
        self.criterion = BoundaryAwareLoss().to(self.device)

        # Initialize Optimizer
        # Using Adam as specified (avoiding AdamW for GRU stability)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Training State
        self.best_val_loss = float("inf")
        self.patience = 10
        self.counter = 0

        self.logger.info(f"Trainer initialized on device: {self.device}")

    def train_epoch(self, train_loader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        running_acc = 0.0
        total_frames = 0

        # Use tqdm for progress tracking if running interactively, else silent
        iterator = train_loader

        for batch_idx, (features, cls_targets, bnd_targets) in enumerate(iterator):
            features = features.to(self.device)
            cls_targets = cls_targets.to(self.device)
            bnd_targets = bnd_targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass (returns dict of outputs from all stages)
            outputs = self.model(features)

            # Compute Multi-Task Loss
            loss, _ = self.criterion(outputs, cls_targets, bnd_targets)

            # Backward pass
            loss.backward()

            # Optimizer step
            self.optimizer.step()

            # Metrics (Stage 3 is final prediction)
            running_loss += loss.item()

            # Calculate Frame Accuracy for Stage 3
            stage3_logits = outputs["stage3_cls"]
            preds = torch.argmax(stage3_logits, dim=1)

            # Mask out padding if necessary?
            # The loader pads, but targets are 0 (background).
            # Accuracy includes background classification.
            correct = (preds == cls_targets).sum().item()
            total = cls_targets.numel()

            running_acc += correct
            total_frames += total

        avg_loss = running_loss / len(train_loader)
        avg_acc = running_acc / total_frames if total_frames > 0 else 0.0

        return avg_loss, avg_acc

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Computes Loss, Frame Accuracy, and Window-level Levenshtein Distance.
        """
        self.model.eval()
        running_loss = 0.0
        running_acc = 0.0
        total_frames = 0

        # For Levenshtein
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for features, cls_targets, bnd_targets in val_loader:
                features = features.to(self.device)
                cls_targets = cls_targets.to(self.device)
                bnd_targets = bnd_targets.to(self.device)

                outputs = self.model(features)

                # Loss
                loss, _ = self.criterion(outputs, cls_targets, bnd_targets)
                running_loss += loss.item()

                # Accuracy (Stage 3)
                stage3_logits = outputs["stage3_cls"]
                preds = torch.argmax(stage3_logits, dim=1)

                correct = (preds == cls_targets).sum().item()
                total = cls_targets.numel()
                running_acc += correct
                total_frames += total

                # Collect for Levenshtein (Window-level)
                # Convert to lists of collapsed sequences
                batch_preds = preds.cpu().numpy()
                batch_targets = cls_targets.cpu().numpy()

                for p, t in zip(batch_preds, batch_targets):
                    p_seq = collapse_predictions(p)
                    t_seq = collapse_predictions(t)
                    all_preds.append(p_seq)
                    all_targets.append(t_seq)

        avg_loss = running_loss / len(val_loader)
        avg_acc = running_acc / total_frames if total_frames > 0 else 0.0

        # Compute Levenshtein
        lev_dist = compute_levenshtein(all_preds, all_targets)

        return avg_loss, avg_acc, lev_dist

    def fit(self, epochs=Config.EPOCHS, load_cached_data=True):
        """
        Main training loop.
        """
        self.logger.info("Loading Data...")
        train_loader, val_loader, test_loader, test_ids = get_dataloaders(
            load_cached_data=load_cached_data
        )

        self.logger.info("Starting Training...")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            val_loss, val_acc, val_lev = self.validate(val_loader)

            self.logger.info(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | "
                f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f} | "
                f"Val Lev: {val_lev:.6f}"
            )

            # Checkpointing & Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.counter = 0
                torch.save(
                    self.model.state_dict(),
                    os.path.join(Config.WORKING_DIR, "best_model.pth"),
                )
                self.logger.info(f"New best model saved with Val Loss: {val_loss:.6f}")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.logger.info("Early stopping triggered.")
                    break

        # After training, generate submission
        self.generate_submission(test_loader, test_ids)

    def generate_submission(self, test_loader, test_ids):
        """
        Generates predictions for the test set and saves to CSV.
        Reconstructs full sequences from overlapping windows.
        """
        self.logger.info("Generating Submission...")

        # Load best model
        best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            self.logger.info("Loaded best model for inference.")
        else:
            self.logger.warning("Best model not found, using current weights.")

        self.model.eval()

        # Access raw dataset to initialize buffers
        # test_loader.dataset is GestureDataset
        dataset = test_loader.dataset
        raw_skeletons = dataset.skeletons  # List of (T, 20, 3)

        # Initialize probability buffers for each sequence
        # seq_probs[seq_idx] = np.array shape (T, NumClasses)
        seq_probs = {}
        seq_counts = {}

        for idx, skel in enumerate(raw_skeletons):
            T = len(skel)
            seq_probs[idx] = np.zeros((T, Config.NUM_CLASSES), dtype=np.float32)
            seq_counts[idx] = np.zeros((T,), dtype=np.float32)

        # Iterate over test loader (windows)
        # We need to map batch items back to sequence indices
        # Since shuffle=False, we can track index
        current_window_idx = 0

        with torch.no_grad():
            for features, _, _ in test_loader:
                batch_size = features.size(0)
                features = features.to(self.device)

                # Forward
                outputs = self.model(features)
                # Use Stage 3 output, apply Softmax
                logits = outputs["stage3_cls"]  # (B, C, W)
                probs = torch.softmax(logits, dim=1).cpu().numpy()  # (B, C, W)

                # Map back to sequences
                for b in range(batch_size):
                    global_idx = current_window_idx + b
                    if global_idx >= len(dataset.windows):
                        break

                    seq_idx, start_frame = dataset.windows[global_idx]

                    # Window data
                    window_probs = probs[b].transpose(1, 0)  # (W, C)

                    # Determine valid range in sequence
                    # dataset.__getitem__ handles padding, but here we just add what we have
                    # The window corresponds to [start_frame : start_frame + window_size]
                    # We need to clip if it goes beyond sequence length
                    seq_len = len(raw_skeletons[seq_idx])
                    end_frame = min(start_frame + Config.WINDOW_SIZE, seq_len)

                    # Length of valid data to add
                    valid_len = end_frame - start_frame

                    if valid_len > 0:
                        # Add probs
                        seq_probs[seq_idx][start_frame:end_frame] += window_probs[
                            :valid_len
                        ]
                        seq_counts[seq_idx][start_frame:end_frame] += 1.0

                current_window_idx += batch_size

        # Decode and Write CSV
        results = []

        for seq_idx in range(len(test_ids)):
            sample_id = test_ids[seq_idx]

            # Average probabilities
            # Avoid division by zero
            counts = seq_counts[seq_idx][:, None]
            counts[counts == 0] = 1.0
            avg_probs = seq_probs[seq_idx] / counts

            # Argmax
            frame_preds = np.argmax(avg_probs, axis=1)

            # Collapse (RLE)
            collapsed_preds = collapse_predictions(frame_preds)

            # Format string
            pred_str = ",".join(map(str, collapsed_preds))
            results.append({"Id": sample_id, "Predicted": pred_str})

        # Create DataFrame
        # The submission format requires: SessionID, Label1, Label2...
        # But the prompt example is: Session00001,2,12,3
        # This implies a CSV without header or specific header?
        # Standard challenge format usually has headers or specific columns.
        # Prompt: "For instance: Session00001,2,12,3"
        # It looks like a CSV where the first column is ID and subsequent are labels,
        # OR a single line per sample joined by commas.

        # Let's write strictly as requested: SessionID,label,label,label

        with open(Config.SUBMISSION_FILE, "w") as f:
            for res in results:
                line = f"{res['Id']}"
                if res["Predicted"]:
                    line += f",{res['Predicted']}"
                f.write(line + "\n")

        self.logger.info(f"Submission saved to {Config.SUBMISSION_FILE}")
