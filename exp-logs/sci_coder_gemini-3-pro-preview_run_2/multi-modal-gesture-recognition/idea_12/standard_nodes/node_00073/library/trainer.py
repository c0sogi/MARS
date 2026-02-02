import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.ndimage
import pandas as pd
from library.config import Config
from library.utils import set_seed, compute_levenshtein
from library.data_loader import get_dataloaders
from library.model import MultiStageModel
from library.loss import CascadedLoss


class Trainer:
    def __init__(self, load_cached_data=True):
        """
        Initializes the Trainer with model, data loaders, optimizer, and loss function.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(Config.SEED)

        # Data Loaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            load_cached_data=load_cached_data
        )

        # Model
        self.model = MultiStageModel().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Function
        self.criterion = CascadedLoss(device=self.device)

        # State
        self.best_val_score = float("inf")
        self.start_epoch = 0

        # Ensure checkpoint directory exists
        Config.init_dirs()

    def decode_predictions(self, cls_probs, mask):
        """
        Decodes frame-wise probabilities into gesture sequences.
        Applies Median Filtering and collapses repeats.

        Args:
            cls_probs: (B, T, C) Tensor of probabilities
            mask: (B, T) Tensor valid mask

        Returns:
            list of list of int: Predicted gesture sequences
        """
        # Convert to numpy
        cls_probs_np = cls_probs.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()

        predictions = []

        for i in range(cls_probs_np.shape[0]):
            # Get valid length
            valid_len = int(mask_np[i].sum())
            if valid_len == 0:
                predictions.append([])
                continue

            # Slice valid frames
            probs = cls_probs_np[i, :valid_len, :]  # (T_valid, C)

            # Argmax
            labels = np.argmax(probs, axis=1)  # (T_valid,)

            # Post-Processing: Median Filter
            # Kernel size 7 is a reasonable default for smoothing ~0.5s jitters at ~10fps
            labels_smooth = scipy.ndimage.median_filter(labels, size=7, mode="nearest")

            # Decode to sequence: Collapse repeats and remove background (0)
            seq = []
            prev = -1
            for l in labels_smooth:
                if l != prev:
                    if l != 0:  # 0 is background
                        seq.append(int(l))
                    prev = l
            predictions.append(seq)

        return predictions

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        metrics_sum = {}
        num_batches = 0

        for batch in self.train_loader:
            features = batch["features"].to(self.device)
            labels_cls = batch["labels_cls"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(features, mask)

            # Compute Loss
            loss, batch_metrics = self.criterion(outputs, labels_cls, mask)

            # Backward
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Accumulate metrics
            for k, v in batch_metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v

            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}

        return avg_loss, avg_metrics

    def validate(self):
        """
        Runs validation and computes Levenshtein Error Rate.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["features"].to(self.device)
                labels_cls = batch["labels_cls"].to(self.device)
                mask = batch["mask"].to(self.device)
                sample_ids = batch["sample_ids"]

                # Forward
                outputs = self.model(features, mask)

                # Loss
                loss, _ = self.criterion(outputs, labels_cls, mask)
                total_loss += loss.item()

                # Inference (Stage 3)
                s3_probs = outputs["stage3"]["cls_probs"]

                # Decode Predictions
                batch_preds = self.decode_predictions(s3_probs, mask)
                all_preds.extend(batch_preds)

                # Decode Targets (Ground Truth)
                # We can extract from labels_cls directly
                labels_cls_np = labels_cls.cpu().numpy()
                mask_np = mask.cpu().numpy()

                for i in range(len(sample_ids)):
                    valid_len = int(mask_np[i].sum())
                    t_seq_raw = labels_cls_np[i, :valid_len]

                    # Collapse repeats and remove background for target
                    # (Though ground truth in metadata is already a list,
                    # here we reconstruct it from frame-wise to be consistent with loader)
                    # Actually, better to rely on the frame-wise labels provided by loader
                    # which were derived from the metadata list.

                    t_seq = []
                    prev = -1
                    for l in t_seq_raw:
                        if l != prev:
                            if l != 0:
                                t_seq.append(int(l))
                            prev = l
                    all_targets.append(t_seq)

        avg_loss = (
            total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0.0
        )

        # Compute Metric
        score = compute_levenshtein(all_preds, all_targets)

        return avg_loss, score

    def train(self, num_epochs=Config.NUM_EPOCHS):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {num_epochs} epochs...")
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_loss, train_metrics = self.train_epoch(epoch)
            val_loss, val_score = self.validate()

            print(f"Epoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")
            print(f"Val Levenshtein Error: {val_score}")

            # Checkpoint & Early Stopping
            if val_score < self.best_val_score:
                self.best_val_score = val_score
                patience_counter = 0
                save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved with score: {val_score}")
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Score: {self.best_val_score}")

    def predict(self):
        """
        Generates predictions for the test set and saves to submission.csv.
        """
        print("Generating predictions on test set...")

        # Load best model
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )
            print("Loaded best model checkpoint.")
        else:
            print("Warning: No checkpoint found. Using current model state.")

        self.model.eval()
        results = []

        with torch.no_grad():
            for batch in self.test_loader:
                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)
                sample_ids = batch["sample_ids"]

                # Forward
                outputs = self.model(features, mask)
                s3_probs = outputs["stage3"]["cls_probs"]

                # Decode
                batch_preds = self.decode_predictions(s3_probs, mask)

                # Format for submission
                for sid, pred_seq in zip(sample_ids, batch_preds):
                    # Format: SessionID,Label1,Label2,...
                    # If empty, just SessionID
                    pred_str = ",".join(map(str, pred_seq))
                    if pred_str:
                        line = f"{sid},{pred_str}"
                    else:
                        line = f"{sid}"  # Should ideally not happen often if gestures exist
                        # If the format strictly requires a comma, handle it.
                        # Based on example: "Session00001,2,12,3" -> if empty "Session00001" or "Session00001,"?
                        # Usually just ID is fine if empty, or ID, (trailing comma).
                        # We'll stick to ID,Label...

                    results.append(line)

        # Write to file
        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        with open(submission_path, "w") as f:
            for line in results:
                f.write(line + "\n")

        print(f"Submission saved to {submission_path}")
