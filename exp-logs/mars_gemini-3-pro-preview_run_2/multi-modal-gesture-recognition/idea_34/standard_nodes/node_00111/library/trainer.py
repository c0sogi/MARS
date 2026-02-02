import os
import torch
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from library.config import (
    HYPERPARAMS,
    CHECKPOINT_DIR,
    SUBMISSION_DIR,
    SEED,
)
from library.utils import (
    set_seed,
    compute_metric_score,
    decode_sequence,
    format_submission_line,
)
from library.model import RLSGCN
from library.loss import DeepSupervisionLoss


class Trainer:
    """
    Trainer class for the Residual-Logit Supervised Gated-Cascaded Network.
    Handles training, validation, checkpointing, and inference.
    """

    def __init__(self, train_loader, val_loader, test_loader, device=None):
        """
        Args:
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            test_loader: DataLoader for test data.
            device: torch.device (optional).
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"Using device: {self.device}")

        # Initialize Model
        set_seed(SEED)
        self.model = RLSGCN().to(self.device)

        # Initialize Loss
        self.criterion = DeepSupervisionLoss().to(self.device)

        # Initialize Optimizer
        hp = HYPERPARAMS
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=hp["learning_rate"],
            weight_decay=hp["weight_decay"],
        )

        # Training State
        self.best_val_score = float("inf")  # Lower is better (Error Rate)
        self.start_epoch = 0
        self.history = []

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Iterate over batches
        # Note: tqdm is not used to keep output clean as per instructions
        for batch in self.train_loader:
            features = batch["features"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Targets
            cls_target = batch["cls_target"].to(self.device)
            bnd_target = batch["bnd_target"].to(self.device)
            targets = {"cls_target": cls_target, "bnd_target": bnd_target}

            # Forward Pass
            self.optimizer.zero_grad()
            outputs = self.model(features, mask)

            # Compute Loss
            loss, _ = self.criterion(outputs, targets, mask)

            # Backward Pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), HYPERPARAMS["clip_grad_norm"]
            )

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self):
        """
        Runs validation and computes Levenshtein Error Rate.
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)

                cls_target = batch["cls_target"].to(self.device)
                bnd_target = batch["bnd_target"].to(self.device)
                targets = {"cls_target": cls_target, "bnd_target": bnd_target}
                lengths = batch["lengths"]

                # Forward Pass
                outputs = self.model(features, mask)
                loss, _ = self.criterion(outputs, targets, mask)
                total_loss += loss.item()
                num_batches += 1

                # Extract Stage 3 Predictions (Best Refinement)
                # Probs: (B, T, C)
                stage3_probs = outputs["stage3"]["cls_probs"]

                # Convert to CPU for metric calculation
                stage3_probs_np = stage3_probs.cpu().numpy()
                targets_np = cls_target.cpu().numpy()
                lengths_np = lengths.cpu().numpy()

                # Process batch for metrics
                for i in range(len(lengths_np)):
                    length = lengths_np[i]

                    # 1. Get raw probabilities for valid frames
                    probs = stage3_probs_np[i, :length, :]

                    # 2. Post-process (Median Filter + Decoding)
                    pred_seq = self._post_process_single(probs)
                    all_preds.append(pred_seq)

                    # 3. Get Target Sequence
                    # Decode the frame-wise target to get the ground truth sequence
                    target_frames = targets_np[i, :length]
                    target_seq = decode_sequence(target_frames)
                    all_targets.append(target_seq)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Compute Metric
        score = compute_metric_score(all_preds, all_targets)

        return avg_loss, score

    def _post_process_single(self, probs):
        """
        Applies post-processing to a single sequence's probability matrix.
        1. Argmax to get labels.
        2. Median Filter with Nearest-Neighbor padding.
        3. Decode to sequence.
        """
        # Argmax
        raw_labels = np.argmax(probs, axis=1)

        # Median Filter
        # Kernel size 7 is a reasonable default for ~20fps data to smooth jitter
        # mode='nearest' implements Nearest-Neighbor Padding at edges
        filtered_labels = median_filter(raw_labels, size=7, mode="nearest")

        # Decode
        return decode_sequence(filtered_labels)

    def fit(self, num_epochs=None):
        """
        Main training loop with Early Stopping.
        """
        if num_epochs is None:
            num_epochs = HYPERPARAMS["num_epochs"]

        print(f"Starting training for {num_epochs} epochs...")

        patience = 10
        patience_counter = 0

        for epoch in range(self.start_epoch, num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, val_score = self.validate()

            print(
                f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val Score (Error Rate): {val_score}"
            )

            # Checkpoint & Early Stopping
            if val_score < self.best_val_score:
                self.best_val_score = val_score
                patience_counter = 0

                # Save Best Model
                save_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                # print(f"New best model saved to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        print(f"Training complete. Best Validation Score: {self.best_val_score}")

    def predict(self):
        """
        Runs inference on the test set and generates the submission file.
        """
        print("Running inference on test set...")

        # Load Best Model
        load_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
        if os.path.exists(load_path):
            self.model.load_state_dict(torch.load(load_path, map_location=self.device))
            print("Loaded best model for inference.")
        else:
            print("Warning: No checkpoint found. Using current model weights.")

        self.model.eval()
        submission_lines = []

        with torch.no_grad():
            for batch in self.test_loader:
                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)
                lengths = batch["lengths"]
                sample_ids = batch["sample_ids"]

                # Forward Pass
                outputs = self.model(features, mask)

                # Use Stage 3 for final predictions
                stage3_probs = outputs["stage3"]["cls_probs"]

                stage3_probs_np = stage3_probs.cpu().numpy()
                lengths_np = lengths.cpu().numpy()

                for i in range(len(sample_ids)):
                    sid = sample_ids[i]
                    length = lengths_np[i]

                    # Get probabilities for valid frames
                    probs = stage3_probs_np[i, :length, :]

                    # Post-process
                    pred_seq = self._post_process_single(probs)

                    # Format Line
                    line = format_submission_line(sid, pred_seq)
                    submission_lines.append(line)

        # Save Submission
        sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        with open(sub_path, "w") as f:
            for line in submission_lines:
                f.write(line + "\n")

        print(f"Submission saved to {sub_path}")
