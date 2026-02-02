import os
import torch
import torch.optim as optim
import numpy as np
from scipy.ndimage import median_filter
from library.config import (
    DEVICE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CACHE_DIR,
    EARLY_STOPPING_PATIENCE,
    NUM_EPOCHS,
    SUBMISSION_FILE,
)
from library.model import DSG_CRCN
from library.loss import ActionSegmentationLoss
from library.utils import set_seed, format_submission


class Trainer:
    """
    Manages training, evaluation, and inference for the DSG-CRCN model.
    """

    def __init__(self):
        set_seed()
        self.device = DEVICE
        self.model = DSG_CRCN().to(self.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        self.criterion = ActionSegmentationLoss()

        # Early Stopping State
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_model_path = os.path.join(CACHE_DIR, "best_model.pth")

    def train_one_epoch(self, loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        metrics_accum = {}
        num_batches = 0

        for batch in loader:
            # Move data to device
            features = batch["features"].to(self.device)
            mask = batch["mask"].to(self.device)
            cls_labels = batch["cls_labels"].to(self.device)
            bnd_labels = batch["bnd_labels"].to(self.device)

            targets = {"cls_labels": cls_labels, "bnd_labels": bnd_labels, "mask": mask}

            # Forward pass
            outputs = self.model(features, mask)

            # Compute loss
            loss, metrics = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            for k, v in metrics.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + v

            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_metrics = {k: v / num_batches for k, v in metrics_accum.items()}

        return avg_loss, avg_metrics

    def evaluate(self, loader):
        """
        Evaluates the model on the validation set.
        Returns average loss and frame-wise accuracy.
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_frames = 0
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)
                cls_labels = batch["cls_labels"].to(self.device)
                bnd_labels = batch["bnd_labels"].to(self.device)

                targets = {
                    "cls_labels": cls_labels,
                    "bnd_labels": bnd_labels,
                    "mask": mask,
                }

                outputs = self.model(features, mask)
                loss, _ = self.criterion(outputs, targets)

                total_loss += loss.item()
                num_batches += 1

                # Compute Frame-wise Accuracy
                # Use final stage classification output
                final_cls = outputs["final_cls"]  # (B, T, C)
                predictions = torch.argmax(final_cls, dim=2)  # (B, T)

                # Mask out padding
                valid_preds = predictions[mask]
                valid_targets = cls_labels[mask]

                total_correct += (valid_preds == valid_targets).sum().item()
                total_frames += valid_targets.numel()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        accuracy = total_correct / total_frames if total_frames > 0 else 0.0

        return avg_loss, accuracy

    def fit(self, train_loader, val_loader, epochs=NUM_EPOCHS):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on {self.device} for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            train_loss, train_metrics = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            print(f"Epoch {epoch}:")
            print(f"  Train Loss: {train_loss}")
            print(f"  Train Metrics: {train_metrics}")
            print(f"  Val Loss: {val_loss}")
            print(f"  Val Accuracy: {val_acc}")

            # Early Stopping Check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  New best model saved to {self.best_model_path}")
            else:
                self.patience_counter += 1
                print(
                    f"  No improvement. Patience: {self.patience_counter}/{EARLY_STOPPING_PATIENCE}"
                )
                if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

        # Load best model for future use
        if os.path.exists(self.best_model_path):
            print(f"Loading best model from {self.best_model_path}")
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )

    def predict(self, test_loader, output_path=SUBMISSION_FILE):
        """
        Generates predictions for the test set and saves to submission file.
        Applies post-processing: Median Filter -> Collapse Repeats -> Remove Background.
        """
        self.model.eval()
        all_sample_ids = []
        all_predictions = []

        print("Generating predictions...")

        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)
                lengths = batch["lengths"]
                sample_ids = batch["sample_ids"]

                outputs = self.model(features, mask)
                final_cls = outputs["final_cls"]  # (B, T, C)

                # Get raw class indices
                batch_preds = torch.argmax(final_cls, dim=2).cpu().numpy()  # (B, T)

                for i, pred_seq in enumerate(batch_preds):
                    length = lengths[i]
                    valid_seq = pred_seq[:length]  # Truncate padding

                    # --- Post-Processing ---
                    # 1. Median Filter (Label-Space Smoothing)
                    # Kernel size 15 as per common video seg practice/heuristics
                    filtered_seq = median_filter(valid_seq, size=15, mode="nearest")

                    # 2. Decoding: Collapse repeats and remove background (0)
                    final_seq = []
                    prev_label = -1

                    for label in filtered_seq:
                        if label != prev_label:
                            if label != 0:  # Remove background
                                final_seq.append(int(label))
                            prev_label = label

                    all_sample_ids.append(sample_ids[i])
                    all_predictions.append(final_seq)

        # Save to CSV
        format_submission(all_sample_ids, all_predictions, output_path)
