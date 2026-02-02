import os
import torch
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from library.config import Config
from library.utils import set_seed, compute_levenshtein
from library.data import get_dataloaders
from library.model import IDCRCN
from library.loss import MultiStageLoss


class Trainer:
    def __init__(self, debug=False):
        self.device = Config.get_device()
        self.debug = debug

        # Initialize Model
        self.model = IDCRCN().to(self.device)

        # Initialize Loss
        self.criterion = MultiStageLoss().to(self.device)

        # Initialize Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler (Cite Lesson 00006)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )

        # Load Data
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            debug=self.debug
        )

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        metrics_sum = {}
        num_batches = 0

        for batch in self.train_loader:
            if batch is None:
                continue

            features = batch["features"].to(self.device)
            mask = batch["mask"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            targets = batch["frame_labels"].to(self.device)

            self.optimizer.zero_grad()

            # Forward Pass
            outputs = self.model(features, mask, lengths)

            # Compute Loss
            loss, metrics = self.criterion(outputs, targets, mask)

            # Backward Pass
            loss.backward()

            # Gradient Clipping (Optional but recommended for RNNs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_metrics = {k: v / num_batches for k, v in metrics_sum.items()}

        return avg_loss, avg_metrics

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                if batch is None:
                    continue

                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)
                lengths = batch["lengths"].to(self.device)
                targets_frames = batch["frame_labels"].to(self.device)
                target_sequences = batch["target_sequence"]  # List of lists

                # Forward Pass
                outputs = self.model(features, mask, lengths)

                # Compute Loss (for Early Stopping)
                loss, _ = self.criterion(outputs, targets_frames, mask)
                total_loss += loss.item()
                num_batches += 1

                # Inference & Decoding using Stage 3 output
                # outputs['stage3'] is (Batch, Classes, Time) - Softmax probabilities
                probs = outputs["stage3"].cpu().numpy()
                lengths_cpu = lengths.cpu().numpy()

                for i in range(len(probs)):
                    length = lengths_cpu[i]
                    # Slice valid frames: (Classes, ValidTime) -> (ValidTime, Classes)
                    p = probs[i, :, :length].T

                    # Get frame-wise labels
                    frame_preds = np.argmax(p, axis=1)

                    # Temporal Smoothing: Median Filter
                    # Use nearest padding to preserve boundaries
                    smoothed_preds = median_filter(
                        frame_preds, size=Config.MEDIAN_FILTER_KERNEL, mode="nearest"
                    )

                    # Decode: Collapse repeats and remove background (0)
                    decoded_seq = []
                    prev = -1
                    for label in smoothed_preds:
                        if label != prev:
                            if label != 0:  # 0 is background
                                decoded_seq.append(int(label))
                            prev = label

                    all_preds.append(decoded_seq)
                    all_targets.append(target_sequences[i])

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Compute Levenshtein Error Rate
        error_rate = compute_levenshtein(all_preds, all_targets)

        return avg_loss, error_rate

    def fit(self):
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

        print(f"Starting training on {self.device}...")

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            train_loss, train_metrics = self.train_epoch(epoch)
            val_loss, val_error = self.validate()

            print(f"Epoch {epoch}/{Config.NUM_EPOCHS}")
            print(f"Train Loss: {train_loss:.6f}")
            print(f"Val Loss: {val_loss:.10f}")  # Full precision
            print(f"Val Error Rate: {val_error:.10f}")  # Full precision

            # Step Scheduler
            self.scheduler.step(val_loss)

            # Early Stopping based on Validation Loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                print("New best model saved.")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print("Training complete.")

    def predict(self):
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        if not os.path.exists(best_model_path):
            print("No best model found. Skipping prediction.")
            return

        print(f"Loading best model from {best_model_path}...")
        self.model.load_state_dict(
            torch.load(best_model_path, map_location=self.device)
        )
        self.model.eval()

        results = []

        with torch.no_grad():
            for batch in self.test_loader:
                if batch is None:
                    continue

                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)
                lengths = batch["lengths"].to(self.device)
                sample_ids = batch["sample_ids"]

                # Forward Pass
                outputs = self.model(features, mask, lengths)

                # Use Stage 3 output
                probs = outputs["stage3"].cpu().numpy()
                lengths_cpu = lengths.cpu().numpy()

                for i in range(len(probs)):
                    length = lengths_cpu[i]
                    sample_id = sample_ids[i]

                    # Slice valid frames
                    p = probs[i, :, :length].T

                    # Argmax
                    frame_preds = np.argmax(p, axis=1)

                    # Temporal Smoothing
                    smoothed_preds = median_filter(
                        frame_preds, size=Config.MEDIAN_FILTER_KERNEL, mode="nearest"
                    )

                    # Decode
                    decoded_seq = []
                    prev = -1
                    for label in smoothed_preds:
                        if label != prev:
                            if label != 0:
                                decoded_seq.append(str(int(label)))
                            prev = label

                    # Format: SessionID,Label1,Label2,...
                    line = [sample_id] + decoded_seq
                    results.append(",".join(line))

        # Save Submission
        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        with open(submission_path, "w") as f:
            for line in results:
                f.write(line + "\n")

        print(f"Submission saved to {submission_path}")


def main():
    set_seed(Config.SEED)
    trainer = Trainer(debug=False)
    trainer.fit()
    trainer.predict()


if __name__ == "__main__":
    main()
