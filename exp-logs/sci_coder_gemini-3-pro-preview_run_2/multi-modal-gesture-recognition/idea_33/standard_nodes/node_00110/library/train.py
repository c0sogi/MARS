import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import Config
from library.model import FISGCN
from library.loss import FISGCNLoss
from library.data_loader import get_dataloaders
from library.utils import (
    compute_normalized_levenshtein,
    post_process_and_decode,
    decode_sequence,
)


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Python's random module is handled in the main script usually,
    # but good to be safe if used here.
    import random

    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """
    Trainer class for the FISG-CN model.
    Handles training loop, validation, checkpointing, and inference.
    """

    def __init__(self, debug=False):
        self.debug = debug
        self.device = torch.device(Config.DEVICE)
        self.checkpoint_dir = Config.CHECKPOINT_DIR
        self.submission_dir = Config.SUBMISSION_DIR
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")

        # Initialize Model
        self.model = FISGCN().to(self.device)

        # Initialize Loss
        self.criterion = FISGCNLoss().to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Load Data
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            batch_size=Config.BATCH_SIZE, debug=self.debug
        )

    def train_epoch(self, epoch_idx):
        """Runs one epoch of training."""
        self.model.train()
        total_loss = 0.0

        # Metrics aggregators
        stage_metrics_sum = {}

        for batch_idx, (features, targets, mask, _) in enumerate(self.train_loader):
            features = features.to(self.device)
            targets = targets.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(features, mask)

            # Compute loss
            loss, metrics = self.criterion(outputs, targets, mask)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Aggregate metrics
            for k, v in metrics.items():
                stage_metrics_sum[k] = stage_metrics_sum.get(k, 0.0) + v

        avg_loss = total_loss / len(self.train_loader)

        # Print training summary
        print(f"Epoch {epoch_idx+1}/{Config.NUM_EPOCHS} [Train] Loss: {avg_loss:.6f}")
        # Optional: Print detailed stage losses if needed
        # for k, v in stage_metrics_sum.items():
        #     print(f"  {k}: {v / len(self.train_loader):.4f}")

        return avg_loss

    def validate(self, epoch_idx):
        """Runs validation and computes Levenshtein error rate."""
        self.model.eval()
        total_loss = 0.0

        all_predictions = []
        all_ground_truths = []

        with torch.no_grad():
            for features, targets, mask, _ in self.val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                mask = mask.to(self.device)

                # Forward pass
                outputs = self.model(features, mask)

                # Compute loss
                loss, _ = self.criterion(outputs, targets, mask)
                total_loss += loss.item()

                # Get predictions from the final stage
                # outputs is a list of dicts, take the last one
                final_output = outputs[-1]
                cls_logits = final_output["cls"]  # (B, T, C)

                # Decode predictions for metric calculation
                # We need to process each sample in the batch individually
                # taking into account the mask (valid length)

                batch_size = features.size(0)
                for b in range(batch_size):
                    # Get valid length
                    valid_len = int(mask[b].sum().item())

                    # Get logits for this sample up to valid length
                    sample_logits = cls_logits[b, :valid_len, :].cpu().numpy()

                    # Get ground truth sequence (remove background 0 and duplicates)
                    # targets[b] is frame-wise labels
                    sample_targets = targets[b, :valid_len].cpu().numpy()
                    gt_sequence = decode_sequence(
                        sample_targets, background_class_id=Config.BACKGROUND_CLASS_ID
                    )

                    # Post-process predictions
                    pred_sequence = post_process_and_decode(
                        sample_logits,
                        kernel_size=7,  # Fixed kernel size for validation
                        background_class_id=Config.BACKGROUND_CLASS_ID,
                    )

                    all_predictions.append(pred_sequence)
                    all_ground_truths.append(gt_sequence)

        avg_loss = total_loss / len(self.val_loader)

        # Compute Metric
        error_rate = compute_normalized_levenshtein(all_predictions, all_ground_truths)

        print(
            f"Epoch {epoch_idx+1}/{Config.NUM_EPOCHS} [Val] Loss: {avg_loss:.6f} | Error Rate: {error_rate}"
        )

        return avg_loss, error_rate

    def fit(self):
        """Main training loop with early stopping."""
        best_error_rate = float("inf")
        patience_counter = 0

        print("Starting training...")

        for epoch in range(Config.NUM_EPOCHS):
            _ = self.train_epoch(epoch)
            _, val_error = self.validate(epoch)

            # Checkpoint & Early Stopping
            if val_error < best_error_rate:
                best_error_rate = val_error
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  New best model saved! Error Rate: {best_error_rate}")
            else:
                patience_counter += 1
                print(
                    f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
                )

            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Error Rate: {best_error_rate}")

    def predict(self):
        """Generates predictions for the test set using the best model."""
        print("Generating submission...")

        # Load best model
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
            print(f"Loaded model from {self.best_model_path}")
        else:
            print("Warning: No checkpoint found. Using current model state.")

        self.model.eval()
        results = []

        with torch.no_grad():
            for features, _, mask, sample_ids in self.test_loader:
                features = features.to(self.device)
                mask = mask.to(self.device)

                outputs = self.model(features, mask)
                final_output = outputs[-1]
                cls_logits = final_output["cls"]

                batch_size = features.size(0)
                for b in range(batch_size):
                    sample_id = sample_ids[b]
                    valid_len = int(mask[b].sum().item())

                    sample_logits = cls_logits[b, :valid_len, :].cpu().numpy()

                    pred_sequence = post_process_and_decode(
                        sample_logits,
                        kernel_size=7,
                        background_class_id=Config.BACKGROUND_CLASS_ID,
                    )

                    # Format: SessionID,Label1,Label2,...
                    pred_str = ",".join(map(str, pred_sequence))
                    results.append(f"{sample_id},{pred_str}")

        # Write submission file
        submission_path = os.path.join(self.submission_dir, "submission.csv")
        with open(submission_path, "w") as f:
            for line in results:
                f.write(line + "\n")

        print(f"Submission saved to {submission_path}")


def train_model(debug=False):
    """
    Entry point function to run the training and inference pipeline.

    Args:
        debug (bool): If True, runs on a small subset of data for debugging.
    """
    set_seed(Config.SEED)

    trainer = Trainer(debug=debug)
    trainer.fit()
    trainer.predict()
