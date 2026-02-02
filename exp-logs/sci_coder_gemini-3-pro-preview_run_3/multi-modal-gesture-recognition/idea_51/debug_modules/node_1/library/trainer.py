import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import time
import json

from library.config import Config
from library.utils import set_seed, levenshtein_distance, process_frame_predictions
from library.data_loader import get_dataloaders, process_sample
from library.model import SKAGN


class CascadedLoss(nn.Module):
    """
    Computes the sum of Weighted Cross-Entropy and Temporal Log-Space Smoothing
    across all three model stages.
    """

    def __init__(self):
        super(CascadedLoss, self).__init__()
        self.class_weights = Config.get_class_weights()
        self.ce_loss = nn.CrossEntropyLoss(weight=self.class_weights)

        self.smooth_weight = Config.SMOOTHING_LOSS_WEIGHT
        self.smooth_threshold = Config.SMOOTHING_LOSS_THRESHOLD

    def temporal_smoothing_loss(self, logits):
        """
        Computes Truncated MSE on log-probabilities between t and t-1.
        """
        # Convert logits to log_probs: (Batch, Time, Classes)
        # Permute logits from (Batch, Classes, Time) to (Batch, Time, Classes) if needed
        # The model output is (Batch, Time, Classes) based on SKAGN definition.

        log_probs = F.log_softmax(logits, dim=2)

        # Calculate diff: log_P(t) - log_P(t-1)
        # Slice: 1: vs :-1
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # MSE
        mse = diff**2

        # Truncate
        truncated_mse = torch.clamp(mse, max=self.smooth_threshold)

        return truncated_mse.mean()

    def forward(self, stage_outputs, targets):
        """
        stage_outputs: tuple of (logits_1, logits_2, logits_3)
        targets: (Batch, Time)
        """
        total_loss = 0.0

        # Flatten targets for CE: (Batch * Time)
        # Logits for CE: (Batch * Time, Classes)
        b, t, c = stage_outputs[0].shape
        flat_targets = targets.view(-1)

        for logits in stage_outputs:
            # 1. Cross Entropy
            flat_logits = logits.reshape(-1, c)
            ce = self.ce_loss(flat_logits, flat_targets)

            # 2. Smoothing
            smooth = self.temporal_smoothing_loss(logits)

            total_loss += ce + (self.smooth_weight * smooth)

        return total_loss


class Trainer:
    def __init__(self, load_cached_data=True):
        set_seed(Config.SEED)
        self.device = torch.device(Config.DEVICE)

        # Data
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            load_cached_data=load_cached_data
        )

        # Model
        self.model = SKAGN().to(self.device)

        # Optimization
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.criterion = CascadedLoss().to(self.device)

        # Paths
        self.best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # Forward (returns tuple of logits)
            outputs = self.model(data)

            loss = self.criterion(outputs, target)
            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate_loss(self):
        """Computes loss on the validation windows."""
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                running_loss += loss.item()

        return running_loss / len(self.val_loader)

    def validate_levenshtein(self):
        """
        Computes Levenshtein distance on full validation sequences.
        This iterates over the validation metadata file directly to reconstruct sequences.
        """
        self.model.eval()

        # Load validation metadata
        if not os.path.exists(Config.VAL_METADATA_PATH):
            print("Validation metadata not found, skipping Levenshtein check.")
            return float("inf")

        df_val = pd.read_csv(Config.VAL_METADATA_PATH)

        total_dist = 0
        total_gestures = 0

        with torch.no_grad():
            for _, row in df_val.iterrows():
                # Process full sequence
                # Note: process_sample returns (Time, InputDim), (Time,)
                features, labels = process_sample(row, Config.INPUT_DIR, augment=False)

                if features is None:
                    continue

                # Prepare input: (1, Time, InputDim)
                x_tensor = (
                    torch.from_numpy(features).float().unsqueeze(0).to(self.device)
                )

                # Inference
                # SKAGN returns (logits1, logits2, logits3)
                _, _, logits_3 = self.model(x_tensor)

                # Decode
                probs = F.softmax(logits_3, dim=2)
                preds = torch.argmax(probs, dim=2).squeeze(0).cpu().numpy()

                # Post-process to get sequence of IDs
                predicted_seq = process_frame_predictions(preds)

                # Ground Truth Sequence
                # Parse JSON labels
                if isinstance(row["labels"], str):
                    gt_list_dicts = json.loads(row["labels"])
                else:
                    gt_list_dicts = row["labels"]

                # Extract IDs in order
                gt_seq = [g["id"] for g in gt_list_dicts]

                # Compute Distance
                dist = levenshtein_distance(predicted_seq, gt_seq)
                total_dist += dist
                total_gestures += len(gt_seq)

        # Avoid division by zero
        if total_gestures == 0:
            return 0.0

        return total_dist / total_gestures

    def fit(self, epochs=Config.EPOCHS, patience=Config.EARLY_STOPPING_PATIENCE):
        best_score = float("inf")
        patience_counter = 0

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_loss()
            val_score = self.validate_levenshtein()

            duration = time.time() - start_time

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Levenshtein (Error Rate): {val_score:.6f} | "
                f"Time: {duration:.2f}s"
            )

            # Checkpoint
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  -> New best model saved! Score: {best_score:.6f}")
            else:
                patience_counter += 1
                print(f"  -> No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Score: {best_score:.6f}")

    def generate_submission(self):
        print("Generating submission...")

        # Load best model
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
            print("Loaded best model checkpoint.")
        else:
            print("Warning: Best model not found, using current weights.")

        self.model.eval()

        # Load test metadata
        if not os.path.exists(Config.TEST_METADATA_PATH):
            raise FileNotFoundError("Test metadata not found.")

        df_test = pd.read_csv(Config.TEST_METADATA_PATH)
        submission_lines = []

        with torch.no_grad():
            for _, row in df_test.iterrows():
                sample_id = row["sample_id"]

                # Process full sequence
                features, _ = process_sample(row, Config.INPUT_DIR, augment=False)

                if features is None:
                    # Fallback for empty/corrupt files
                    submission_lines.append(f"{sample_id},")
                    continue

                # Prepare input
                x_tensor = (
                    torch.from_numpy(features).float().unsqueeze(0).to(self.device)
                )

                # Inference
                _, _, logits_3 = self.model(x_tensor)

                # Decode
                probs = F.softmax(logits_3, dim=2)
                preds = torch.argmax(probs, dim=2).squeeze(0).cpu().numpy()

                # Post-process
                predicted_seq = process_frame_predictions(preds)

                # Format string: "SessionID,Label1,Label2,..."
                pred_str = ",".join(map(str, predicted_seq))
                line = f"{sample_id},{pred_str}"
                submission_lines.append(line)

        # Write to file
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        with open(Config.SUBMISSION_PATH, "w") as f:
            for line in submission_lines:
                f.write(line + "\n")

        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training():
    trainer = Trainer(load_cached_data=True)
    trainer.fit()
    trainer.generate_submission()


if __name__ == "__main__":
    # This block is for testing the module independently if needed,
    # but the prompt asks to only implement the class/functions.
    # The user will call run_training() or instantiate Trainer.
    pass
