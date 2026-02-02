import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import json
import time

# Import from provided library files
from library.config import Config
from library.losses import CombinedLoss
from library.models import GestureNet
from library.utils import decode_predictions, compute_levenshtein
from library.dataset import get_datasets


class Trainer:
    """
    Trainer class for the Dual-Stream Dynamic-Static Network.
    Handles training, validation, and inference.
    """

    def __init__(self):
        # Setup device
        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        self.model = GestureNet().to(self.device)

        # Initialize Loss
        self.criterion = CombinedLoss().to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Ensure output directories exist
        Config.setup_directories()

    def load_ground_truth(self, csv_path):
        """
        Parses the metadata CSV to extract ground truth gesture sequences for validation.

        Args:
            csv_path (str): Path to the CSV file (e.g., val.csv).

        Returns:
            dict: Mapping from sample_id to list of gesture IDs.
        """
        if not os.path.exists(csv_path):
            print(f"Warning: Ground truth file {csv_path} not found.")
            return {}

        df = pd.read_csv(csv_path)
        gt_map = {}

        for _, row in df.iterrows():
            sample_id = row["sample_id"]
            labels_raw = row["labels"]

            # Parse JSON labels
            if isinstance(labels_raw, str):
                try:
                    labels_list = json.loads(labels_raw)
                except json.JSONDecodeError:
                    labels_list = []
            else:
                labels_list = []

            # Sort by start time to ensure correct sequence order
            labels_list.sort(key=lambda x: x.get("begin", 0))

            # Extract gesture IDs
            sequence = [int(item["id"]) for item in labels_list]
            gt_map[sample_id] = sequence

        return gt_map

    def train_epoch(self, dataloader):
        """
        Runs one epoch of training.
        """
        self.model.train()

        total_loss = 0.0
        total_ce1 = 0.0
        total_ce2 = 0.0
        total_smooth = 0.0
        num_samples = 0

        for static_x, dynamic_x, targets in dataloader:
            # Move to device
            static_x = static_x.to(self.device)
            dynamic_x = dynamic_x.to(self.device)
            targets = targets.to(self.device)

            batch_size = static_x.size(0)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            stage1_logits, stage2_logits = self.model(static_x, dynamic_x)

            # Permute logits for Loss: (B, T, C) -> (B, C, T)
            stage1_logits_t = stage1_logits.permute(0, 2, 1)
            stage2_logits_t = stage2_logits.permute(0, 2, 1)

            # Compute Loss
            loss, ce1, ce2, smooth = self.criterion(
                stage1_logits_t, stage2_logits_t, targets
            )

            # Backward pass
            loss.backward()

            # Update weights
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item() * batch_size
            total_ce1 += ce1.item() * batch_size
            total_ce2 += ce2.item() * batch_size
            total_smooth += smooth.item() * batch_size
            num_samples += batch_size

        # Average metrics
        metrics = {
            "loss": total_loss / num_samples if num_samples > 0 else 0,
            "ce1": total_ce1 / num_samples if num_samples > 0 else 0,
            "ce2": total_ce2 / num_samples if num_samples > 0 else 0,
            "smooth": total_smooth / num_samples if num_samples > 0 else 0,
        }

        return metrics

    def validate(self, dataloader, gt_map):
        """
        Runs validation inference and computes Levenshtein error rate.
        """
        self.model.eval()

        predicted_sequences = []
        ground_truth_sequences = []

        with torch.no_grad():
            for static_x, dynamic_x, sample_ids in dataloader:
                static_x = static_x.to(self.device)
                dynamic_x = dynamic_x.to(self.device)

                # Forward pass (we only care about the refined output)
                _, stage2_logits = self.model(static_x, dynamic_x)

                # Convert logits to probabilities
                probs = torch.softmax(stage2_logits, dim=2)  # (B, T, C)

                # Iterate over the batch
                for i in range(len(sample_ids)):
                    sid = sample_ids[i]
                    sample_probs = probs[i].cpu().numpy()  # (T, C)

                    # Decode sequence
                    pred_seq = decode_predictions(sample_probs)
                    predicted_sequences.append(pred_seq)

                    # Retrieve Ground Truth
                    if sid in gt_map:
                        ground_truth_sequences.append(gt_map[sid])
                    else:
                        ground_truth_sequences.append([])

        # Compute Metric
        score = compute_levenshtein(predicted_sequences, ground_truth_sequences)
        return score

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print("Initializing datasets...")
        # Load datasets (using caching from dataset.py)
        train_dataset, val_dataset, _ = get_datasets(load_cached_data=True)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            drop_last=True,
        )

        # Validation loader with batch_size=1 to handle variable sequence lengths
        val_loader = DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS
        )

        # Load Ground Truth for Validation
        val_gt_map = self.load_ground_truth(Config.VAL_CSV)

        print(f"Starting training on device: {self.device}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        best_score = float("inf")
        patience_counter = 0

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            start_time = time.time()

            # Train Step
            train_metrics = self.train_epoch(train_loader)

            # Validation Step
            val_score = self.validate(val_loader, val_gt_map)

            elapsed = time.time() - start_time

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch}/{Config.NUM_EPOCHS} | Time: {elapsed:.2f}s | "
                f"Train Loss: {train_metrics['loss']:.8f} | "
                f"Val Levenshtein: {val_score:.8f}"
            )

            # Checkpoint & Early Stopping
            if val_score < best_score:
                best_score = val_score
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"New best model saved to {Config.MODEL_SAVE_PATH}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(
                        f"Early stopping triggered. No improvement for {Config.PATIENCE} epochs."
                    )
                    break

        print(f"Training complete. Best Validation Score: {best_score:.8f}")

    def predict(self):
        """
        Generates predictions for the test set and saves to submission file.
        """
        print("Loading test data...")
        _, _, test_dataset = get_datasets(load_cached_data=True)

        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS
        )

        # Load Best Model
        if not os.path.exists(Config.MODEL_SAVE_PATH):
            print(f"Error: Model checkpoint {Config.MODEL_SAVE_PATH} not found.")
            return

        print(f"Loading model from {Config.MODEL_SAVE_PATH}...")
        self.model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
        )
        self.model.eval()

        results = []
        print("Running inference on test set...")

        with torch.no_grad():
            for static_x, dynamic_x, sample_ids in test_loader:
                static_x = static_x.to(self.device)
                dynamic_x = dynamic_x.to(self.device)

                # Forward
                _, stage2_logits = self.model(static_x, dynamic_x)
                probs = torch.softmax(stage2_logits, dim=2)

                for i in range(len(sample_ids)):
                    sid = sample_ids[i]
                    sample_probs = probs[i].cpu().numpy()

                    # Decode
                    pred_seq = decode_predictions(sample_probs)

                    # Format: SessionID,label1,label2,...
                    pred_str = ",".join(map(str, pred_seq))
                    results.append(f"{sid},{pred_str}")

        # Save Submission
        output_path = Config.SUBMISSION_PATH
        try:
            with open(output_path, "w") as f:
                for line in results:
                    f.write(line + "\n")
            print(f"Submission saved successfully to {output_path}")
        except Exception as e:
            print(f"Error saving submission: {e}")
