import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import Config
from library.utils import (
    set_seed,
    compute_levenshtein_ratio,
    apply_median_filter,
    decode_predictions,
)
from library.model import MVAIIN
from library.data_loader import get_dataloaders


class Trainer:
    def __init__(self):
        # 1. Setup Environment
        set_seed(Config.SEED)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 2. Data Loaders
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders()

        # 3. Model
        self.model = MVAIIN().to(self.device)

        # 4. Loss Function
        # Background weight = 0.5, others = 1.0
        class_weights = torch.ones(Config.NUM_CLASSES, device=self.device)
        class_weights[Config.BACKGROUND_CLASS_ID] = Config.BG_WEIGHT

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=Config.LABEL_SMOOTHING,
            ignore_index=-100,  # Default ignore index
        )

        # 5. Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS
        )

        # 6. Checkpointing
        self.best_metric = float("inf")
        self.patience_counter = 0
        self.best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        count = 0

        for skeleton, audio, labels, lengths in self.train_loader:
            skeleton = skeleton.to(self.device)
            audio = audio.to(self.device)
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass: (B, T, C)
            outputs = self.model(skeleton, audio, lengths)

            # Reshape for CrossEntropyLoss: (B, C, T) vs (B, T)
            outputs = outputs.permute(0, 2, 1)

            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * skeleton.size(0)
            count += skeleton.size(0)

        return total_loss / count if count > 0 else 0.0

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        count = 0

        all_preds_seq = []
        all_targets_seq = []

        with torch.no_grad():
            for skeleton, audio, labels, lengths in self.val_loader:
                skeleton = skeleton.to(self.device)
                audio = audio.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)

                # Forward
                outputs = self.model(skeleton, audio, lengths)  # (B, T, C)

                # Loss calculation
                outputs_perm = outputs.permute(0, 2, 1)
                loss = self.criterion(outputs_perm, labels)
                total_loss += loss.item() * skeleton.size(0)
                count += skeleton.size(0)

                # Decoding for Metric
                # Get class indices: (B, T)
                preds = torch.argmax(outputs, dim=2).cpu().numpy()
                targets = labels.cpu().numpy()

                # Iterate over batch
                for i in range(len(preds)):
                    length = lengths[i].item()

                    # Slice valid length
                    pred_raw = preds[i, :length]
                    target_raw = targets[i, :length]

                    # 1. Apply Median Filter
                    pred_smooth = apply_median_filter(
                        pred_raw, window_size=Config.MEDIAN_FILTER_WINDOW
                    )

                    # 2. Decode to Sequence (RLE + Filtering)
                    pred_seq = decode_predictions(
                        pred_smooth,
                        min_segment_length=Config.MIN_SEGMENT_LENGTH,
                        background_class_id=Config.BACKGROUND_CLASS_ID,
                    )

                    # Target sequence (Ground Truth is already clean, but we decode to get the list of IDs)
                    # We use min_segment_length=1 for GT to capture everything intended
                    target_seq = decode_predictions(
                        target_raw,
                        min_segment_length=1,
                        background_class_id=Config.BACKGROUND_CLASS_ID,
                    )

                    all_preds_seq.append(pred_seq)
                    all_targets_seq.append(target_seq)

        avg_loss = total_loss / count if count > 0 else 0.0

        # Compute Metric
        error_rate = compute_levenshtein_ratio(all_preds_seq, all_targets_seq)

        return avg_loss, error_rate

    def run(self):
        print(f"Starting training for {Config.EPOCHS} epochs...")

        for epoch in range(1, Config.EPOCHS + 1):
            train_loss = self.train_epoch()
            val_loss, val_error_rate = self.validate()

            self.scheduler.step()

            print(
                f"Epoch {epoch}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Error Rate: {val_error_rate:.6f}"
            )

            # Early Stopping & Checkpointing
            if val_error_rate < self.best_metric:
                self.best_metric = val_error_rate
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved! (Error Rate: {val_error_rate:.6f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= Config.PATIENCE:
                    print(
                        f"Early stopping triggered after {Config.PATIENCE} epochs without improvement."
                    )
                    break

        print(f"Training finished. Best Validation Error Rate: {self.best_metric:.6f}")

    def generate_submission(self):
        print("Generating submission...")

        # Load best model
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
            print("Loaded best model checkpoint.")
        else:
            print("Warning: No checkpoint found. Using current model state.")

        self.model.eval()
        results = []

        # Get test dataframe to map index back to Sample ID
        test_df = pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))

        # Iterate sequentially
        with torch.no_grad():
            # Test loader batch_size is 1
            for i, (skeleton, audio, _, lengths) in enumerate(self.test_loader):
                skeleton = skeleton.to(self.device)
                audio = audio.to(self.device)
                lengths = lengths.to(self.device)

                outputs = self.model(skeleton, audio, lengths)
                preds = torch.argmax(outputs, dim=2).cpu().numpy()

                # Batch size is 1
                length = lengths[0].item()
                pred_raw = preds[0, :length]

                # Post-processing
                pred_smooth = apply_median_filter(
                    pred_raw, window_size=Config.MEDIAN_FILTER_WINDOW
                )
                pred_seq = decode_predictions(
                    pred_smooth,
                    min_segment_length=Config.MIN_SEGMENT_LENGTH,
                    background_class_id=Config.BACKGROUND_CLASS_ID,
                )

                # Format string
                sample_id = test_df.iloc[i]["sample_id"]
                label_str = ",".join(map(str, pred_seq))
                results.append(f"{sample_id},{label_str}")

        # Save to file
        submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        with open(submission_path, "w") as f:
            for line in results:
                f.write(line + "\n")

        print(f"Submission saved to {submission_path}")


def train_model():
    trainer = Trainer()
    trainer.run()
    trainer.generate_submission()
