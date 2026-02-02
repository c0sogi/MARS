import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import time
from library.config import Config
from library.model import MSC_IIN
from library.data_loader import get_dataloaders
from library.utils import (
    compute_levenshtein_score,
    median_filter,
    rle_decode,
    set_seed,
)


class Trainer:
    """
    Manages the training, validation, and submission generation for the MSC-IIN model.
    """

    def __init__(self):
        # 1. Setup Environment
        set_seed()
        self.device = Config.get_device()
        print(f"Using device: {self.device}")

        # 2. Data
        print("Initializing DataLoaders...")
        self.train_loader, self.val_loader, self.test_loader = get_dataloaders()

        # 3. Model
        print("Initializing Model...")
        self.model = MSC_IIN().to(self.device)

        # 4. Loss Function
        # Background Class Weight = 0.5, others = 1.0
        class_weights = torch.ones(Config.NUM_CLASSES).to(self.device)
        class_weights[Config.BACKGROUND_CLASS_ID] = Config.BACKGROUND_WEIGHT

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=Config.LABEL_SMOOTHING,
            reduction="mean",
        )

        # 5. Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
        )

        # 6. State Tracking
        self.best_ler = float("inf")
        self.patience_counter = 0
        self.early_stopping_patience = 10  # Stop if no improvement for 10 epochs

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            if batch is None:
                continue

            # Move to device
            skeleton = batch["skeleton"].to(self.device)
            audio = batch["audio"].to(self.device)
            labels = batch["labels"].to(self.device)
            lengths = batch["lengths"].to(self.device)

            # Forward
            self.optimizer.zero_grad()
            logits = self.model(skeleton, audio, lengths)  # (B, T, C)

            # Reshape for Loss: (B*T, C) vs (B*T)
            # We flatten the batch and time dimensions
            logits_flat = logits.view(-1, Config.NUM_CLASSES)
            labels_flat = labels.view(-1)

            loss = self.criterion(logits_flat, labels_flat)

            # Backward
            loss.backward()

            # Gradient Clipping (Standard stability practice for RNNs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                if batch is None:
                    continue

                skeleton = batch["skeleton"].to(self.device)
                audio = batch["audio"].to(self.device)
                labels = batch["labels"].to(self.device)
                lengths = batch["lengths"].to(self.device)

                logits = self.model(skeleton, audio, lengths)

                # Get predictions: (B, T)
                preds_raw = torch.argmax(logits, dim=2).cpu().numpy()
                targets_raw = labels.cpu().numpy()

                # Process each sequence in the batch
                for i in range(len(batch["ids"])):
                    length = lengths[i].item()

                    # 1. Slice valid length
                    p_seq = preds_raw[i, :length]
                    t_seq = targets_raw[i, :length]

                    # 2. Median Filter
                    p_smoothed = median_filter(
                        p_seq, window_size=Config.MEDIAN_FILTER_WINDOW
                    )

                    # 3. RLE Decode
                    decoded_pred = rle_decode(
                        p_smoothed,
                        background_id=Config.BACKGROUND_CLASS_ID,
                        min_length=Config.MIN_SEGMENT_LENGTH,
                    )

                    # Target decoding (ground truth might also need RLE if it's frame-wise,
                    # but usually we compare against the frame-wise ground truth converted to sequence.
                    # However, the metric is Levenshtein on the *sequence of gestures*.
                    # The loader provides frame-wise labels. We must decode targets too to get the list of gestures.)
                    decoded_target = rle_decode(
                        t_seq,
                        background_id=Config.BACKGROUND_CLASS_ID,
                        min_length=1,  # Ground truth shouldn't be filtered aggressively
                    )

                    all_preds.append(decoded_pred)
                    all_targets.append(decoded_target)

        score = compute_levenshtein_score(all_preds, all_targets)
        return score

    def run(self):
        print("Starting training...")
        start_time = time.time()

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_ler = self.validate()

            # Scheduler Step
            self.scheduler.step()

            duration = time.time() - epoch_start
            print(
                f"Epoch {epoch}/{Config.NUM_EPOCHS} | Time: {duration:.2f}s | Train Loss: {train_loss:.6f} | Val LER: {val_ler}"
            )

            # Checkpoint
            if val_ler < self.best_ler:
                self.best_ler = val_ler
                self.patience_counter = 0
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                print(f"New best model saved with LER: {val_ler}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        total_time = time.time() - start_time
        print(f"Training finished in {total_time:.2f}s. Best Val LER: {self.best_ler}")

    def generate_submission(self):
        print("Generating submission...")

        # Load Best Model
        if os.path.exists(Config.BEST_MODEL_PATH):
            self.model.load_state_dict(
                torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
            )
            print("Loaded best model checkpoint.")
        else:
            print("Warning: No checkpoint found, using current model state.")

        self.model.eval()
        results = []

        with torch.no_grad():
            for batch in self.test_loader:
                if batch is None:
                    continue

                skeleton = batch["skeleton"].to(self.device)
                audio = batch["audio"].to(self.device)
                lengths = batch["lengths"].to(self.device)
                ids = batch["ids"]

                logits = self.model(skeleton, audio, lengths)
                preds_raw = torch.argmax(logits, dim=2).cpu().numpy()

                for i, sample_id in enumerate(ids):
                    length = lengths[i].item()
                    p_seq = preds_raw[i, :length]

                    # Post-processing
                    p_smoothed = median_filter(
                        p_seq, window_size=Config.MEDIAN_FILTER_WINDOW
                    )
                    decoded_seq = rle_decode(
                        p_smoothed,
                        background_id=Config.BACKGROUND_CLASS_ID,
                        min_length=Config.MIN_SEGMENT_LENGTH,
                    )

                    # Format string: "1,2,3"
                    pred_str = ",".join(map(str, decoded_seq))
                    results.append({"Id": sample_id, "Predicted": pred_str})

        # Save to CSV
        # The prompt example format: Session00001,2,12,3
        # But standard submission usually requires headers or specific format.
        # The prompt says: "Session00001,2,12,3". It implies no header or specific header.
        # However, looking at sample_submission.csv description in prompt:
        # "Id,Sequence" -> "300, 13 14 2..."
        # Wait, the prompt "Submission Format" section says:
        # "Session00001,2,12,3"
        # But the dataset info shows "randomPredictions.csv" has headers "Id, Sequence".
        # I will follow the "Submission Format" section text explicitly: "Session00001,2,12,3".
        # But to be safe and compatible with standard CSV readers, I will write lines.

        # Actually, let's stick to the prompt's explicit example:
        # "Session00001,2,12,3"
        # This looks like a CSV where the first column is ID and subsequent columns are labels?
        # Or just one column with ID and one string column?
        # Let's assume the format: ID,Label1,Label2,...
        # Wait, "Session00001,2,12,3" implies comma separated values.

        with open(Config.SUBMISSION_PATH, "w") as f:
            for res in results:
                line = f"{res['Id']},{res['Predicted']}\n"
                f.write(line)

        print(f"Submission saved to {Config.SUBMISSION_PATH}")
