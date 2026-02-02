import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.model import SymG_CRCN
from library.loss import MultiStageLoss
from library.dataset import get_dataloader
from library.utils import set_seed, compute_levenshtein_score, apply_median_filter


class Trainer:
    """
    Manages training, validation, and inference for the SymG-CRCN model.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)

        # Initialize Model
        self.model = SymG_CRCN().to(self.device)

        # Initialize Loss
        self.criterion = MultiStageLoss().to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Paths
        self.best_model_path = Config.BEST_MODEL_PATH
        self.submission_path = Config.SUBMISSION_PATH

    def decode_batch(self, logits, lengths):
        """
        Decodes batch logits into gesture sequences.
        1. Argmax to get class indices.
        2. Apply Median Filter.
        3. Collapse repeats and remove background (class 0).
        """
        # logits: (B, T, C) -> probs/argmax -> (B, T)
        preds = torch.argmax(logits, dim=2).cpu().numpy()

        decoded_sequences = []

        # Apply smoothing
        preds_smoothed = apply_median_filter(preds, kernel_size=Config.MEDIAN_FILTER_K)

        for i, seq in enumerate(preds_smoothed):
            length = lengths[i]
            valid_seq = seq[:length]

            # Collapse repeats and remove background
            collapsed = []
            prev = None
            for label in valid_seq:
                if label != prev:
                    if label != Config.BACKGROUND_CLASS_ID:
                        collapsed.append(int(label))
                    prev = label
            decoded_sequences.append(collapsed)

        return decoded_sequences

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # Move data to device
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)
            boundaries = batch["boundaries"].to(self.device)
            mask = batch["mask"].to(self.device)
            lengths = batch["lengths"].to(self.device)

            # Forward pass
            outputs = self.model(features, mask, lengths)

            # Compute loss
            loss, _ = self.criterion(outputs, labels, boundaries, mask)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                features = batch["features"].to(self.device)
                labels = batch["labels"].to(self.device)
                boundaries = batch["boundaries"].to(self.device)
                mask = batch["mask"].to(self.device)
                lengths = batch[
                    "lengths"
                ]  # Keep on CPU for decoding logic if needed, or move if model needs it

                # Model needs lengths on device for packing
                outputs = self.model(features, mask, lengths.to(self.device))

                # Compute loss
                loss, _ = self.criterion(outputs, labels, boundaries, mask)
                total_loss += loss.item()
                num_batches += 1

                # Decode predictions (using Stage 3 outputs)
                s3_logits = outputs["stage3_cls"]
                batch_preds = self.decode_batch(s3_logits, lengths)
                all_preds.extend(batch_preds)

                # Process Ground Truth
                # labels is (B, T), padded with -1 or 0.
                # We need to extract the sequence of gestures (1-20)
                labels_np = labels.cpu().numpy()
                for i, seq in enumerate(labels_np):
                    length = lengths[i]
                    valid_seq = seq[:length]

                    # Collapse repeats and remove background/padding
                    # Note: Ground truth in 'labels' is frame-wise.
                    # The metadata provided 'labels' as a list of gestures (e.g., 2, 12, 3).
                    # However, the dataset loader converts frame-wise annotations.
                    # To evaluate correctly against the metric which compares *sequences of gestures*,
                    # we should collapse the frame-wise ground truth as well.
                    collapsed = []
                    prev = None
                    for label in valid_seq:
                        # Assuming 0 is background, -1 is padding
                        if label > 0:
                            if label != prev:
                                collapsed.append(int(label))
                                prev = label
                            # If label is same as prev, ignore (collapse)
                        else:
                            # Background or padding resets the 'prev' tracker?
                            # Usually background separates gestures.
                            if label == 0:
                                prev = 0  # Explicit background
                            # If -1, just ignore

                    all_targets.append(collapsed)

        avg_loss = total_loss / max(1, num_batches)

        # Compute Levenshtein Score
        lev_score = compute_levenshtein_score(all_preds, all_targets)

        return avg_loss, lev_score

    def train(self, num_epochs=Config.NUM_EPOCHS):
        print(f"Starting training for {num_epochs} epochs...")

        train_loader = get_dataloader(
            "train", batch_size=Config.BATCH_SIZE, shuffle=True, augment=True
        )
        val_loader = get_dataloader(
            "val", batch_size=Config.BATCH_SIZE, shuffle=False, augment=False
        )

        best_score = float("inf")
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_score = self.validate(val_loader)

            print(
                f"Epoch {epoch}/{num_epochs} | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val Levenshtein: {val_score}"
            )

            # Early Stopping Check
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved with score: {best_score}")
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

    def predict(self):
        print("Starting inference on test set...")

        # Load best model
        if not os.path.exists(self.best_model_path):
            print(
                "No best model found. Using current model state (warning: might be untrained)."
            )
        else:
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
            print("Loaded best model.")

        self.model.eval()
        test_loader = get_dataloader(
            "test", batch_size=Config.BATCH_SIZE, shuffle=False, augment=False
        )

        results = []

        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)
                lengths = batch["lengths"]  # CPU
                sample_ids = batch["sample_ids"]

                outputs = self.model(features, mask, lengths.to(self.device))

                # Use Stage 3 for final predictions
                s3_logits = outputs["stage3_cls"]

                batch_preds = self.decode_batch(s3_logits, lengths)

                for sid, pred_seq in zip(sample_ids, batch_preds):
                    # Format: SessionID,label1,label2,...
                    pred_str = ",".join(map(str, pred_seq))
                    results.append(f"{sid},{pred_str}")

        # Save submission
        with open(self.submission_path, "w") as f:
            for line in results:
                f.write(line + "\n")

        print(f"Submission saved to {self.submission_path}")
