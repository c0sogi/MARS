import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.signal
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, get_device, setup_logger
from library.model import MDCRCN
from library.loss import MaskedWeightedCrossEntropy, TMSELoss
from library.data_loader import GestureDataset, collate_fn


class Trainer:
    """
    Trainer class for the Masked Dual-Stage Cascaded Recurrent-Convolutional Network (MD-CRCN).
    Handles the training loop, validation, metric calculation, and checkpointing.
    """

    def __init__(self, load_cached_data=True, limit=None):
        """
        Initialize the Trainer.

        Args:
            load_cached_data (bool): Whether to load data from cache.
            limit (int, optional): Limit the number of samples for debugging.
        """
        # 1. Setup Environment
        self.device = get_device()
        set_seed(Config.SEED)
        self.logger = setup_logger(
            "MD-CRCN-Trainer", os.path.join(Config.WORKING_DIR, "train.log")
        )

        self.logger.info(f"Device: {self.device}")

        # 2. Data Loaders
        self.logger.info("Initializing Data Loaders...")
        train_dataset = GestureDataset(
            split="train", load_cached_data=load_cached_data, limit=limit
        )
        val_dataset = GestureDataset(
            split="val", load_cached_data=load_cached_data, limit=limit
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        # 3. Model
        self.logger.info("Initializing Model...")
        self.model = MDCRCN().to(self.device)

        # 4. Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # 5. Loss Functions
        self.ce_loss = MaskedWeightedCrossEntropy().to(self.device)
        self.tmse_loss = TMSELoss(threshold=Config.TMSE_THRESHOLD).to(self.device)

    def _levenshtein_distance(self, s1, s2):
        """
        Computes the Levenshtein distance between two lists of integers.
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _decode_predictions(self, logits, mask):
        """
        Decodes frame-wise logits into a sequence of gesture IDs.
        Applies median filtering and removes background/duplicates.

        Args:
            logits: (Frames, Classes) tensor
            mask: (Frames,) tensor

        Returns:
            List[int]: Predicted gesture sequence
        """
        # 1. Get valid frames based on mask
        valid_len = int(mask.sum().item())
        valid_logits = logits[:valid_len]  # (T, C)

        # 2. Argmax to get labels
        probs = torch.softmax(valid_logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()

        # 3. Median Filtering (Smoothing)
        # Kernel size must be odd
        k = Config.MEDIAN_FILTER_KERNEL_SIZE
        if len(preds) >= k:
            preds = scipy.signal.medfilt(preds, kernel_size=k)

        # 4. Collapse repetitions and remove background
        sequence = []
        prev = -1
        for p in preds:
            p = int(p)
            if p != prev:
                if p != Config.BACKGROUND_LABEL:
                    sequence.append(p)
                prev = p

        return sequence

    def _decode_targets(self, targets, mask):
        """
        Decodes frame-wise targets into ground truth sequence.
        """
        valid_len = int(mask.sum().item())
        valid_targets = targets[:valid_len].cpu().numpy()

        sequence = []
        prev = -1
        for t in valid_targets:
            t = int(t)
            # Targets in dataset are frame-wise, so we collapse duplicates
            # Background is 0
            if t != prev:
                if t != Config.BACKGROUND_LABEL:
                    sequence.append(t)
                prev = t
        return sequence

    def train_epoch(self, epoch):
        """
        Runs one training epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (features, targets, mask, _) in enumerate(self.train_loader):
            features = features.to(self.device)
            targets = targets.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()

            # Forward Pass
            outputs = self.model(features, mask)

            # --- Loss Calculation ---
            # Stage 1: CE
            loss_s1 = self.ce_loss(outputs["stage1"], targets, mask)

            # Stage 2: CE + T-MSE
            loss_s2_ce = self.ce_loss(outputs["stage2"], targets, mask)
            probs_s2 = torch.softmax(outputs["stage2"], dim=2)
            loss_s2_tmse = self.tmse_loss(probs_s2, mask)
            loss_s2 = loss_s2_ce + loss_s2_tmse

            # Stage 3: CE + T-MSE
            loss_s3_ce = self.ce_loss(outputs["stage3"], targets, mask)
            probs_s3 = torch.softmax(outputs["stage3"], dim=2)
            loss_s3_tmse = self.tmse_loss(probs_s3, mask)
            loss_s3 = loss_s3_ce + loss_s3_tmse

            # Total Loss (Deep Supervision)
            loss = loss_s1 + loss_s2 + loss_s3

            # Backward
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        self.logger.info(f"Epoch {epoch} | Train Loss: {avg_loss:.6f}")
        return avg_loss

    def validate(self, epoch):
        """
        Runs validation loop and calculates Levenshtein Error Rate.
        """
        self.model.eval()
        total_loss = 0.0
        total_dist = 0
        total_gestures = 0
        num_batches = 0

        with torch.no_grad():
            for features, targets, mask, _ in self.val_loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                mask = mask.to(self.device)

                # Forward Pass
                outputs = self.model(features, mask)

                # Validation Loss (using Stage 3 output primarily for tracking)
                loss = self.ce_loss(outputs["stage3"], targets, mask)
                total_loss += loss.item()
                num_batches += 1

                # --- Metric Calculation ---
                # Use Stage 3 output for final prediction
                logits = outputs["stage3"]  # (B, T, C)

                for i in range(logits.size(0)):
                    # Decode Prediction
                    pred_seq = self._decode_predictions(logits[i], mask[i])

                    # Decode Ground Truth
                    true_seq = self._decode_targets(targets[i], mask[i])

                    # Calculate Distance
                    dist = self._levenshtein_distance(pred_seq, true_seq)

                    total_dist += dist
                    total_gestures += len(true_seq)

        avg_loss = total_loss / num_batches

        # Avoid division by zero if validation set is empty or has no gestures (unlikely)
        if total_gestures == 0:
            error_rate = 1.0
        else:
            error_rate = total_dist / total_gestures

        self.logger.info(
            f"Epoch {epoch} | Val Loss: {avg_loss:.6f} | Error Rate: {error_rate}"
        )
        return avg_loss, error_rate

    def fit(self, num_epochs=Config.NUM_EPOCHS):
        """
        Main training loop with Early Stopping.
        """
        best_error_rate = float("inf")
        patience_counter = 0

        self.logger.info(f"Starting training for {num_epochs} epochs...")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_error = self.validate(epoch)

            # Scheduler Step
            self.scheduler.step(val_loss)

            # Checkpoint & Early Stopping
            if val_error < best_error_rate:
                best_error_rate = val_error
                patience_counter = 0
                self.logger.info(f"New best model found! Error Rate: {best_error_rate}")
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
            else:
                patience_counter += 1
                self.logger.info(
                    f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                self.logger.info("Early stopping triggered.")
                break

        self.logger.info(f"Training complete. Best Error Rate: {best_error_rate}")

    def predict_test_set(self):
        """
        Generates predictions for the test set and saves to submission.csv.
        """
        self.logger.info("Loading best model for inference...")
        self.model.load_state_dict(
            torch.load(Config.BEST_MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

        test_dataset = GestureDataset(split="test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
        )

        results = []

        self.logger.info("Generating predictions...")
        with torch.no_grad():
            for features, _, mask, sample_ids in test_loader:
                features = features.to(self.device)
                mask = mask.to(self.device)

                outputs = self.model(features, mask)
                logits = outputs["stage3"]

                for i in range(len(sample_ids)):
                    pred_seq = self._decode_predictions(logits[i], mask[i])

                    # Format: SessionID,label1 label2...
                    # Cite debug_lesson_10: Serialize sequence into a single space-separated string
                    # to ensure the CSV has a fixed number of columns (2).
                    seq_str = " ".join(map(str, pred_seq))
                    results.append(f"{sample_ids[i]},{seq_str}")

        # Save to file
        with open(Config.SUBMISSION_PATH, "w") as f:
            # Write Header matching randomPredictions.csv format
            f.write("Id,Sequence\n")
            for line in results:
                f.write(line + "\n")

        self.logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
