import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.signal import medfilt
from tqdm import tqdm

from library.config import Config
from library.utils import (
    get_logger,
    compute_levenshtein,
    save_checkpoint,
    save_submission,
)
from library.model import IICGRN

logger = get_logger(__name__)


def train_one_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for batch in dataloader:
        if batch is None:
            continue

        # Move data to device
        skeleton = batch["skeleton"].to(device)
        audio = batch["audio"].to(device)
        targets = batch["dense_labels"].to(device)
        lengths = batch["lengths"]

        optimizer.zero_grad()

        # Forward pass
        # Output shape: (Batch, Time, NumClasses)
        logits = model(skeleton, audio, lengths)

        # Flatten for CrossEntropyLoss
        # Logits: (Batch * Time, NumClasses)
        # Targets: (Batch * Time)
        loss = criterion(logits.reshape(-1, Config.NUM_CLASSES), targets.reshape(-1))

        # Backward
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        running_loss += loss.item() * skeleton.size(0)
        count += skeleton.size(0)

    # Step scheduler if it's per-iteration or handle it outside
    # Ideally CosineAnnealing is per epoch usually, but can be per step.
    # We'll leave scheduler stepping to the main loop or here if needed.

    avg_loss = running_loss / count if count > 0 else 0.0
    return avg_loss


def decode_predictions(logits_batch, lengths_batch):
    """
    Decodes frame-wise logits into gesture sequences.
    Applies Median Filter -> RLE -> Filtering.
    """
    # Get class indices: (Batch, Time)
    preds = torch.argmax(logits_batch, dim=2).cpu().numpy()

    decoded_sequences = []

    for i, raw_seq in enumerate(preds):
        length = lengths_batch[i]
        # Truncate to actual length (though model sees padding, for metric we care about valid region)
        # However, prompt says "do not mask padded time-steps" for LOSS.
        # For INFERENCE/METRIC, we should probably look at the whole valid sequence.
        valid_seq = raw_seq[:length]

        # 1. Median Filter
        # Kernel size must be odd. Config says 5.
        if len(valid_seq) >= Config.MEDIAN_FILTER_KERNEL:
            smoothed_seq = medfilt(valid_seq, kernel_size=Config.MEDIAN_FILTER_KERNEL)
        else:
            smoothed_seq = valid_seq

        # 2. Run-Length Encoding & Filtering
        # Logic: Iterate, group same labels, keep if != 0 and len >= 5
        final_seq = []

        if len(smoothed_seq) == 0:
            decoded_sequences.append([])
            continue

        current_label = smoothed_seq[0]
        current_len = 1

        # Append a dummy element to force processing the last group
        # or handle loop finish
        for label in smoothed_seq[1:]:
            if label == current_label:
                current_len += 1
            else:
                # Process previous group
                if (
                    current_label != Config.BACKGROUND_LABEL
                    and current_len >= Config.MIN_GESTURE_LENGTH
                ):
                    final_seq.append(int(current_label))

                current_label = label
                current_len = 1

        # Process last group
        if (
            current_label != Config.BACKGROUND_LABEL
            and current_len >= Config.MIN_GESTURE_LENGTH
        ):
            final_seq.append(int(current_label))

        decoded_sequences.append(final_seq)

    return decoded_sequences


def evaluate(model, dataloader, criterion, device):
    """
    Evaluates the model: computes loss and Levenshtein error rate.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_preds = []
    all_targets = []
    all_sample_ids = []

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue

            skeleton = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            dense_targets = batch["dense_labels"].to(device)
            seq_targets = batch["sequence_labels"]  # List of lists
            lengths_tensor = batch["lengths"]
            lengths = lengths_tensor.cpu().numpy()
            sample_ids = batch["sample_ids"]

            # Forward
            logits = model(skeleton, audio, lengths_tensor)

            # Loss
            loss = criterion(
                logits.reshape(-1, Config.NUM_CLASSES), dense_targets.reshape(-1)
            )
            running_loss += loss.item() * skeleton.size(0)
            count += skeleton.size(0)

            # Decode
            batch_preds = decode_predictions(logits, lengths)

            all_preds.extend(batch_preds)
            all_targets.extend(seq_targets)
            all_sample_ids.extend(sample_ids)

    avg_loss = running_loss / count if count > 0 else 0.0

    # Compute Levenshtein
    lev_score = compute_levenshtein(all_preds, all_targets)

    return avg_loss, lev_score, all_preds, all_sample_ids


class Trainer:
    def __init__(self, device_str="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = torch.device(device_str)
        self.model = IICGRN().to(self.device)

        # Loss Function
        # Weighting: Background (0) gets 0.5, others get 1.0
        class_weights = torch.ones(Config.NUM_CLASSES).to(self.device)
        class_weights[Config.BACKGROUND_LABEL] = (
            Config.BACKGROUND_LABEL_WEIGHT
            if hasattr(Config, "BACKGROUND_LABEL_WEIGHT")
            else Config.BACKGROUND_WEIGHT
        )

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=Config.LABEL_SMOOTHING
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=1e-6
        )

    def fit(self, train_loader, val_loader, epochs=Config.EPOCHS):
        logger.info(f"Starting training on {self.device} for {epochs} epochs.")

        best_lev = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = train_one_epoch(
                self.model, train_loader, self.criterion, self.optimizer, self.device
            )

            # Step scheduler
            self.scheduler.step()

            # Validate
            val_loss, val_lev, _, _ = evaluate(
                self.model, val_loader, self.criterion, self.device
            )

            logger.info(
                f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val LEV: {val_lev}"
            )

            # Checkpointing & Early Stopping
            if val_lev < best_lev:
                best_lev = val_lev
                patience_counter = 0
                save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
                save_checkpoint(self.model, self.optimizer, epoch, val_lev, save_path)
                logger.info(f"New best model saved with LEV: {best_lev}")
            else:
                patience_counter += 1

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                logger.info(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break

    def predict(self, test_loader, checkpoint_path=None):
        """
        Runs inference on the test set and generates submission.
        """
        if checkpoint_path is None:
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

        if os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            logger.warning("No checkpoint found! Using current model weights.")

        # Evaluate returns predictions
        # Note: Test set usually has empty sequence_labels, so lev_score will be meaningless/0, which is fine.
        _, _, preds, sample_ids = evaluate(
            self.model, test_loader, self.criterion, self.device
        )

        # Save submission
        save_submission(sample_ids, preds, Config.SUBMISSION_PATH)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
