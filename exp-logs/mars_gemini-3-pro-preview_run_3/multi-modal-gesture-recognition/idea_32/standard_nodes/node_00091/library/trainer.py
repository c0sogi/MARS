import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from library import config, utils, model, data_loader


class CombinedLoss(nn.Module):
    """
    Computes the weighted sum of CrossEntropy and TruncatedMSE losses
    across all three stages of the GHCMN model.
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()
        # Class weights for CrossEntropy (penalize background less)
        weights = torch.ones(config.NUM_CLASSES)
        weights[config.BACKGROUND_CLASS_ID] = config.BACKGROUND_WEIGHT
        self.ce = nn.CrossEntropyLoss(weight=weights.to(config.DEVICE))

        # Smoothing loss for temporal consistency
        self.tmse = utils.TruncatedMSELoss(threshold=config.SMOOTHING_THRESHOLD)

    def forward(self, outputs, targets):
        """
        Args:
            outputs: Dictionary containing logits and probs from model stages.
            targets: Tensor of shape (Batch, Time) containing ground truth labels.
        """
        # Flatten targets for CrossEntropy: (Batch * Time)
        targets_flat = targets.view(-1)

        # --- Stage 1 ---
        logits1 = outputs["stage1_logits"]  # (B, T, C)
        loss1 = self.ce(logits1.reshape(-1, config.NUM_CLASSES), targets_flat)

        # --- Stage 2 ---
        logits2 = outputs["stage2_logits"]
        loss2_ce = self.ce(logits2.reshape(-1, config.NUM_CLASSES), targets_flat)
        # Apply smoothing on log-probabilities
        log_probs2 = torch.log_softmax(logits2, dim=2)
        loss2_tmse = self.tmse(log_probs2)
        loss2 = loss2_ce + config.SMOOTHING_LOSS_WEIGHT * loss2_tmse

        # --- Stage 3 ---
        logits3 = outputs["stage3_logits"]
        loss3_ce = self.ce(logits3.reshape(-1, config.NUM_CLASSES), targets_flat)
        log_probs3 = torch.log_softmax(logits3, dim=2)
        loss3_tmse = self.tmse(log_probs3)
        loss3 = loss3_ce + config.SMOOTHING_LOSS_WEIGHT * loss3_tmse

        # Weighted Sum
        total_loss = (
            config.LOSS_WEIGHT_STAGE1 * loss1
            + config.LOSS_WEIGHT_STAGE2 * loss2
            + config.LOSS_WEIGHT_STAGE3 * loss3
        )

        return total_loss


class Trainer:
    def __init__(self):
        # Set seeds for reproducibility
        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)

        self.device = config.DEVICE

        # Data Loaders
        print("Initializing Data Loaders...")
        self.train_loader, self.val_loader, self.test_loader = (
            data_loader.get_data_loaders(load_cached_data=True)
        )

        # Model
        print("Initializing Model...")
        self.model = model.GHCMN().to(self.device)

        # Optimization
        self.criterion = CombinedLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)

        # Training State
        self.best_val_score = float("inf")
        self.patience_counter = 0
        self.patience_limit = 10  # Early stopping patience

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(features)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self, loader, compute_metric=True):
        """
        Runs inference on the loader.
        If compute_metric is True, calculates Levenshtein distance (requires ground truth).
        Returns average loss and metric (or None).
        """
        self.model.eval()
        running_loss = 0.0

        # Buffers for sequence reconstruction
        # Map sample_idx -> tensor of shape (SeqLen, NumClasses)
        seq_preds = {}
        seq_counts = {}

        # Initialize buffers based on dataset metadata
        dataset = loader.dataset

        # We need to know the length of each sequence to pre-allocate
        # The dataset stores full skeletons in self.skeletons
        for i in range(len(dataset.ids)):
            seq_len = dataset.skeletons[i].shape[0]
            seq_preds[i] = torch.zeros(
                (seq_len, config.NUM_CLASSES), device=self.device
            )
            seq_counts[i] = torch.zeros((seq_len, 1), device=self.device)

        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(loader):
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)

                # Calculate Validation Loss
                if compute_metric:
                    loss = self.criterion(outputs, labels)
                    running_loss += loss.item()

                # Accumulate probabilities for reconstruction (Stage 3)
                probs = outputs["stage3_probs"]  # (B, T, C)

                # Map batch items back to original sequences
                start_idx = batch_idx * config.BATCH_SIZE
                for i in range(features.size(0)):
                    global_idx = start_idx + i
                    # Retrieve metadata from dataset
                    sample_idx, start_frame = dataset.windows[global_idx]

                    # Determine valid window length (handle padding edge case)
                    seq_len = seq_preds[sample_idx].shape[0]
                    window_len = config.WINDOW_SIZE

                    # The loader might pad the input if it's shorter than window,
                    # or if it's the last window.
                    # We simply add the prediction to the buffer.
                    # We clamp the end index to the sequence length.
                    end_frame = min(start_frame + window_len, seq_len)
                    valid_len = end_frame - start_frame

                    if valid_len > 0:
                        # Slice valid probabilities
                        w_probs = probs[i, :valid_len, :]
                        seq_preds[sample_idx][start_frame:end_frame] += w_probs
                        seq_counts[sample_idx][start_frame:end_frame] += 1

        avg_loss = running_loss / len(loader) if len(loader) > 0 else 0.0

        score = 0.0
        if compute_metric:
            total_dist = 0
            total_len = 0

            for i in range(len(dataset.ids)):
                # Average probabilities
                avg_probs = seq_preds[i] / seq_counts[i].clamp(min=1)
                pred_labels = torch.argmax(avg_probs, dim=1).cpu().numpy()

                # Decode sequence
                pred_seq = utils.process_gesture_sequence(pred_labels)

                # Get Ground Truth
                gt_frame_labels = dataset.labels[i]
                gt_seq = utils.process_gesture_sequence(gt_frame_labels)

                dist = utils.levenshtein_distance(pred_seq, gt_seq)
                total_dist += dist
                total_len += len(gt_seq)

            score = total_dist / total_len if total_len > 0 else 0.0

        return avg_loss, score, seq_preds, seq_counts

    def train(self):
        print(f"Starting training for {config.NUM_EPOCHS} epochs...")

        for epoch in range(1, config.NUM_EPOCHS + 1):
            start_time = time.time()

            train_loss = self.train_epoch(epoch)
            val_loss, val_score, _, _ = self.validate(
                self.val_loader, compute_metric=True
            )

            duration = time.time() - start_time

            print(
                f"Epoch {epoch}/{config.NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Levenshtein: {val_score:.6f} | "
                f"Time: {duration:.2f}s"
            )

            # Checkpointing & Early Stopping
            if val_score < self.best_val_score:
                self.best_val_score = val_score
                self.patience_counter = 0
                torch.save(self.model.state_dict(), config.BEST_MODEL_PATH)
                print(f"  -> New best model saved (Score: {val_score:.6f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience_limit:
                    print("Early stopping triggered.")
                    break

        print("Training complete.")

    def generate_submission(self):
        print("Loading best model for inference...")
        if os.path.exists(config.BEST_MODEL_PATH):
            self.model.load_state_dict(
                torch.load(config.BEST_MODEL_PATH, map_location=self.device)
            )
        else:
            print("Warning: Best model not found, using current weights.")

        print("Generating predictions on test set...")
        _, _, seq_preds, seq_counts = self.validate(
            self.test_loader, compute_metric=False
        )

        submission_rows = []
        dataset = self.test_loader.dataset

        for i in range(len(dataset.ids)):
            sample_id = dataset.ids[i]

            # Average probabilities
            avg_probs = seq_preds[i] / seq_counts[i].clamp(min=1)
            pred_labels = torch.argmax(avg_probs, dim=1).cpu().numpy()

            # Decode
            pred_seq = utils.process_gesture_sequence(pred_labels)

            # Format: SessionID,label1,label2,...
            label_str = ",".join(map(str, pred_seq))
            submission_rows.append(f"{sample_id},{label_str}")

        # Write to file
        with open(config.SUBMISSION_PATH, "w") as f:
            for row in submission_rows:
                f.write(row + "\n")

        print(f"Submission saved to {config.SUBMISSION_PATH}")
