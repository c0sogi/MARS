import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import logging
from library.config import config
from library.model import RHCKN
from library.dataset import get_dataloaders
from library.utils import (
    process_predictions_for_submission,
    compute_sequence_accuracy,
    setup_logger,
)

# Setup logger
logger = setup_logger()


class TruncatedMSELogLoss(nn.Module):
    """
    Computes Truncated MSE Loss on Log-Probabilities to enforce temporal smoothness.
    L = mean( min( || log(p_t) - log(p_{t-1}) ||^2, threshold ) )
    This encourages adjacent frames to have similar probability distributions,
    reducing flickering in predictions.
    """

    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold

    def forward(self, logits):
        # logits: (Batch, Time, Classes)
        # Convert to log probabilities
        log_probs = F.log_softmax(logits, dim=-1)

        # Calculate difference between adjacent frames: (Batch, Time-1, Classes)
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Squared error
        mse = diff.pow(2)

        # Truncate the error to avoid penalizing valid transitions too heavily
        truncated_mse = torch.clamp(mse, max=self.threshold)

        # Mean over all dimensions (Batch, Time, Classes)
        return truncated_mse.mean()


class Trainer:
    """
    Manages the training, validation, and prediction lifecycle of the RHC-KN model.
    """

    def __init__(self, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize Model
        self.model = RHCKN().to(self.device)

        # Optimizer: Adam is used for stable convergence of the recurrent backbone
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Loss Functions
        # Weighted Cross Entropy to handle class imbalance (Background vs Gestures)
        class_weights = config.get_class_weights(self.device)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

        # Smoothing Loss for Refinement Stages
        self.smooth_loss = TruncatedMSELogLoss(threshold=config.LOG_MSE_THRESHOLD)

        # Training State
        self.best_score = float("inf")  # Metric is Error Rate (Lower is better)
        self.start_epoch = 0

    def train_epoch(self, loader):
        """
        Runs one epoch of training.
        Computes the Cascaded Loss: Sum of CE losses for all stages + Smoothing losses for refinement stages.
        """
        self.model.train()
        total_loss = 0.0

        for batch_idx, (features, labels) in enumerate(loader):
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass: returns logits for all 3 stages (Deep Supervision)
            logits1, logits2, logits3 = self.model(features)

            # Permute logits to (Batch, Classes, Time) for CrossEntropyLoss
            l1_perm = logits1.permute(0, 2, 1)
            l2_perm = logits2.permute(0, 2, 1)
            l3_perm = logits3.permute(0, 2, 1)

            # 1. Classification Losses (Weighted CE)
            loss_stage1 = self.ce_loss(l1_perm, labels)
            loss_stage2 = self.ce_loss(l2_perm, labels)
            loss_stage3 = self.ce_loss(l3_perm, labels)

            # 2. Smoothing Losses (Stage 2 & 3 only)
            # Applied on original logits (Batch, Time, Classes)
            smooth_stage2 = self.smooth_loss(logits2)
            smooth_stage3 = self.smooth_loss(logits3)

            # 3. Total Cascaded Loss
            loss = (
                config.LOSS_WEIGHT_STAGE1 * loss_stage1
                + config.LOSS_WEIGHT_STAGE2 * loss_stage2
                + config.LOSS_WEIGHT_STAGE3 * loss_stage3
                + config.SMOOTHING_LOSS_WEIGHT * (smooth_stage2 + smooth_stage3)
            )

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def _reconstruct_sequences(self, loader):
        """
        Reconstructs full sequence probabilities from windowed predictions.
        This is critical for calculating sequence-level metrics like Levenshtein distance.
        Assumes the loader iterates sequentially (shuffle=False) over samples.
        """
        self.model.eval()
        dataset = loader.dataset

        # Initialize buffers for each sample in the dataset
        # We need to access the raw samples to know the full length
        sample_preds = []
        sample_counts = []

        for s in dataset.samples:
            # Determine length from skeleton data
            length = s["skeleton"].shape[0]
            sample_preds.append(
                torch.zeros(length, config.NUM_CLASSES, device=self.device)
            )
            sample_counts.append(torch.zeros(length, 1, device=self.device))

        global_window_idx = 0

        with torch.no_grad():
            for features, _ in loader:
                features = features.to(self.device)

                # Forward pass - use Stage 3 (Final Refinement) for prediction
                _, _, logits3 = self.model(features)
                probs = torch.softmax(logits3, dim=-1)  # (Batch, Window, Classes)

                batch_size = features.size(0)

                for i in range(batch_size):
                    # Safety check
                    if global_window_idx >= len(dataset.windows):
                        break

                    # Get window metadata: which sample and what range does this window cover?
                    s_idx, start, end = dataset.windows[global_window_idx]

                    # Determine valid length
                    # The window in dataset.windows is (start, end) indices of the original sample.
                    # The model output is always config.WINDOW_SIZE.
                    # If end - start < config.WINDOW_SIZE, the input was padded.
                    # We only want to accumulate the valid (non-padded) predictions.
                    valid_len = end - start

                    # Accumulate predictions
                    valid_probs = probs[i, :valid_len, :]

                    sample_preds[s_idx][start:end] += valid_probs
                    sample_counts[s_idx][start:end] += 1.0

                    global_window_idx += 1

        # Normalize accumulated probabilities by the overlap count
        final_sequences = []
        for p, c in zip(sample_preds, sample_counts):
            # Avoid division by zero
            c = c.clamp(min=1.0)
            avg_probs = p / c
            final_sequences.append(avg_probs)

        return final_sequences

    def validate(self, loader):
        """
        Validates the model using the Levenshtein Distance metric on full sequences.
        This aligns perfectly with the competition metric.
        """
        reconstructed_probs = self._reconstruct_sequences(loader)

        hyp_sequences = []
        ref_sequences = []

        dataset = loader.dataset

        for i, probs in enumerate(reconstructed_probs):
            # Decode: Argmax -> RLE -> Filter Short -> Remove Background
            frame_preds = torch.argmax(probs, dim=1).cpu().numpy()
            hyp_list = process_predictions_for_submission(
                frame_preds, background_class=0
            )
            hyp_sequences.append(hyp_list)

            # Get Ground Truth
            # dataset.samples[i]['labels'] is a numpy array of frame labels
            gt_frame_labels = dataset.samples[i]["labels"]
            ref_list = process_predictions_for_submission(
                gt_frame_labels, background_class=0
            )
            ref_sequences.append(ref_list)

        # Compute Metric: Total Levenshtein Distance / Total GT Gestures
        error_rate = compute_sequence_accuracy(hyp_sequences, ref_sequences)
        return error_rate

    def fit(self, train_loader, val_loader, epochs=None):
        """
        Main training loop with Early Stopping.
        """
        if epochs is None:
            epochs = config.EPOCHS

        logger.info(f"Starting training for {epochs} epochs on device {self.device}")

        patience_counter = 0

        for epoch in range(self.start_epoch, epochs):
            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_error = self.validate(val_loader)

            logger.info(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Error Rate: {val_error:.6f}"
            )

            # Checkpoint & Early Stopping
            if val_error < self.best_score:
                self.best_score = val_error
                patience_counter = 0
                torch.save(self.model.state_dict(), config.MODEL_SAVE_PATH)
                logger.info(
                    f"New best model saved with Error Rate: {self.best_score:.6f}"
                )
            else:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break

    def generate_submission(self, test_loader):
        """
        Generates predictions for the test set and saves to CSV.
        Loads the best model weights before prediction.
        """
        # Load best model
        if os.path.exists(config.MODEL_SAVE_PATH):
            logger.info(f"Loading best model from {config.MODEL_SAVE_PATH}")
            self.model.load_state_dict(
                torch.load(config.MODEL_SAVE_PATH, map_location=self.device)
            )
        else:
            logger.warning("No saved model found. Using current weights.")

        reconstructed_probs = self._reconstruct_sequences(test_loader)
        dataset = test_loader.dataset

        results = []

        for i, probs in enumerate(reconstructed_probs):
            sample_id = dataset.samples[i]["id"]

            # Decode
            frame_preds = torch.argmax(probs, dim=1).cpu().numpy()
            pred_labels = process_predictions_for_submission(
                frame_preds, background_class=0
            )

            # Format string: "SessionID,Label1,Label2,..."
            pred_str = ",".join(map(str, pred_labels))
            results.append(f"{sample_id},{pred_str}")

        # Write to file
        with open(config.SUBMISSION_FILE, "w") as f:
            for line in results:
                f.write(line + "\n")

        logger.info(f"Submission saved to {config.SUBMISSION_FILE}")


def run_training():
    """
    Helper function to execute the full training and submission pipeline.
    """
    train_loader, val_loader, test_loader = get_dataloaders()
    trainer = Trainer()
    trainer.fit(train_loader, val_loader)
    trainer.generate_submission(test_loader)
