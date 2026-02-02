import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from library.config import Config
from library.model import HCNCSN
from library.utils import compute_levenshtein_score, decode_sequence, TruncatedMSELoss


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    """
    Manages the training, validation, and inference processes for the HC-NCSN model.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(Config.SEED)

        # Initialize Model
        self.model = HCNCSN().to(self.device)

        # Optimizer (Adam, no Weight Decay here if handled in param groups, but Config has global WD)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss Functions
        # Weighted Cross Entropy
        class_weights = Config.CLASS_WEIGHTS.to(self.device)
        self.ce_criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Smoothing Loss
        self.smooth_criterion = TruncatedMSELoss(
            threshold=Config.SMOOTHING_THRESHOLD
        ).to(self.device)

    def train_epoch(self, dataloader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass -> (Batch, Time, Classes)
            logits1, logits2, logits3 = self.model(inputs)

            # Reshape for CrossEntropy: (Batch, Classes, Time)
            logits1_t = logits1.permute(0, 2, 1)
            logits2_t = logits2.permute(0, 2, 1)
            logits3_t = logits3.permute(0, 2, 1)

            # 1. Classification Losses
            loss_ce1 = self.ce_criterion(logits1_t, targets)
            loss_ce2 = self.ce_criterion(logits2_t, targets)
            loss_ce3 = self.ce_criterion(logits3_t, targets)

            # 2. Smoothing Losses (Stage 2 & 3)
            # Apply log_softmax before passing to TruncatedMSELoss (which expects log-probs)
            log_probs2 = F.log_softmax(logits2, dim=2)
            log_probs3 = F.log_softmax(logits3, dim=2)

            loss_smooth2 = self.smooth_criterion(log_probs2)
            loss_smooth3 = self.smooth_criterion(log_probs3)

            # 3. Total Loss
            loss = (
                Config.LOSS_WEIGHT_STAGE1 * loss_ce1
                + Config.LOSS_WEIGHT_STAGE2 * loss_ce2
                + Config.LOSS_WEIGHT_STAGE3 * loss_ce3
                + Config.SMOOTHING_LAMBDA * (loss_smooth2 + loss_smooth3)
            )

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def predict_sequence(self, features):
        """
        Performs sliding window inference on a single sequence.

        Args:
            features (torch.Tensor): Shape (Time, InputDim)

        Returns:
            np.ndarray: Aggregated probabilities of shape (Time, NumClasses)
        """
        self.model.eval()

        seq_len = features.size(0)
        window_size = Config.WINDOW_SIZE
        stride = Config.INFERENCE_STRIDE

        # Pad if shorter than window
        if seq_len < window_size:
            pad_len = window_size - seq_len
            # Pad features: (Time, Dim) -> pad last dim (0,0), pad 2nd to last (0, pad_len)
            # F.pad pads from last dimension backwards
            features_padded = F.pad(
                features, (0, 0, 0, pad_len), mode="constant", value=0
            )
        else:
            features_padded = features
            pad_len = 0

        padded_len = features_padded.size(0)

        # Prepare windows
        windows = []
        indices = []

        for start in range(0, padded_len - window_size + 1, stride):
            end = start + window_size
            windows.append(features_padded[start:end])
            indices.append((start, end))

        # Handle last window if not covered perfectly
        if padded_len > window_size and (padded_len - window_size) % stride != 0:
            start = padded_len - window_size
            end = padded_len
            # Avoid duplicate if already added
            if indices[-1][0] != start:
                windows.append(features_padded[start:end])
                indices.append((start, end))

        if not windows:  # Should not happen due to padding logic
            windows.append(features_padded)
            indices.append((0, padded_len))

        # Stack into batches
        # For very long sequences, we might need to chunk this batch, but usually fits in memory
        batch_input = torch.stack(windows).to(self.device)

        with torch.no_grad():
            _, _, logits3 = self.model(batch_input)
            probs = F.softmax(logits3, dim=2)  # (NumWindows, WindowSize, Classes)

        # Aggregate
        output_probs = torch.zeros((padded_len, Config.NUM_CLASSES), device=self.device)
        counts = torch.zeros((padded_len, 1), device=self.device)

        for i, (start, end) in enumerate(indices):
            output_probs[start:end] += probs[i]
            counts[start:end] += 1.0

        # Average
        output_probs = output_probs / (counts + 1e-8)

        # Trim padding
        if pad_len > 0:
            output_probs = output_probs[:seq_len]

        return output_probs.cpu().numpy()

    def validate(self, dataloader):
        """
        Evaluates the model on the validation set using Levenshtein distance.
        Expects dataloader to return full sequences (batch_size=1).
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                # Unpack batch (batch_size=1)
                features, targets, _ = batch
                features = features.squeeze(0)  # (Time, Dim)
                targets = targets.squeeze(0).numpy()  # (Time,)

                # Inference
                probs = self.predict_sequence(features)

                # Decode
                pred_seq = decode_sequence(probs)

                # Process targets (frame-wise to sequence of IDs)
                # We need to convert frame-wise targets to the list of gesture IDs for metric
                # The metric function expects list of gesture IDs.
                # We can use the same decode logic or extract from RLE of targets.
                # However, the targets provided by the loader are frame-wise labels (0, 1..20).
                # We should extract the non-zero segments.
                target_seq = decode_sequence(
                    torch.tensor(targets).unsqueeze(1).repeat(1, Config.NUM_CLASSES)
                )
                # Note: decode_sequence expects probs, but we can hack it or write a simple extractor.
                # Actually, strictly speaking, we should extract directly from the frame labels.

                # Extract target sequence from frame labels directly
                target_seq_list = []
                last_label = -1
                for t in targets:
                    if t != last_label:
                        if t != 0:  # Assuming 0 is background
                            target_seq_list.append(int(t))
                        last_label = t
                    # If we are in a segment of non-zero, we don't add again until it changes.
                    # But wait, if we have 1 1 1 0 0 1 1 1, that is two gestures of class 1?
                    # The dataset definition says "Sequence gestures...".
                    # Usually, adjacent identical labels are one gesture unless separated by background or other.
                    # Let's use a simple RLE on the targets.

                # Re-doing target extraction properly:
                # 1. RLE
                target_segments = []
                if len(targets) > 0:
                    curr = targets[0]
                    for i in range(1, len(targets)):
                        if targets[i] != curr:
                            if curr != 0:
                                target_segments.append(int(curr))
                            curr = targets[i]
                    if curr != 0:
                        target_segments.append(int(curr))

                all_preds.append(pred_seq)
                all_targets.append(target_segments)

        score = compute_levenshtein_score(all_preds, all_targets)
        return score

    def fit(self, train_loader, val_loader):
        """
        Main training loop with Early Stopping.
        """
        best_score = float("inf")
        patience_counter = 0

        print(f"Starting training on device: {self.device}")

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = self.train_epoch(train_loader)
            val_score = self.validate(val_loader)

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Levenshtein Error: {val_score:.6f}"
            )

            # Checkpoint
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                # print(f"  New best model saved.")
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print(f"Training complete. Best Validation Score: {best_score:.6f}")

    def generate_submission(self, test_loader, output_path):
        """
        Generates predictions for the test set and saves to CSV.
        """
        # Load best model
        if os.path.exists(Config.MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )
            print("Loaded best model for submission generation.")
        else:
            print("Warning: No checkpoint found. Using current model state.")

        self.model.eval()
        results = []

        print("Generating predictions...")
        with torch.no_grad():
            for batch in test_loader:
                features, _, sample_ids = batch
                features = features.squeeze(0)  # (Time, Dim)
                sample_id = sample_ids[0]  # Tuple of size 1

                probs = self.predict_sequence(features)
                pred_seq = decode_sequence(probs)

                # Format: SessionID,Label1,Label2,...
                # If empty, just SessionID
                pred_str = ",".join(map(str, pred_seq))
                results.append(f"{sample_id},{pred_str}")

        # Save to file
        with open(output_path, "w") as f:
            for line in results:
                f.write(line + "\n")

        print(f"Submission saved to {output_path}")
