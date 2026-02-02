import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import (
    LogSpaceSmoothingLoss,
    compute_dataset_metrics,
    decode_predictions,
    calculate_levenshtein_distance,
)
from library.data_loader import get_data_loaders

# ==========================================
# Model Architecture: SHC-GKN
# ==========================================


class GatedConv1d(nn.Module):
    """
    Dilated Temporal Convolution with Gated Activation (Tanh * Sigmoid).
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(GatedConv1d, self).__init__()
        # Centered padding: padding = dilation for kernel_size=3
        self.conv_f = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation
        )
        self.conv_g = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        f = torch.tanh(self.conv_f(x))
        g = torch.sigmoid(self.conv_g(x))
        return self.dropout(f * g)


class TCNStack(nn.Module):
    """
    Stack of Gated TCN layers with monotonically increasing dilation.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, dilations, dropout):
        super(TCNStack, self).__init__()
        layers = []
        current_in = in_channels
        for d in dilations:
            layers.append(
                GatedConv1d(
                    current_in,
                    hidden_channels,
                    kernel_size=3,
                    dilation=d,
                    dropout=dropout,
                )
            )
            current_in = hidden_channels

        self.net = nn.Sequential(*layers)
        self.final = nn.Conv1d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # x: (B, C, T)
        out = self.net(x)
        return self.final(out)


class SHCGKN(nn.Module):
    """
    Stabilized High-Capacity Gated-Kinematic Network.
    Stage 1: Gated Bi-GRU
    Stage 2: Refinement TCN (Shared Input, Independent Weights)
    Stage 3: Refinement TCN (Independent Weights)
    """

    def __init__(self, config=Config):
        super(SHCGKN, self).__init__()

        # Stage 1: Stabilized Gated Encoder
        self.bn_input = nn.BatchNorm1d(config.INPUT_DIM)
        self.gate_fc = nn.Linear(config.INPUT_DIM, config.INPUT_DIM)

        self.gru = nn.GRU(
            input_size=config.INPUT_DIM,
            hidden_size=config.HIDDEN_SIZE // 2,  # Bidirectional split
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout_encoder = nn.Dropout(config.DROPOUT_ENCODER)
        self.fc_stage1 = nn.Linear(config.HIDDEN_SIZE, config.NUM_CLASSES)

        # Stage 2: Monotonic Non-Causal Refinement
        self.stage2_tcn = TCNStack(
            in_channels=config.NUM_CLASSES,
            hidden_channels=config.TCN_CHANNELS,
            out_channels=config.NUM_CLASSES,
            dilations=config.TCN_DILATIONS,
            dropout=config.DROPOUT_TCN,
        )

        # Stage 3: Independent Iterative Refinement
        self.stage3_tcn = TCNStack(
            in_channels=config.NUM_CLASSES,
            hidden_channels=config.TCN_CHANNELS,
            out_channels=config.NUM_CLASSES,
            dilations=config.TCN_DILATIONS,
            dropout=config.DROPOUT_TCN,
        )

    def forward(self, x):
        # x: (Batch, Time, Features)
        B, T, F = x.size()

        # --- Stage 1 ---
        # Normalize features (B, F, T)
        x_perm = x.permute(0, 2, 1)
        x_norm = self.bn_input(x_perm).permute(0, 2, 1)  # Back to (B, T, F)

        # Input Gating
        gate = torch.sigmoid(self.gate_fc(x_norm))
        x_gated = x_norm * gate

        # Encoder
        gru_out, _ = self.gru(x_gated)
        gru_out = self.dropout_encoder(gru_out)
        logits1 = self.fc_stage1(gru_out)  # (B, T, Classes)
        probs1 = torch.softmax(logits1, dim=2)

        # --- Stage 2 ---
        # Input: Probs from Stage 1. TCN expects (B, C, T)
        probs1_perm = probs1.permute(0, 2, 1)
        logits2 = self.stage2_tcn(probs1_perm).permute(0, 2, 1)  # (B, T, Classes)
        probs2 = torch.softmax(logits2, dim=2)

        # --- Stage 3 ---
        # Input: Probs from Stage 2
        probs2_perm = probs2.permute(0, 2, 1)
        logits3 = self.stage3_tcn(probs2_perm).permute(0, 2, 1)  # (B, T, Classes)
        probs3 = torch.softmax(logits3, dim=2)

        return {
            "logits1": logits1,
            "probs1": probs1,
            "logits2": logits2,
            "probs2": probs2,
            "logits3": logits3,
            "probs3": probs3,
        }


# ==========================================
# Trainer
# ==========================================


class Trainer:
    def __init__(self, config=Config):
        self.config = config
        self.device = config.get_device()

        # Data Loaders
        self.train_loader, self.val_loader, self.test_loader = get_data_loaders(config)

        # Model
        self.model = SHCGKN(config).to(self.device)

        # Optimization
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Loss Functions
        self.ce_criterion = nn.CrossEntropyLoss(
            weight=config.get_class_weights_tensor().to(self.device)
        )
        self.smooth_criterion = LogSpaceSmoothingLoss(
            threshold=config.SMOOTHING_THRESHOLD
        )

        # State
        self.best_val_score = float("inf")
        self.patience_counter = 0
        self.early_stopping_patience = 10

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(features)

            # Deep Supervision Loss
            # Stage 1: CE
            loss1 = self.ce_criterion(
                outputs["logits1"].reshape(-1, self.config.NUM_CLASSES),
                labels.reshape(-1),
            )

            # Stage 2: CE + Smoothing
            loss2_ce = self.ce_criterion(
                outputs["logits2"].reshape(-1, self.config.NUM_CLASSES),
                labels.reshape(-1),
            )
            loss2_smooth = self.smooth_criterion(
                F.log_softmax(outputs["logits2"], dim=2)
            )
            loss2 = loss2_ce + self.config.SMOOTHING_LOSS_WEIGHT * loss2_smooth

            # Stage 3: CE + Smoothing
            loss3_ce = self.ce_criterion(
                outputs["logits3"].reshape(-1, self.config.NUM_CLASSES),
                labels.reshape(-1),
            )
            loss3_smooth = self.smooth_criterion(
                F.log_softmax(outputs["logits3"], dim=2)
            )
            loss3 = loss3_ce + self.config.SMOOTHING_LOSS_WEIGHT * loss3_smooth

            loss = loss1 + loss2 + loss3

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self, loader):
        """
        Reconstructs full sequences from sliding windows and computes Levenshtein distance.
        """
        self.model.eval()

        dataset = loader.dataset
        total_frames = (
            dataset.all_labels.shape[0] if dataset.all_labels is not None else 0
        )

        # If dataset is empty (e.g. debug mode too small), return worst score
        if total_frames == 0:
            return float("inf")

        # Global accumulators for reconstruction
        global_probs = np.zeros(
            (total_frames, self.config.NUM_CLASSES), dtype=np.float32
        )
        global_counts = np.zeros((total_frames,), dtype=np.float32)

        with torch.no_grad():
            for batch_idx, (features, _) in enumerate(loader):
                features = features.to(self.device)
                outputs = self.model(features)
                probs = outputs["probs3"].cpu().numpy()  # Use Stage 3 output

                # Map batch windows back to global timeline
                start_window_idx = batch_idx * loader.batch_size

                for i in range(features.size(0)):
                    window_idx = start_window_idx + i
                    if window_idx < len(dataset.windows):
                        start_frame, end_frame = dataset.windows[window_idx]

                        # Accumulate probabilities (Temporal Ensembling)
                        # Window might be padded in __getitem__, but dataset.windows stores valid range in original array
                        # However, __getitem__ returns fixed window_size.
                        # We need to slice the prediction to match the target range if it was padded?
                        # In data_loader.py, windows are created based on sample_indices.
                        # __getitem__ slices from all_skeleton.
                        # If window length < window_size, it pads.
                        # We should only add the valid part.

                        valid_len = end_frame - start_frame
                        pred_len = probs.shape[1]  # window_size

                        # If the window in dataset is actually smaller than window_size (at boundaries)
                        # The loader pads the end. We take the first valid_len frames.

                        # Wait, data_loader logic:
                        # windows list contains (start, end) indices into the big array.
                        # __getitem__ calculates length = end - start.
                        # If length < window_size, it pads.
                        # So we should only take probs[i, :length, :]

                        actual_len = min(valid_len, pred_len)

                        global_probs[start_frame : start_frame + actual_len] += probs[
                            i, :actual_len
                        ]
                        global_counts[start_frame : start_frame + actual_len] += 1.0

        # Normalize
        # Avoid division by zero
        mask = global_counts > 0
        global_probs[mask] /= global_counts[mask, None]

        # Reconstruct Sequences and Compute Metrics
        all_preds = []
        all_targets = []

        # Iterate over samples using sample_indices
        for start, end in dataset.sample_indices:
            # Extract sample probabilities
            sample_probs = global_probs[start:end]

            # Decode
            pred_seq = decode_predictions(sample_probs)
            all_preds.append(pred_seq)

            # Get Ground Truth
            # Extract from all_labels
            gt_labels = dataset.all_labels[start:end]

            # Convert frame-wise GT to sequence (ignoring background)
            # Simple RLE on GT
            gt_seq = decode_predictions(np.eye(self.config.NUM_CLASSES)[gt_labels])
            all_targets.append(gt_seq)

        score = compute_dataset_metrics(all_preds, all_targets)
        return score

    def generate_submission(self):
        """
        Generates predictions for the test set and saves to CSV.
        """
        print("Generating submission...")
        self.model.eval()

        dataset = self.test_loader.dataset

        # Reconstruct logic (same as validate)
        # We need total frames. For test set, all_labels is 0s but has correct shape.
        total_frames = dataset.all_labels.shape[0]

        global_probs = np.zeros(
            (total_frames, self.config.NUM_CLASSES), dtype=np.float32
        )
        global_counts = np.zeros((total_frames,), dtype=np.float32)

        with torch.no_grad():
            for batch_idx, (features, _) in enumerate(self.test_loader):
                features = features.to(self.device)
                outputs = self.model(features)
                probs = outputs["probs3"].cpu().numpy()

                start_window_idx = batch_idx * self.test_loader.batch_size

                for i in range(features.size(0)):
                    window_idx = start_window_idx + i
                    if window_idx < len(dataset.windows):
                        start_frame, end_frame = dataset.windows[window_idx]
                        valid_len = end_frame - start_frame
                        pred_len = probs.shape[1]
                        actual_len = min(valid_len, pred_len)

                        global_probs[start_frame : start_frame + actual_len] += probs[
                            i, :actual_len
                        ]
                        global_counts[start_frame : start_frame + actual_len] += 1.0

        mask = global_counts > 0
        global_probs[mask] /= global_counts[mask, None]

        # Read test metadata to get Sample IDs
        test_meta = pd.read_csv(self.config.TEST_METADATA_PATH)
        if self.config.DEBUG:
            test_meta = test_meta.head(self.config.DEBUG_SUBSET_SIZE)

        results = []

        # dataset.sample_indices corresponds to rows in test_meta
        for idx, (start, end) in enumerate(dataset.sample_indices):
            sample_id = test_meta.iloc[idx]["sample_id"]

            sample_probs = global_probs[start:end]
            pred_seq = decode_predictions(sample_probs)

            # Format: "SampleID,Label1,Label2,..."
            # Labels are ints. Join with commas.
            pred_str = ",".join(map(str, pred_seq))
            results.append(f"{sample_id},{pred_str}")

        # Save
        with open(self.config.SUBMISSION_PATH, "w") as f:
            for line in results:
                f.write(line + "\n")

        print(f"Submission saved to {self.config.SUBMISSION_PATH}")

    def run(self):
        print(f"Starting training on device: {self.device}")

        for epoch in range(self.config.NUM_EPOCHS):
            start_time = time.time()

            train_loss = self.train_epoch(epoch)
            val_score = self.validate(self.val_loader)

            duration = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{self.config.NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Levenshtein: {val_score:.6f} | "
                f"Time: {duration:.1f}s"
            )

            # Checkpoint
            if val_score < self.best_val_score:
                self.best_val_score = val_score
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
                print(f"  New best model saved! Score: {self.best_val_score:.6f}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early Stopping
            if self.patience_counter >= self.early_stopping_patience:
                print("Early stopping triggered.")
                break

        # Load best model for submission
        if os.path.exists(self.config.MODEL_SAVE_PATH):
            self.model.load_state_dict(
                torch.load(self.config.MODEL_SAVE_PATH, map_location=self.device)
            )

        self.generate_submission()
