import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from library.utils import (
    set_seed,
    compute_normalized_levenshtein_score,
    decode_predictions_to_sequence,
)
from library.data_loader import GestureDataset

# ==========================================
# Configuration & Constants
# ==========================================
NUM_CLASSES = 21  # 0 (Background) + 20 Gestures
INPUT_DIM = 193  # 20 joints * 3 coords * 3 kinematics + 13 MFCCs
HIDDEN_DIM = 64
WINDOW_SIZE = 64
STRIDE = 32  # 50% overlap
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_DIR = "./working/cache"
SUBMISSION_DIR = "./submission"
BEST_MODEL_PATH = "./working/best_model.pth"

# ==========================================
# Model Architecture
# ==========================================


class GatedTCNBlock(nn.Module):
    """
    Gated Dilated Temporal Convolutional Block.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        # Calculate padding to maintain temporal length (assuming centered/same)
        padding = (kernel_size - 1) * dilation // 2

        self.conv_filter = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=padding
        )
        self.conv_gate = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=padding
        )

        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.downsample = None

    def forward(self, x):
        # Filter (tanh) and Gate (sigmoid) branches
        f = torch.tanh(self.conv_filter(x))
        g = torch.sigmoid(self.conv_gate(x))
        out = f * g

        out = self.dropout(out)

        # Residual Connection
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class BiGRUEncoder(nn.Module):
    """
    Stage 1: Kinematic Sequence Encoder using Bi-GRU.
    """

    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x: (B, T, D)
        self.gru.flatten_parameters()
        out, _ = self.gru(x)
        logits = self.fc(out)  # (B, T, C)
        # Permute to (B, C, T) for consistency with TCNs
        return logits.permute(0, 2, 1)


class RefinementStage(nn.Module):
    """
    Stage 2 & 3: Attentive Gated Refinement Module.
    Takes class probabilities as input.
    """

    def __init__(self, num_classes, hidden_dim, num_layers=5):
        super().__init__()
        # Input projection: Probabilities (C) -> Hidden (H)
        self.conv_in = nn.Conv1d(num_classes, hidden_dim, 1)

        layers = []
        for i in range(num_layers):
            dilation = 2**i  # 1, 2, 4, 8, 16
            layers.append(
                GatedTCNBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=dilation)
            )
        self.layers = nn.Sequential(*layers)

        # Output projection: Hidden (H) -> Logits (C)
        self.conv_out = nn.Conv1d(hidden_dim, num_classes, 1)

    def forward(self, probs):
        # probs: (B, C, T)
        x = self.conv_in(probs)
        x = self.layers(x)
        logits = self.conv_out(x)
        return logits


class AKCIRN(nn.Module):
    """
    Kinematically-Consistent Iterative Refinement Network (KC-IRN).
    Three-stage cascaded architecture.
    """

    def __init__(
        self, input_dim=INPUT_DIM, num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM
    ):
        super().__init__()
        self.stage1 = BiGRUEncoder(input_dim, hidden_dim, num_classes)
        self.stage2 = RefinementStage(num_classes, hidden_dim)
        self.stage3 = RefinementStage(num_classes, hidden_dim)

    def forward(self, x):
        # x: (B, T, D)

        # Stage 1
        logits1 = self.stage1(x)  # (B, C, T)
        probs1 = F.softmax(logits1, dim=1)

        # Stage 2 (Input: Strictly probabilities from Stage 1)
        logits2 = self.stage2(probs1)
        probs2 = F.softmax(logits2, dim=1)

        # Stage 3 (Input: Strictly probabilities from Stage 2, independent weights)
        logits3 = self.stage3(probs2)

        return logits1, logits2, logits3


# ==========================================
# Loss Function
# ==========================================


class CascadedSmoothLoss(nn.Module):
    """
    Weighted Cross-Entropy + Log-Space Temporal Smoothing.
    """

    def __init__(self, num_classes, background_weight=0.2, smooth_threshold=1.0):
        super().__init__()
        # Class weights: Downweight background (class 0)
        weights = torch.ones(num_classes)
        weights[0] = background_weight
        self.ce = nn.CrossEntropyLoss(weight=weights)
        self.smooth_threshold = smooth_threshold

    def smooth_loss(self, logits):
        # logits: (B, C, T)
        log_probs = F.log_softmax(logits, dim=1)
        # Difference between adjacent frames
        diff = log_probs[:, :, 1:] - log_probs[:, :, :-1]
        mse = diff.pow(2)
        # Truncated MSE
        mse = torch.clamp(mse, max=self.smooth_threshold)
        return mse.mean()

    def forward(self, l1, l2, l3, targets):
        # targets: (B, T)
        loss1 = self.ce(l1, targets)

        loss2_ce = self.ce(l2, targets)
        loss2_sm = self.smooth_loss(l2)
        loss2 = loss2_ce + loss2_sm

        loss3_ce = self.ce(l3, targets)
        loss3_sm = self.smooth_loss(l3)
        loss3 = loss3_ce + loss3_sm

        return loss1 + loss2 + loss3


# ==========================================
# Training & Inference Logic
# ==========================================


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        l1, l2, l3 = model(features)

        loss = criterion(l1, l2, l3, labels)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device, samples_dict):
    """
    Reconstructs sequences from sliding windows and computes Levenshtein score.
    """
    model.eval()
    dataset = loader.dataset

    # Initialize buffers for sequence reconstruction
    sample_probs = {}
    sample_counts = {}

    # Pre-allocate buffers based on dataset info
    for s in dataset.samples:
        sid = s["sample_id"]
        t = s["skeleton"].shape[0]
        sample_probs[sid] = np.zeros((t, NUM_CLASSES), dtype=np.float32)
        sample_counts[sid] = np.zeros((t,), dtype=np.float32)

    with torch.no_grad():
        # Iterate over loader
        for i, (features, _) in enumerate(loader):
            features = features.to(device)
            # Use Stage 3 output
            _, _, l3 = model(features)
            probs = F.softmax(l3, dim=1).cpu().numpy()  # (B, C, T)

            # Map batch items back to sample/frame
            batch_size = features.size(0)
            start_idx = i * loader.batch_size

            for b in range(batch_size):
                global_idx = start_idx + b
                if global_idx >= len(dataset.windows):
                    break

                sample_idx, start_frame = dataset.windows[global_idx]
                sample_data = dataset.samples[sample_idx]
                sid = sample_data["sample_id"]

                # Transpose to (T, C)
                p = probs[b].transpose(1, 0)

                # Determine valid length (handle padding at edges)
                actual_len = sample_data["skeleton"].shape[0]
                valid_len = min(WINDOW_SIZE, actual_len - start_frame)

                if valid_len > 0:
                    sample_probs[sid][start_frame : start_frame + valid_len] += p[
                        :valid_len
                    ]
                    sample_counts[sid][start_frame : start_frame + valid_len] += 1.0

    # Decode predictions
    predictions = {}
    for sid, prob_sum in sample_probs.items():
        counts = sample_counts[sid]
        counts[counts == 0] = 1.0  # Avoid div by zero
        avg_probs = prob_sum / counts[:, None]

        pred_labels = np.argmax(avg_probs, axis=1)
        seq = decode_predictions_to_sequence(pred_labels)
        predictions[sid] = seq

    score = compute_normalized_levenshtein_score(predictions, samples_dict)
    return score


def run_experiment():
    set_seed(42)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Metadata Paths
    train_meta = "./metadata/train.csv"
    val_meta = "./metadata/val.csv"
    test_meta = "./metadata/test.csv"

    # Datasets
    print("Loading datasets...")
    # Train: Augmentation enabled, Stride 32 (50% overlap)
    train_ds = GestureDataset(
        train_meta,
        split="train",
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        cache_dir=CACHE_DIR,
        augment=True,
    )
    # Val: No augmentation, Stride 32
    val_ds = GestureDataset(
        val_meta,
        split="val",
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        cache_dir=CACHE_DIR,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    # Prepare validation targets
    val_targets = {}
    for s in val_ds.samples:
        seq = decode_predictions_to_sequence(s["labels"])
        val_targets[s["sample_id"]] = seq

    # Model Setup
    model = AKCIRN(
        input_dim=INPUT_DIM, num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = CascadedSmoothLoss(
        NUM_CLASSES, background_weight=0.2, smooth_threshold=1.0
    ).to(DEVICE)

    # Training Loop
    best_score = float("inf")
    patience = 8
    counter = 0

    print("Starting training...")
    for epoch in range(50):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_score = validate(model, val_loader, DEVICE, val_targets)

        print(f"Epoch {epoch+1}: Loss={train_loss}, Val Score={val_score}")

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Best Validation Score: {best_score}")

    # ==========================================
    # Submission Generation
    # ==========================================
    print("Generating submission...")

    # Load Best Model
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()

    test_ds = GestureDataset(
        test_meta,
        split="test",
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        cache_dir=CACHE_DIR,
        augment=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    # Inference Buffers
    test_probs = {}
    test_counts = {}
    for s in test_ds.samples:
        sid = s["sample_id"]
        t = s["skeleton"].shape[0]
        test_probs[sid] = np.zeros((t, NUM_CLASSES), dtype=np.float32)
        test_counts[sid] = np.zeros((t,), dtype=np.float32)

    with torch.no_grad():
        for i, (features, _) in enumerate(test_loader):
            features = features.to(DEVICE)
            _, _, l3 = model(features)
            probs = F.softmax(l3, dim=1).cpu().numpy()

            batch_size = features.size(0)
            start_idx = i * test_loader.batch_size

            for b in range(batch_size):
                global_idx = start_idx + b
                if global_idx >= len(test_ds.windows):
                    break

                sample_idx, start_frame = test_ds.windows[global_idx]
                sample_data = test_ds.samples[sample_idx]
                sid = sample_data["sample_id"]

                p = probs[b].transpose(1, 0)
                actual_len = sample_data["skeleton"].shape[0]
                valid_len = min(WINDOW_SIZE, actual_len - start_frame)

                if valid_len > 0:
                    test_probs[sid][start_frame : start_frame + valid_len] += p[
                        :valid_len
                    ]
                    test_counts[sid][start_frame : start_frame + valid_len] += 1.0

    # Decode and Save
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    with open(submission_path, "w") as f:
        for s in test_ds.samples:
            sid = s["sample_id"]
            counts = test_counts[sid]
            counts[counts == 0] = 1.0
            avg_probs = test_probs[sid] / counts[:, None]

            pred_labels = np.argmax(avg_probs, axis=1)
            seq = decode_predictions_to_sequence(pred_labels)

            seq_str = ",".join(map(str, seq))
            f.write(f"{sid},{seq_str}\n")

    print(f"Submission saved to {submission_path}")


# Execute Pipeline
run_experiment()
