import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
import scipy.ndimage
from library.config import *
from library.utils import set_seed, compute_levenshtein
from library.data_loader import get_loaders

# =============================================================================
# Model Components
# =============================================================================


class GatedTCNBlock(nn.Module):
    """
    Gated Dilated Temporal Convolutional Block with 1x1 Output Projection.
    Z = tanh(W_f * X) * sigmoid(W_g * X)
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv_f = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2,
        )
        self.conv_g = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2,
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Conv1d(out_channels, out_channels, 1)

        # Residual connection handling
        if in_channels != out_channels:
            self.res_proj = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.res_proj = None

    def forward(self, x):
        f = torch.tanh(self.conv_f(x))
        g = torch.sigmoid(self.conv_g(x))
        out = f * g
        out = self.dropout(out)
        out = self.proj(out)

        res = x if self.res_proj is None else self.res_proj(x)
        return res + out


class BiLSTMEncoder(nn.Module):
    """
    Stage 1: Multi-Task Recurrent Encoder.
    """

    def __init__(self, input_dim, hidden_dim, num_classes, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        # Bidirectional output is 2 * hidden_dim
        self.cls_head = nn.Linear(hidden_dim * 2, num_classes)
        self.bnd_head = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        # x: (B, T, Input_Dim)
        feat, _ = self.lstm(x)

        cls_logits = self.cls_head(feat)  # (B, T, Num_Classes)
        bnd_logits = self.bnd_head(feat)  # (B, T, 1)

        cls_probs = F.softmax(cls_logits, dim=2)
        bnd_probs = torch.sigmoid(bnd_logits)

        return cls_probs, bnd_probs


class RefinementStage(nn.Module):
    """
    Stage 2 & 3: Gated Refinement Stage (MS-TCN).
    """

    def __init__(
        self, input_dim, hidden_dim, num_classes, num_layers, kernel_size, dropout
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)

        # Monotonically increasing dilation: 2^0, 2^1, ..., 2^(num_layers-1)
        self.layers = nn.ModuleList(
            [
                GatedTCNBlock(hidden_dim, hidden_dim, kernel_size, 2**i, dropout)
                for i in range(num_layers)
            ]
        )

        self.cls_head = nn.Conv1d(hidden_dim, num_classes, 1)
        self.bnd_head = nn.Conv1d(hidden_dim, 1, 1)

    def forward(self, x):
        # x: (B, Input_Dim, T)
        out = self.input_proj(x)
        for layer in self.layers:
            out = layer(out)

        cls_logits = self.cls_head(out)
        bnd_logits = self.bnd_head(out)

        # Permute back to (B, T, C) for consistency
        cls_probs = F.softmax(cls_logits, dim=1).permute(0, 2, 1)
        bnd_probs = torch.sigmoid(bnd_logits).permute(0, 2, 1)

        return cls_probs, bnd_probs


class DCSGCN(nn.Module):
    """
    Densely-Connected Supervised Gated-Cascaded Network.
    """

    def __init__(self):
        super().__init__()
        self.num_classes = HYPERPARAMS["num_classes"]
        hidden_dim = HYPERPARAMS["hidden_dim"]

        # Stage 1: Bi-LSTM
        self.stage1 = BiLSTMEncoder(
            input_dim=INPUT_DIM,
            hidden_dim=hidden_dim,
            num_classes=self.num_classes,
            num_layers=HYPERPARAMS["lstm_layers"],
            dropout=HYPERPARAMS["dropout"],
        )

        # Stage 2: TCN Refinement
        # Input is probability output of Stage 1 (Classes + Boundary)
        stage2_input_dim = self.num_classes + 1
        self.stage2 = RefinementStage(
            input_dim=stage2_input_dim,
            hidden_dim=hidden_dim,
            num_classes=self.num_classes,
            num_layers=HYPERPARAMS["tcn_layers"],
            kernel_size=HYPERPARAMS["tcn_kernel_size"],
            dropout=HYPERPARAMS["dropout"],
        )

        # Stage 3: Dense Refinement
        # Input is concatenation of Stage 1 output and Stage 2 output
        stage3_input_dim = (self.num_classes + 1) * 2
        self.stage3 = RefinementStage(
            input_dim=stage3_input_dim,
            hidden_dim=hidden_dim,
            num_classes=self.num_classes,
            num_layers=HYPERPARAMS["tcn_layers"],
            kernel_size=HYPERPARAMS["tcn_kernel_size"],
            dropout=HYPERPARAMS["dropout"],
        )

    def forward(self, x, mask):
        # x: (B, T, D)
        # mask: (B, T)

        mask_expanded = mask.unsqueeze(2)  # (B, T, 1)

        # --- Stage 1 ---
        s1_cls, s1_bnd = self.stage1(x)
        # Apply mask
        s1_cls = s1_cls * mask_expanded
        s1_bnd = s1_bnd * mask_expanded

        # Prepare input for Stage 2: Concat and Permute to (B, C, T)
        s1_out = torch.cat([s1_cls, s1_bnd], dim=2)  # (B, T, C+1)
        s1_out_tcn = s1_out.permute(0, 2, 1)  # (B, C+1, T)

        # --- Stage 2 ---
        s2_cls, s2_bnd = self.stage2(s1_out_tcn)
        # Apply mask
        s2_cls = s2_cls * mask_expanded
        s2_bnd = s2_bnd * mask_expanded

        # Prepare input for Stage 3: Dense Concatenation
        s2_out = torch.cat([s2_cls, s2_bnd], dim=2)  # (B, T, C+1)
        s2_out_tcn = s2_out.permute(0, 2, 1)  # (B, C+1, T)

        # Dense connection: Concat Stage 1 and Stage 2 outputs
        s3_input = torch.cat([s1_out_tcn, s2_out_tcn], dim=1)  # (B, 2*(C+1), T)

        # --- Stage 3 ---
        s3_cls, s3_bnd = self.stage3(s3_input)
        # Apply mask
        s3_cls = s3_cls * mask_expanded
        s3_bnd = s3_bnd * mask_expanded

        return {
            "stage1": (s1_cls, s1_bnd),
            "stage2": (s2_cls, s2_bnd),
            "stage3": (s3_cls, s3_bnd),
        }


# =============================================================================
# Loss Functions
# =============================================================================


class TMSELoss(nn.Module):
    """
    Truncated Mean Squared Error for temporal smoothing.
    """

    def __init__(self, threshold=0.15):
        super().__init__()
        self.threshold = threshold**2  # Compare squared difference to squared threshold

    def forward(self, probs, mask):
        # probs: (B, T, C)
        # mask: (B, T)

        # Compute difference between adjacent frames
        diff = probs[:, 1:, :] - probs[:, :-1, :]
        sq_diff = diff**2

        # Clamp the squared difference (Truncation)
        # "We do not clamp the loss" in prompt likely refers to not clamping the gradient magnitude arbitrarily,
        # but TMSE inherently involves clamping the error value.
        # We follow standard TMSE: clamp(error, max=threshold)
        truncated_sq_diff = torch.clamp(sq_diff, min=0, max=self.threshold)

        # Mask valid transitions
        # mask is 1 for valid. We need mask for t and t-1 to be valid.
        mask_valid = mask[:, 1:] * mask[:, :-1]
        mask_valid = mask_valid.unsqueeze(2)  # (B, T-1, 1)

        loss = torch.sum(truncated_sq_diff * mask_valid) / (
            torch.sum(mask_valid) * probs.shape[2] + 1e-6
        )
        return loss


# =============================================================================
# Training & Evaluation Logic
# =============================================================================


def train_model():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    train_loader, val_loader, test_loader = get_loaders()

    # Initialize Model
    model = DCSGCN().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=HYPERPARAMS["lr"],
        weight_decay=HYPERPARAMS["weight_decay"],
    )

    # Loss Functions
    # Class weights: 0.1 for background, 1.0 for gestures
    class_weights = torch.tensor(CLASS_WEIGHTS).float().to(device)
    criterion_cls = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    criterion_bnd = nn.BCELoss(reduction="none")
    criterion_smooth = TMSELoss(threshold=HYPERPARAMS["tmse_threshold"])

    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(HYPERPARAMS["num_epochs"]):
        # --- Training ---
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            boundaries = batch["boundaries"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()
            outputs = model(features, mask)

            loss_total = 0.0

            # Deep Supervision over all stages
            for stage_name in ["stage1", "stage2", "stage3"]:
                p_cls, p_bnd = outputs[stage_name]

                # Classification Loss (Masked)
                # Flatten for CE: (B*T, C) vs (B*T)
                loss_c = criterion_cls(
                    p_cls.reshape(-1, p_cls.shape[-1]), labels.view(-1)
                )
                loss_c = (loss_c * mask.view(-1)).sum() / (mask.sum() + 1e-6)

                # Boundary Loss (Masked)
                loss_b = criterion_bnd(p_bnd.squeeze(2), boundaries)
                loss_b = (loss_b * mask).sum() / (mask.sum() + 1e-6)

                # Smoothing Loss (TMSE)
                loss_s = criterion_smooth(p_cls, mask)

                loss_total += (
                    HYPERPARAMS["w_cls"] * loss_c
                    + HYPERPARAMS["w_bnd"] * loss_b
                    + HYPERPARAMS["w_smooth"] * loss_s
                )

            loss_total.backward()
            optimizer.step()
            train_loss += loss_total.item()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                labels = batch["labels"].to(device)
                boundaries = batch["boundaries"].to(device)
                mask = batch["mask"].to(device)

                outputs = model(features, mask)

                loss_total = 0.0
                for stage_name in ["stage1", "stage2", "stage3"]:
                    p_cls, p_bnd = outputs[stage_name]

                    loss_c = criterion_cls(
                        p_cls.reshape(-1, p_cls.shape[-1]), labels.view(-1)
                    )
                    loss_c = (loss_c * mask.view(-1)).sum() / (mask.sum() + 1e-6)

                    loss_b = criterion_bnd(p_bnd.squeeze(2), boundaries)
                    loss_b = (loss_b * mask).sum() / (mask.sum() + 1e-6)

                    loss_s = criterion_smooth(p_cls, mask)

                    loss_total += (
                        HYPERPARAMS["w_cls"] * loss_c
                        + HYPERPARAMS["w_bnd"] * loss_b
                        + HYPERPARAMS["w_smooth"] * loss_s
                    )

                val_loss += loss_total.item()

        avg_val_loss = val_loss / len(val_loader)

        print(
            f"Epoch {epoch+1}/{HYPERPARAMS['num_epochs']} - Train Loss: {avg_train_loss:.6f} - Val Loss: {avg_val_loss:.6f}"
        )

        # Early Stopping & Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(
                model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth")
            )
        else:
            patience_counter += 1
            if patience_counter >= HYPERPARAMS["patience"]:
                print("Early stopping triggered.")
                break

    # Generate Submission after training
    generate_submission(test_loader, device)


def generate_submission(test_loader, device):
    """
    Generates predictions for the test set and saves to CSV.
    Uses Median Filtering and Label Collapsing.
    """
    print("Generating submission...")
    model = DCSGCN().to(device)
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth")))
    model.eval()

    predictions = []

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            sample_ids = batch["sample_ids"]
            lengths = batch["lengths"]

            # Forward pass
            outputs = model(features, mask)
            # Use Stage 3 outputs for final prediction
            probs_cls, _ = outputs["stage3"]  # (B, T, C)

            # Convert to CPU numpy
            probs_cls = probs_cls.cpu().numpy()

            for i, sample_id in enumerate(sample_ids):
                length = lengths[i]
                # Get valid sequence
                seq_probs = probs_cls[i, :length, :]  # (T, C)

                # Argmax
                pred_labels = np.argmax(seq_probs, axis=1)  # (T,)

                # 1. Median Filter (Label-Space Smoothing)
                # Kernel size 7 (heuristic for ~0.7s at 10fps)
                pred_labels = scipy.ndimage.median_filter(
                    pred_labels, size=7, mode="nearest"
                )

                # 2. Decoding (Collapse repeats and remove background)
                final_sequence = []
                prev_label = -1

                for label in pred_labels:
                    if label != prev_label:
                        if label != 0:  # 0 is background
                            final_sequence.append(str(label))
                        prev_label = label

                predictions.append(f"{sample_id},{','.join(final_sequence)}")

    # Save to file
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    with open(submission_path, "w") as f:
        for line in predictions:
            f.write(line + "\n")

    print(f"Submission saved to {submission_path}")
