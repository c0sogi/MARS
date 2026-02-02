import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pandas as pd
import numpy as np
import os
import math

from library.config import Config
from library.utils import AverageMeter, kl_divergence_loss
from library.data import get_dataloaders

# =========================================================================
# Model Components
# =========================================================================


class Inception1D(nn.Module):
    """
    Multi-Scale 1D Convolutional Block.
    Applies parallel convolutions with different kernel sizes to extract
    frequency-specific features.
    """

    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7, 9]):
        super().__init__()
        self.branches = nn.ModuleList()

        # Calculate channels per branch
        self.branch_channels = out_channels // len(kernel_sizes)
        remainder = out_channels % len(kernel_sizes)

        for i, k in enumerate(kernel_sizes):
            # Distribute remainder channels to the first few branches
            out_c = self.branch_channels + (1 if i < remainder else 0)

            # Padding to maintain temporal dimension
            pad = k // 2

            branch = nn.Sequential(
                nn.Conv1d(in_channels, out_c, kernel_size=k, padding=pad, bias=False),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True),
            )
            self.branches.append(branch)

    def forward(self, x):
        # x shape: (Batch, Channels, Time)
        outputs = [branch(x) for branch in self.branches]
        return torch.cat(outputs, dim=1)


class EEGEncoder(nn.Module):
    """
    Stream A: Processes raw EEG signals using Inception1D blocks.
    Preserves temporal sequence for attention.
    """

    def __init__(self, in_channels=20, base_filters=64, out_dim=256):
        super().__init__()

        # Initial projection
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=1, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
        )

        # Encoder Stages
        # We want to downsample time but keep it as a sequence.
        # Input length: 5000.

        self.layer1 = nn.Sequential(
            Inception1D(base_filters, base_filters * 2, Config.EEG_KERNEL_SIZES),
            nn.MaxPool1d(kernel_size=4),  # 5000 -> 1250
        )

        self.layer2 = nn.Sequential(
            Inception1D(base_filters * 2, base_filters * 4, Config.EEG_KERNEL_SIZES),
            nn.MaxPool1d(kernel_size=4),  # 1250 -> 312
        )

        self.layer3 = nn.Sequential(
            Inception1D(base_filters * 4, out_dim, Config.EEG_KERNEL_SIZES),
            nn.MaxPool1d(kernel_size=4),  # 312 -> 78
        )

        self.layer4 = nn.Sequential(
            Inception1D(out_dim, out_dim, Config.EEG_KERNEL_SIZES),
            nn.MaxPool1d(kernel_size=2),  # 78 -> 39
        )

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: (B, 20, 5000)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dropout(x)
        # Output: (B, out_dim, 39)
        return x


class SpecEncoder(nn.Module):
    """
    Stream B: Processes Spectrograms using EfficientNet-B0.
    """

    def __init__(self, out_dim=256, pretrained=True):
        super().__init__()

        # Load EfficientNet backbone
        # num_classes=0 removes the classifier head
        self.backbone = timm.create_model(
            Config.SPEC_BACKBONE, pretrained=pretrained, num_classes=0, in_chans=3
        )

        # Get feature dimension of the backbone (1280 for EffNet-B0)
        self.in_features = self.backbone.num_features

        # Projection to match EEG embedding dimension
        self.proj = nn.Conv2d(self.in_features, out_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, 3, 512, 512)

        # Extract features
        # timm forward_features returns (B, C, H, W)
        x = self.backbone.forward_features(x)  # (B, 1280, 16, 16)

        # Project channels
        x = self.proj(x)  # (B, 256, 16, 16)

        # Flatten spatial dimensions to form a sequence
        B, C, H, W = x.shape
        x = x.view(B, C, H * W)  # (B, 256, 256)

        return x


class CrossAttentionFusion(nn.Module):
    """
    Attentive Fusion Module.
    Uses EEG features as Queries and Spectrogram features as Keys/Values.
    """

    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()

        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, eeg_feats, spec_feats):
        # eeg_feats: (B, Dim, T_eeg)
        # spec_feats: (B, Dim, S_spec)

        # Permute for MultiheadAttention: (Seq_Len, Batch, Dim)
        q = eeg_feats.permute(2, 0, 1)  # (T_eeg, B, Dim)
        k = spec_feats.permute(2, 0, 1)  # (S_spec, B, Dim)
        v = spec_feats.permute(2, 0, 1)  # (S_spec, B, Dim)

        # Attention
        # attn_output: (T_eeg, B, Dim)
        attn_output, _ = self.mha(q, k, v)

        # Residual Connection + Norm
        # We add the attention output to the original EEG query features
        output = self.norm(q + self.dropout(attn_output))

        # Permute back to (B, Dim, T_eeg)
        output = output.permute(1, 2, 0)

        return output


class DualStreamNetwork(nn.Module):
    """
    Spectrogram-Guided Attentive Dual-Stream Network.
    """

    def __init__(self):
        super().__init__()

        self.eeg_encoder = EEGEncoder(
            in_channels=Config.EEG_CHANNELS,
            base_filters=Config.EEG_BASE_FILTERS,
            out_dim=Config.ATTENTION_DIM,
        )

        self.spec_encoder = SpecEncoder(
            out_dim=Config.ATTENTION_DIM, pretrained=Config.PRETRAINED
        )

        self.fusion = CrossAttentionFusion(
            embed_dim=Config.ATTENTION_DIM, num_heads=8, dropout=0.2
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(Config.ATTENTION_DIM, 128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(128, Config.NUM_CLASSES),
        )

    def forward(self, inputs):
        eeg_data, spec_data = inputs

        # 1. Encode Streams
        eeg_feats = self.eeg_encoder(eeg_data)  # (B, 256, 39)
        spec_feats = self.spec_encoder(spec_data)  # (B, 256, 256)

        # 2. Attentive Fusion
        # Refines EEG features with global spectrogram context
        fused_feats = self.fusion(eeg_feats, spec_feats)  # (B, 256, 39)

        # 3. Pooling
        pooled = self.global_pool(fused_feats).squeeze(-1)  # (B, 256)

        # 4. Classification
        logits = self.classifier(pooled)

        return logits


# =========================================================================
# Training & Inference Logic
# =========================================================================


def train_model():
    """
    Executes the training pipeline.
    """
    # Setup
    device = torch.device(Config.DEVICE)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # DataLoaders
    train_loader, val_loader, _ = get_dataloaders()

    # Model
    model = DualStreamNetwork().to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LR,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # Mixed Precision
    scaler = (
        torch.amp.GradScaler("cuda")
        if Config.MIXED_PRECISION and device.type == "cuda"
        else None
    )

    # Tracking
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        # --- Training ---
        model.train()
        train_loss_meter = AverageMeter()

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            eeg, spec = inputs
            eeg, spec = eeg.to(device), spec.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocore("cuda"):
                    logits = model((eeg, spec))
                    loss = kl_divergence_loss(logits, targets)

                scaler.scale(loss).backward()

                # Gradient Clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model((eeg, spec))
                loss = kl_divergence_loss(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
                optimizer.step()

            scheduler.step()
            train_loss_meter.update(loss.item(), eeg.size(0))

        # --- Validation ---
        model.eval()
        val_loss_meter = AverageMeter()

        with torch.no_grad():
            for inputs, targets in val_loader:
                eeg, spec = inputs
                eeg, spec = eeg.to(device), spec.to(device)
                targets = targets.to(device)

                if scaler:
                    with torch.amp.autocast("cuda"):
                        logits = model((eeg, spec))
                        loss = kl_divergence_loss(logits, targets)
                else:
                    logits = model((eeg, spec))
                    loss = kl_divergence_loss(logits, targets)

                val_loss_meter.update(loss.item(), eeg.size(0))

        # --- Logging & Saving ---
        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"Train Loss: {train_loss_meter.avg}")
        print(f"Val Loss: {val_loss_meter.avg}")

        if val_loss_meter.avg < best_val_loss:
            best_val_loss = val_loss_meter.avg
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print("Training complete.")


def inference():
    """
    Generates predictions for the test set and saves the submission file.
    """
    device = torch.device(Config.DEVICE)
    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        print(f"Error: Model file not found at {best_model_path}")
        return

    # Load Data
    _, _, test_loader = get_dataloaders()
    test_df = pd.read_csv(Config.TEST_CSV)

    # Load Model
    model = DualStreamNetwork().to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    print("Starting inference...")

    all_probs = []

    with torch.no_grad():
        for inputs in test_loader:
            eeg, spec = inputs
            eeg, spec = eeg.to(device), spec.to(device)

            # Forward pass
            logits = model((eeg, spec))

            # Apply Softmax to get probabilities
            probs = F.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())

    # Concatenate predictions
    predictions = np.concatenate(all_probs, axis=0)

    # Ensure probabilities sum to 1 (Softmax guarantees this, but good to be safe)
    # predictions = predictions / predictions.sum(axis=1, keepdims=True)

    # Create Submission DataFrame
    submission = pd.DataFrame(predictions, columns=Config.CLASS_NAMES)
    submission.insert(0, "eeg_id", test_df["eeg_id"])

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission.head())
