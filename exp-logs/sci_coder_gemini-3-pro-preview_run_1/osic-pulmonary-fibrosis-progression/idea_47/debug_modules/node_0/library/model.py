import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pandas as pd
import numpy as np

from library.config import Config, seed_everything
from library.data import get_dataloaders
from library.utils import laplace_log_likelihood_metric

# ==========================================
# 1. Model Architecture Components
# ==========================================


class VisualBackbone(nn.Module):
    """
    Independent Low-Capacity Visual Backbone.
    Uses EfficientNet-B0 initialized with ImageNet weights.
    Extracts high-fidelity features without down-projection.
    """

    def __init__(self, model_name="efficientnet_b0", pretrained=True):
        super().__init__()
        # Load EfficientNet-B0.
        # num_classes=0 and global_pool='avg' ensures we get the feature vector directly.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        # EfficientNet-B0 output dimension is 1280
        self.out_dim = 1280

    def forward(self, x):
        # Input: (B, 3, 224, 224) -> Output: (B, 1280)
        return self.backbone(x)


class TabularEncoder(nn.Module):
    """
    Shared-Latent Tabular Encoder.
    Processes raw metadata into a robust Shared Latent Vector (T_lat).
    """

    def __init__(self, input_dim, latent_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256), nn.GELU(), nn.Linear(256, latent_dim), nn.GELU()
        )

    def forward(self, x):
        # Input: (B, num_features) -> Output: (B, 128)
        return self.mlp(x)


class SLHDAN(nn.Module):
    """
    Shared-Latent Holistic Dual-Axis Network (SLH-DAN).
    Integrates independent visual backbones with a shared latent tabular topology
    via a bifurcated flow and pre-norm symmetric attention.
    """

    def __init__(self, tabular_input_dim):
        super().__init__()

        # 1. Independent Visual Backbones
        self.backbone_axial = VisualBackbone()
        self.backbone_coronal = VisualBackbone()

        # 2. Shared-Latent Tabular Encoder
        self.tabular_encoder = TabularEncoder(
            tabular_input_dim, latent_dim=Config.LATENT_DIM
        )

        # 3. Bifurcated Flow Projections
        # Flow A: Align T_lat for fusion (128 -> 1280)
        self.lat_to_align = nn.Linear(Config.LATENT_DIM, Config.BACKBONE_OUT_DIM)

        # 4. Contextualization (Pre-Norm Symmetric Attention)
        # Token dim is 1280 (matching backbone output)
        self.norm_tokens = nn.LayerNorm(Config.BACKBONE_OUT_DIM)
        self.attention = nn.MultiheadAttention(
            embed_dim=Config.BACKBONE_OUT_DIM,
            num_heads=8,
            batch_first=True,
            dropout=0.1,
        )

        # FFN Block (Standard Transformer component for capacity)
        self.norm_ffn = nn.LayerNorm(Config.BACKBONE_OUT_DIM)
        self.ffn = nn.Sequential(
            nn.Linear(Config.BACKBONE_OUT_DIM, Config.BACKBONE_OUT_DIM * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(Config.BACKBONE_OUT_DIM * 4, Config.BACKBONE_OUT_DIM),
            nn.Dropout(0.1),
        )

        # 5. Balanced Prior-Anchored Head
        # Concatenates Holistic Fused Vector (1280) + Shared Latent Vector (128)
        # Input dim: 1280 + 128 = 1408
        head_input_dim = Config.BACKBONE_OUT_DIM + Config.LATENT_DIM
        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 3),  # Outputs: alpha, sigma_base, sigma_growth
        )

    def forward(self, img_ax, img_cor, tab):
        # --- Visual Encoding ---
        v_ax = self.backbone_axial(img_ax)  # (B, 1280)
        v_cor = self.backbone_coronal(img_cor)  # (B, 1280)

        # --- Tabular Encoding ---
        t_lat = self.tabular_encoder(tab)  # (B, 128) -> Shared Latent Vector

        # --- Bifurcated Flow ---
        # Flow A: Alignment for visual fusion
        t_align = self.lat_to_align(t_lat)  # (B, 1280)

        # --- Contextualization ---
        # Stack tokens: [Axial, Coronal, Aligned_Tabular]
        tokens = torch.stack([v_ax, v_cor, t_align], dim=1)  # (B, 3, 1280)

        # Pre-Norm Attention
        tokens_norm = self.norm_tokens(tokens)
        attn_out, _ = self.attention(tokens_norm, tokens_norm, tokens_norm)
        tokens = tokens + attn_out

        # FFN
        tokens_norm2 = self.norm_ffn(tokens)
        ffn_out = self.ffn(tokens_norm2)
        tokens = tokens + ffn_out

        # Holistic Readout (Global Average Pooling across all updated tokens)
        h_fused = torch.mean(tokens, dim=1)  # (B, 1280)

        # --- Prediction Head ---
        # Flow B: Skip connection of raw Shared Latent Vector
        combined = torch.cat([h_fused, t_lat], dim=1)  # (B, 1408)

        out = self.head(combined)

        # Extract parameters
        alpha = out[:, 0]
        # Enforce positivity for confidence parameters using Softplus
        sigma_base = F.softplus(out[:, 1])
        sigma_growth = F.softplus(out[:, 2])

        return alpha, sigma_base, sigma_growth


# ==========================================
# 2. Execution Logic
# ==========================================


def run_training():
    """
    Executes the training loop, validation, and submission generation.
    """
    # Setup
    device = Config.DEVICE
    seed_everything(Config.SEED)

    # Create submission directory if it doesn't exist
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # Determine tabular input dimension dynamically
    sample_batch = next(iter(train_loader))
    tab_dim = sample_batch["tabular"].shape[1]
    print(f"Tabular Feature Dimension: {tab_dim}")

    # Initialize Model
    model = SLHDAN(tabular_input_dim=tab_dim).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # Training State
    best_metric = -float("inf")
    patience_counter = 0

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # --- TRAIN LOOP ---
        model.train()
        train_losses = []

        for batch in train_loader:
            # Move inputs to device
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tab = batch["tabular"].to(device)

            # Metadata for Parametric Head
            week = batch["week"].to(device)
            baseline_week = batch["baseline_week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            target_fvc = batch["target"].to(device)

            optimizer.zero_grad()

            # Forward Pass
            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tab)

            # Parametric Inference: FVC = Base + alpha * delta_t
            delta_t = week - baseline_week
            fvc_pred = baseline_fvc + alpha * delta_t

            # Confidence Inference: Sigma = Base + Growth * |delta_t|
            confidence = sigma_base + sigma_growth * torch.abs(delta_t)

            # Loss Calculation
            # Metric returns negative value (higher is better).
            # We minimize Loss = -Metric.
            score = laplace_log_likelihood_metric(target_fvc, fvc_pred, confidence)
            loss = -score

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # --- VALIDATION LOOP ---
        model.eval()
        val_scores = []

        with torch.no_grad():
            for batch in val_loader:
                img_ax = batch["image_axial"].to(device)
                img_cor = batch["image_coronal"].to(device)
                tab = batch["tabular"].to(device)

                week = batch["week"].to(device)
                baseline_week = batch["baseline_week"].to(device)
                baseline_fvc = batch["baseline_fvc"].to(device)
                target_fvc = batch["target"].to(device)

                alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tab)

                delta_t = week - baseline_week
                fvc_pred = baseline_fvc + alpha * delta_t
                confidence = sigma_base + sigma_growth * torch.abs(delta_t)

                score = laplace_log_likelihood_metric(target_fvc, fvc_pred, confidence)
                val_scores.append(score.item())

        # Metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_score = np.mean(val_scores)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Score: {avg_val_score:.6f}"
        )

        # Early Stopping & Checkpointing
        if avg_val_score > best_metric:
            best_metric = avg_val_score
            torch.save(model.state_dict(), Config.MODEL_PATH)
            patience_counter = 0
            print(f"  New best model saved! Score: {best_metric:.6f}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # --- SUBMISSION GENERATION ---
    print("Generating submission...")

    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    results = []

    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tab = batch["tabular"].to(device)

            week = batch["week"].to(device)
            baseline_week = batch["baseline_week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            patient_ids = batch["patient_id"]

            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tab)

            delta_t = week - baseline_week
            fvc_pred = baseline_fvc + alpha * delta_t
            confidence = sigma_base + sigma_growth * torch.abs(delta_t)

            # Clip confidence for final submission (Metric logic does this, but we need it in CSV)
            confidence = torch.clamp(confidence, min=Config.MIN_CONFIDENCE)

            # Move to CPU
            fvc_pred = fvc_pred.cpu().numpy()
            confidence = confidence.cpu().numpy()
            week_cpu = week.cpu().numpy()

            for i in range(len(patient_ids)):
                pid = patient_ids[i]
                w = int(week_cpu[i])
                patient_week = f"{pid}_{w}"

                results.append(
                    {
                        "Patient_Week": patient_week,
                        "FVC": fvc_pred[i],
                        "Confidence": confidence[i],
                    }
                )

    # Format Submission
    submission_df = pd.DataFrame(results)

    # Load sample submission to ensure correct row order and completeness
    sample = pd.read_csv(Config.SAMPLE_SUBMISSION)
    final_sub = pd.merge(
        sample[["Patient_Week"]], submission_df, on="Patient_Week", how="left"
    )

    # Fill any missing values with defaults (though overlap should be exact)
    final_sub["FVC"] = final_sub["FVC"].fillna(2000)
    final_sub["Confidence"] = final_sub["Confidence"].fillna(100)

    final_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
