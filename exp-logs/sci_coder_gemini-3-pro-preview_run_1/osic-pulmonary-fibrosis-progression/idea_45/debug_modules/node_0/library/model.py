import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import AverageMeter, laplace_log_likelihood_metric

# ==========================================
# 1. Model Architecture
# ==========================================


class TabularEncoder(nn.Module):
    def __init__(self, input_dim=7, hidden_dim=64, output_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.mlp(x)


class SLHDAN(nn.Module):
    """
    Shared-Latent Holistic Dual-Axis Network.
    Combines two EfficientNet-B0 backbones with a shared latent tabular encoder
    via a transformer-based fusion mechanism.
    """

    def __init__(self):
        super().__init__()
        # 1. Visual Backbones (EfficientNet-B0)
        # num_classes=0 returns the pooled feature vector (1280 dim)
        self.axial_backbone = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.coronal_backbone = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0, global_pool="avg"
        )

        # 2. Tabular Encoder
        # Input features: Weeks, Percent, Age, Sex, SmokingStatus(3) -> 7
        self.tabular_encoder = TabularEncoder(
            input_dim=7, hidden_dim=64, output_dim=Config.LATENT_DIM
        )

        # 3. Fusion Module
        self.visual_dim = Config.VISUAL_DIM  # 1280
        self.latent_dim = Config.LATENT_DIM  # 128

        # Project latent to visual dim for attention
        self.latent_projector = nn.Linear(self.latent_dim, self.visual_dim)

        # Pre-Norm Attention
        self.norm1 = nn.LayerNorm(self.visual_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.visual_dim, num_heads=4, batch_first=True
        )

        # FFN
        self.norm2 = nn.LayerNorm(self.visual_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.visual_dim, 2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, self.visual_dim),
        )

        # 4. Prediction Head
        # Concatenates H_fused (1280) + T_lat (128) -> 1408
        self.head_input_dim = self.visual_dim + self.latent_dim
        self.head = nn.Sequential(
            nn.Linear(self.head_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # alpha, sigma_base, sigma_growth
        )

    def forward(self, axial, coronal, tabular):
        # Visual Features
        v_ax = self.axial_backbone(axial)  # (B, 1280)
        v_cor = self.coronal_backbone(coronal)  # (B, 1280)

        # Tabular Latent
        t_lat = self.tabular_encoder(tabular)  # (B, 128)

        # Project Latent for Fusion
        t_align = self.latent_projector(t_lat)  # (B, 1280)

        # Stack Tokens: [Axial, Coronal, Tabular]
        tokens = torch.stack([v_ax, v_cor, t_align], dim=1)  # (B, 3, 1280)

        # Attention Block (Pre-Norm)
        x = self.norm1(tokens)
        attn_out, _ = self.attn(x, x, x)
        x = tokens + attn_out

        # FFN Block
        y = self.norm2(x)
        y = self.ffn(y)
        x = x + y

        # Holistic Readout (GAP)
        h_fused = x.mean(dim=1)  # (B, 1280)

        # Concatenate with original latent
        h_final = torch.cat([h_fused, t_lat], dim=1)  # (B, 1408)

        # Prediction
        preds = self.head(h_final)

        alpha = preds[:, 0]
        sigma_base = F.softplus(preds[:, 1])
        sigma_growth = F.softplus(preds[:, 2])

        return alpha, sigma_base, sigma_growth


# ==========================================
# 2. Training Utilities
# ==========================================


def criterion(alpha, sigma_base, sigma_growth, dt, baseline_fvc, target_fvc):
    """
    Calculates the loss based on the negative modified Laplace Log Likelihood.
    """
    # 1. Calculate Predictions
    # FVC = Baseline + alpha * dt
    fvc_pred = baseline_fvc + alpha * dt

    # Confidence = sigma_base + sigma_growth * |dt|
    sigma_pred = sigma_base + sigma_growth * torch.abs(dt)

    # 2. Calculate Metric Terms (Differentiable approximation logic)
    # The metric uses clipped error and clipped sigma.
    # We must replicate this in the loss to optimize for the metric.

    # Delta: |True - Pred| clipped at 1000
    abs_err = torch.abs(target_fvc - fvc_pred)
    delta = torch.clamp(abs_err, max=Config.MAX_ERROR)

    # Sigma Clipped: max(sigma, 70)
    sigma_clipped = torch.clamp(sigma_pred, min=Config.MIN_CONFIDENCE)

    # Metric Formula: - (sqrt(2) * Delta) / sigma - ln(sqrt(2) * sigma)
    # Loss = -Metric = (sqrt(2) * Delta) / sigma + ln(sqrt(2) * sigma)

    sqrt_2 = 1.41421356
    term1 = (sqrt_2 * delta) / sigma_clipped
    term2 = torch.log(sqrt_2 * sigma_clipped)

    loss = torch.mean(term1 + term2)
    return loss, fvc_pred, sigma_pred


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    meter = AverageMeter()

    for batch in loader:
        # Move to device
        axial = batch["axial"].to(device)
        coronal = batch["coronal"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["target"].to(device)
        dt = batch["dt"].to(device)
        baseline = batch["baseline_fvc"].to(device)

        optimizer.zero_grad()

        # Forward
        alpha, sigma_base, sigma_growth = model(axial, coronal, tabular)

        # Loss
        loss, _, _ = criterion(alpha, sigma_base, sigma_growth, dt, baseline, target)

        loss.backward()
        optimizer.step()

        meter.update(loss.item(), axial.size(0))

    return meter.avg


def validate(model, loader, device):
    model.eval()
    meter = AverageMeter()
    # We also want to track the actual competition metric
    metric_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)
            dt = batch["dt"].to(device)
            baseline = batch["baseline_fvc"].to(device)

            alpha, sigma_base, sigma_growth = model(axial, coronal, tabular)

            loss, fvc_pred, sigma_pred = criterion(
                alpha, sigma_base, sigma_growth, dt, baseline, target
            )

            # Calculate actual metric
            metric_val = laplace_log_likelihood_metric(target, fvc_pred, sigma_pred)

            meter.update(loss.item(), axial.size(0))
            metric_meter.update(metric_val, axial.size(0))

    return meter.avg, metric_meter.avg


def train_model(train_loader, val_loader):
    device = Config.DEVICE
    model = SLHDAN().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    best_metric = -float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_metric = validate(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Metric: {val_metric:.6f}"
        )

        # Save Best
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  -> New Best Model Saved! Metric: {best_metric:.6f}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    return model


def predict_and_submit(test_loader):
    device = Config.DEVICE
    model = SLHDAN().to(device)

    # Load best weights
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        print("Loaded best model for inference.")
    else:
        print("Warning: Best model not found. Using current weights.")

    model.eval()

    results = []

    with torch.no_grad():
        for batch in test_loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            dt = batch["dt"].to(device)
            baseline = batch["baseline_fvc"].to(device)
            patient_week_ids = batch["patient_week_id"]

            alpha, sigma_base, sigma_growth = model(axial, coronal, tabular)

            # Calculate predictions
            fvc_pred = baseline + alpha * dt
            sigma_pred = sigma_base + sigma_growth * torch.abs(dt)

            fvc_pred = fvc_pred.cpu().numpy()
            sigma_pred = sigma_pred.cpu().numpy()

            for i in range(len(patient_week_ids)):
                results.append(
                    {
                        "Patient_Week": patient_week_ids[i],
                        "FVC": fvc_pred[i],
                        "Confidence": sigma_pred[i],
                    }
                )

    # Create DataFrame
    sub_df = pd.DataFrame(results)

    # Ensure columns order
    sub_df = sub_df[["Patient_Week", "FVC", "Confidence"]]

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
