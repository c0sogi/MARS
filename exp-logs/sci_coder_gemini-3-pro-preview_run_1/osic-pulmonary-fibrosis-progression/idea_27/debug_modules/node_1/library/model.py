import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import pandas as pd
import math

from library.config import Config
from library.data import get_dataloaders
from library.utils import AverageMeter, calculate_metric

# ==========================================
# 1. Model Architecture
# ==========================================


class TabularEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim // 2),
            nn.GELU(),
            nn.Linear(output_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class VCDAN(nn.Module):
    """
    Visual-Contextualized Dual-Axis Network (VC-DAN)
    """

    def __init__(self):
        super().__init__()
        # 1. Visual Backbones (Independent)
        # EfficientNet-B0 output is 1280
        self.backbone_ax = models.efficientnet_b0(weights="IMAGENET1K_V1")
        self.backbone_cor = models.efficientnet_b0(weights="IMAGENET1K_V1")

        # Remove classifiers, keep feature extractors
        self.backbone_ax.classifier = nn.Identity()
        self.backbone_cor.classifier = nn.Identity()

        self.visual_dim = Config.VISUAL_DIM  # 1280
        self.tabular_dim = 6  # Age, Sex, Smoke, Percent, Base_FVC, Rel_Week

        # 2. Tabular Encoder (Up-projection)
        self.tab_encoder = TabularEncoder(self.tabular_dim, self.visual_dim)

        # 3. Symmetric Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=self.visual_dim,
            num_heads=Config.N_HEADS,
            dropout=Config.DROPOUT,
            batch_first=True,
        )

        # 4. Head
        # Input: Pooled Visual (1280) + Raw Tabular (6) via skip connection
        head_in_dim = self.visual_dim + self.tabular_dim
        self.head = nn.Sequential(
            nn.Linear(head_in_dim, 512),
            nn.GELU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(512, 2),  # Output: FVC, Confidence
        )

    def forward_features(self, x, backbone):
        # x: (B, 3, 224, 224)
        # backbone.features gives (B, 1280, 7, 7)
        x = backbone.features(x)
        x = F.adaptive_avg_pool2d(x, 1)  # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)  # (B, 1280)
        return x

    def forward(self, img_ax, img_cor, tabular):
        # 1. Extract Visual Features
        v_ax = self.forward_features(img_ax, self.backbone_ax)
        v_cor = self.forward_features(img_cor, self.backbone_cor)

        # 2. Encode Tabular
        v_tab = self.tab_encoder(tabular)

        # 3. Stack Tokens: [Axial, Coronal, Tabular]
        # Shape: (B, 3, 1280)
        tokens = torch.stack([v_ax, v_cor, v_tab], dim=1)

        # 4. Attention (Self-Attention)
        # Output: (B, 3, 1280)
        attn_out, _ = self.attn(tokens, tokens, tokens)

        # 5. Visual-Exclusive Pooled Readout
        # Extract refined visual tokens (indices 0 and 1)
        v_ax_refined = attn_out[:, 0, :]
        v_cor_refined = attn_out[:, 1, :]

        # Average Pool of visual tokens only
        v_pool = (v_ax_refined + v_cor_refined) / 2.0

        # 6. Skip Connection & Head
        # Concat with raw tabular features (skip connection)
        combined = torch.cat([v_pool, tabular], dim=1)

        # Predict
        out = self.head(combined)

        fvc_pred = out[:, 0]
        # Enforce positivity for confidence using Softplus
        confidence_pred = F.softplus(out[:, 1])

        return fvc_pred, confidence_pred


# ==========================================
# 2. Loss Function
# ==========================================


def laplace_log_likelihood_loss(y_pred, y_std, y_true):
    """
    Differentiable implementation of the competition metric.
    Loss = -Metric
    """
    # Clipping logic from metric definition
    sigma_clipped = torch.clamp(y_std, min=Config.MIN_CONFIDENCE)
    delta = torch.abs(y_true - y_pred)
    delta_clipped = torch.clamp(delta, max=Config.MAX_ERROR)

    sqrt_2 = math.sqrt(2)
    metric = -(sqrt_2 * delta_clipped) / sigma_clipped - torch.log(
        sqrt_2 * sigma_clipped
    )

    # We want to maximize metric, so minimize negative metric
    return -torch.mean(metric)


# ==========================================
# 3. Training Routine
# ==========================================


def train_model():
    print("Initializing Training...")

    # Data
    train_loader, val_loader, _ = get_dataloaders(load_cached_data=True)

    # Model
    model = VCDAN().to(Config.DEVICE)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # Tracking
    best_metric = -float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss_meter = AverageMeter()

        for batch in train_loader:
            img_ax = batch["img_ax"].to(Config.DEVICE)
            img_cor = batch["img_cor"].to(Config.DEVICE)
            tabular = batch["tabular"].to(Config.DEVICE)
            target = batch["target"].to(Config.DEVICE)

            optimizer.zero_grad()

            fvc_pred, conf_pred = model(img_ax, img_cor, tabular)

            loss = laplace_log_likelihood_loss(fvc_pred, conf_pred, target)

            loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item(), img_ax.size(0))

        # Validation
        model.eval()
        val_metric_meter = AverageMeter()

        with torch.no_grad():
            for batch in val_loader:
                img_ax = batch["img_ax"].to(Config.DEVICE)
                img_cor = batch["img_cor"].to(Config.DEVICE)
                tabular = batch["tabular"].to(Config.DEVICE)
                target = batch["target"].to(Config.DEVICE)

                fvc_pred, conf_pred = model(img_ax, img_cor, tabular)

                # Calculate metric using library utility
                score = calculate_metric(target, fvc_pred, conf_pred)
                val_metric_meter.update(score, img_ax.size(0))

        scheduler.step()

        current_metric = val_metric_meter.avg
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss_meter.avg:.4f} | Val Metric: {current_metric:.6f}"
        )

        # Checkpointing & Early Stopping
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  -> New Best Model Saved! (Metric: {best_metric:.6f})")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(
                f"Early stopping triggered after {patience_counter} epochs without improvement."
            )
            break

    print(f"Training Complete. Best Validation Metric: {best_metric:.6f}")


# ==========================================
# 4. Inference Routine
# ==========================================


def generate_submission():
    print("Generating Submission...")

    # Load Data
    _, _, test_loader = get_dataloaders(load_cached_data=True)

    # Load Model
    model = VCDAN().to(Config.DEVICE)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        )
        print("Loaded best model checkpoint.")
    else:
        print("Warning: No checkpoint found. Using untrained model.")

    model.eval()

    results = []

    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["img_ax"].to(Config.DEVICE)
            img_cor = batch["img_cor"].to(Config.DEVICE)
            tabular = batch["tabular"].to(Config.DEVICE)
            patient_weeks = batch["patient_week"]  # List of strings

            fvc_pred, conf_pred = model(img_ax, img_cor, tabular)

            fvc_pred = fvc_pred.cpu().numpy()
            conf_pred = conf_pred.cpu().numpy()

            for pw, f, c in zip(patient_weeks, fvc_pred, conf_pred):
                results.append({"Patient_Week": pw, "FVC": f, "Confidence": c})

    # Save
    sub_df = pd.DataFrame(results)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(sub_df.head())
