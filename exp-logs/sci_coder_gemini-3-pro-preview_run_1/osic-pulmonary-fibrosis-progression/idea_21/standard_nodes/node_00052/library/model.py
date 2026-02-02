import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from tqdm import tqdm

from library.config import Config
from library.utils import compute_metric


class VisualBackbone(nn.Module):
    """
    Extracts high-fidelity global features from a specific CT view (Axial or Coronal).
    Uses EfficientNet-B0.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        # Load EfficientNet-B0
        # num_classes=0 removes the classifier and returns the pooled features (if global_pool is set)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.out_dim = self.backbone.num_features  # 1280 for B0

    def forward(self, x):
        # x: (B, 3, 224, 224)
        return self.backbone(x)


class TabularProjector(nn.Module):
    """
    Projects low-dimensional clinical metadata to high-dimensional visual space.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim // 2),
            nn.BatchNorm1d(output_dim // 2),
            nn.GELU(),
            nn.Linear(output_dim // 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class SymmetricAttention(nn.Module):
    """
    Contextualizes Axial, Coronal, and Tabular tokens using Self-Attention.
    """

    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens):
        # tokens: (B, Seq_Len, Embed_Dim) -> (B, 3, 1280)
        # Self-attention
        attn_out, _ = self.attn(tokens, tokens, tokens)
        # Residual + Norm
        return self.norm(tokens + attn_out)


class GatedAggregator(nn.Module):
    """
    Dynamically weights Axial and Coronal views based on clinical priors.
    """

    def __init__(self, tabular_dim):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # Outputs weights for [Axial, Coronal]
        )

    def forward(self, tabular_raw, vis_tokens):
        # tabular_raw: (B, 5)
        # vis_tokens: (B, 2, 1280) -> [Axial, Coronal]

        # Predict logits and softmax to get weights
        logits = self.gate_net(tabular_raw)
        weights = F.softmax(logits, dim=1).unsqueeze(-1)  # (B, 2, 1)

        # Weighted sum
        # weights[:, 0] * ax + weights[:, 1] * cor
        aggregated = (vis_tokens * weights).sum(dim=1)  # (B, 1280)
        return aggregated


class CG_SDAN(nn.Module):
    """
    Clinically-Gated Symmetric Dual-Axis Network.
    """

    def __init__(self):
        super().__init__()

        # 1. Independent Visual Backbones
        self.backbone_ax = VisualBackbone(pretrained=Config.BACKBONE_PRETRAINED)
        self.backbone_cor = VisualBackbone(pretrained=Config.BACKBONE_PRETRAINED)

        # 2. Tabular Projector
        # Input: 5 features (Week, Percent, Age, Sex, Smoke)
        self.tab_projector = TabularProjector(5, Config.VISUAL_FEATURE_DIM)

        # 3. Symmetric Attention
        self.attention = SymmetricAttention(Config.VISUAL_FEATURE_DIM)

        # 4. Gating Mechanism
        self.aggregator = GatedAggregator(5)

        # 5. Prior-Anchored Head
        # Input: Visual (1280) + Raw Tabular (5)
        head_in_dim = Config.VISUAL_FEATURE_DIM + 5

        self.head = nn.Sequential(
            nn.Linear(head_in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3),  # alpha, sigma_base, sigma_growth
        )

    def forward(self, img_ax, img_cor, tabular):
        # 1. Extract Visual Features
        v_ax = self.backbone_ax(img_ax)  # (B, 1280)
        v_cor = self.backbone_cor(img_cor)  # (B, 1280)

        # 2. Project Tabular
        v_tab = self.tab_projector(tabular)  # (B, 1280)

        # 3. Stack and Attend
        # Sequence: [Axial, Coronal, Tabular]
        tokens = torch.stack([v_ax, v_cor, v_tab], dim=1)  # (B, 3, 1280)
        context_tokens = self.attention(tokens)

        # Extract contextualized visual tokens (indices 0 and 1)
        vis_context = context_tokens[:, :2, :]  # (B, 2, 1280)

        # 4. Gated Aggregation
        # Use raw tabular features for gating logic
        v_aggregated = self.aggregator(tabular, vis_context)  # (B, 1280)

        # 5. Prior-Anchored Prediction
        # Skip connection: Concatenate aggregated visual with raw tabular
        combined = torch.cat([v_aggregated, tabular], dim=1)

        out = self.head(combined)

        # Unpack outputs
        alpha = out[:, 0].unsqueeze(1)  # Slope

        # Apply Softplus to sigmas to ensure positivity
        sigma_base = F.softplus(out[:, 1].unsqueeze(1))
        sigma_growth = F.softplus(out[:, 2].unsqueeze(1))

        return alpha, sigma_base, sigma_growth


def criterion(
    alpha,
    sigma_base,
    sigma_growth,
    target_fvc,
    meta_weeks,
    meta_base_fvc,
    meta_base_week,
):
    """
    Differentiable Modified Laplace Log Likelihood Loss.
    """
    # 1. Calculate Predicted FVC based on linear trajectory
    # FVC = Base + alpha * delta_t
    delta_t = meta_weeks - meta_base_week
    pred_fvc = meta_base_fvc + alpha * delta_t

    # 2. Calculate Predicted Confidence (Sigma)
    # Sigma = Base + Growth * |delta_t|
    pred_sigma = sigma_base + sigma_growth * torch.abs(delta_t)

    # 3. Metric Calculation (Negative of the metric to minimize loss)

    # Clip Sigma: max(sigma, 70)
    # Use torch.clamp_min for differentiability (subgradient)
    sigma_clipped = torch.clamp(pred_sigma, min=Config.CONFIDENCE_MIN_THRESHOLD)

    # Calculate Delta: min(|True - Pred|, 1000)
    abs_err = torch.abs(target_fvc - pred_fvc)
    delta = torch.clamp(abs_err, max=Config.ERROR_MAX_THRESHOLD)

    sqrt_2 = 1.41421356

    # Metric formula: - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
    # We want to MAXIMIZE metric, so we MINIMIZE:
    # Loss = (sqrt(2) * delta) / sigma + ln(sqrt(2) * sigma)

    term1 = (sqrt_2 * delta) / sigma_clipped
    term2 = torch.log(sqrt_2 * sigma_clipped)

    loss = torch.mean(term1 + term2)

    return loss


def train_model(train_loader, val_loader):
    """
    Executes the training loop with Early Stopping.
    """
    device = torch.device(Config.DEVICE)
    model = CG_SDAN().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    best_metric = -float("inf")
    patience_counter = 0

    print(f"Starting training on {device} for {Config.N_EPOCHS} epochs...")

    for epoch in range(Config.N_EPOCHS):
        # --- Training ---
        model.train()
        train_loss_accum = 0.0

        for batch in train_loader:
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            targets = batch["target"].to(device)

            # Extract metadata for trajectory calculation
            # Note: DataLoader collates numbers into tensors
            m_weeks = batch["meta"]["Weeks"].to(device).view(-1, 1)
            m_base_fvc = batch["meta"]["Baseline_FVC"].to(device).view(-1, 1)
            m_base_week = batch["meta"]["Baseline_Week"].to(device).view(-1, 1)

            optimizer.zero_grad()

            alpha, s_base, s_growth = model(img_ax, img_cor, tabular)

            loss = criterion(
                alpha, s_base, s_growth, targets, m_weeks, m_base_fvc, m_base_week
            )

            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        scheduler.step()
        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation ---
        model.eval()
        val_preds_fvc = []
        val_preds_sigma = []
        val_true = []

        with torch.no_grad():
            for batch in val_loader:
                img_ax = batch["image_axial"].to(device)
                img_cor = batch["image_coronal"].to(device)
                tabular = batch["tabular"].to(device)
                targets = batch["target"].to(device)

                m_weeks = batch["meta"]["Weeks"].to(device).view(-1, 1)
                m_base_fvc = batch["meta"]["Baseline_FVC"].to(device).view(-1, 1)
                m_base_week = batch["meta"]["Baseline_Week"].to(device).view(-1, 1)

                alpha, s_base, s_growth = model(img_ax, img_cor, tabular)

                # Calculate FVC and Sigma for metric
                delta_t = m_weeks - m_base_week
                p_fvc = m_base_fvc + alpha * delta_t
                p_sigma = s_base + s_growth * torch.abs(delta_t)

                val_preds_fvc.extend(p_fvc.cpu().numpy().flatten())
                val_preds_sigma.extend(p_sigma.cpu().numpy().flatten())
                val_true.extend(targets.cpu().numpy().flatten())

        # Compute Metric
        val_metric = compute_metric(val_true, val_preds_fvc, val_preds_sigma)

        print(
            f"Epoch {epoch+1}/{Config.N_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Metric: {val_metric:.6f}"
        )

        # --- Early Stopping & Saving ---
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            # print(f"  -> New best model saved! ({val_metric:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Validation Metric: {best_metric:.6f}")
    return best_metric


def predict(test_loader):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    device = torch.device(Config.DEVICE)

    # Load Model
    model = CG_SDAN().to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No trained model found. Using random weights.")

    model.eval()

    results = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)

            # Metadata
            m_weeks = batch["meta"]["Weeks"].to(device).view(-1, 1)
            m_base_fvc = batch["meta"]["Baseline_FVC"].to(device).view(-1, 1)
            m_base_week = batch["meta"]["Baseline_Week"].to(device).view(-1, 1)
            patient_weeks = batch["meta"]["Patient_Week"]  # List of strings

            # Forward
            alpha, s_base, s_growth = model(img_ax, img_cor, tabular)

            # Calculate final predictions
            delta_t = m_weeks - m_base_week
            pred_fvc = m_base_fvc + alpha * delta_t
            pred_sigma = s_base + s_growth * torch.abs(delta_t)

            # Move to CPU
            pred_fvc = pred_fvc.cpu().numpy().flatten()
            pred_sigma = pred_sigma.cpu().numpy().flatten()

            # Store results
            for pw, fvc, sigma in zip(patient_weeks, pred_fvc, pred_sigma):
                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": sigma})

    # Create DataFrame
    sub_df = pd.DataFrame(results)

    # Ensure Confidence is clipped at 70 (Metric requirement)
    # Though the loss function handles it, the submission file should also reflect valid values
    sub_df["Confidence"] = sub_df["Confidence"].apply(lambda x: max(x, 70))

    # Save
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
