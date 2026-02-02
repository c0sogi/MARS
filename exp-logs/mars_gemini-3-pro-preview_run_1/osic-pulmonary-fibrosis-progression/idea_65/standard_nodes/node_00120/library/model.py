import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.utils import AverageMeter, LaplaceLogLikelihood
from library.data import get_dataloaders

# ==========================================
# Model Architecture
# ==========================================


class VisualBackbone(nn.Module):
    def __init__(self, model_name="efficientnet_b0", pretrained=True):
        super().__init__()
        # Create EfficientNet B0
        # num_classes=0 removes the classifier
        # global_pool='avg' ensures we get the pooled feature vector
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.out_dim = self.backbone.num_features  # Should be 1280 for B0

    def forward(self, x):
        # x: (B, 3, 224, 224)
        return self.backbone(x)  # (B, 1280)


class TabularEncoder(nn.Module):
    def __init__(self, input_dim=7, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, latent_dim), nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class FusionBlock(nn.Module):
    def __init__(self, embed_dim=1280, latent_dim=128):
        super().__init__()
        # Project latent tabular to visual dimension
        self.align_proj = nn.Linear(latent_dim, embed_dim)
        self.align_norm = nn.LayerNorm(embed_dim)

        # Pre-Norm Attention components
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=8, batch_first=True
        )
        self.dropout1 = nn.Dropout(0.1)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, v_ax, v_cor, t_lat):
        # v_ax, v_cor: (B, 1280)
        # t_lat: (B, 128)

        # Align tabular
        t_align = self.align_proj(t_lat)
        t_align = self.align_norm(t_align)  # (B, 1280)

        # Stack sequence: [Axial, Coronal, Tabular]
        # (B, 3, 1280)
        seq = torch.stack([v_ax, v_cor, t_align], dim=1)

        # Self-Attention (Pre-Norm)
        seq_norm = self.norm1(seq)
        attn_out, _ = self.attn(seq_norm, seq_norm, seq_norm)
        seq = seq + self.dropout1(attn_out)

        # FFN (Pre-Norm)
        seq_norm = self.norm2(seq)
        ffn_out = self.ffn(seq_norm)
        seq = seq + self.dropout2(ffn_out)

        # Unstack
        v_ax_ctx = seq[:, 0, :]
        v_cor_ctx = seq[:, 1, :]
        t_ctx = seq[:, 2, :]

        return v_ax_ctx, v_cor_ctx, t_ctx


class TSCPNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Independent Visual Backbones
        self.vis_backbone_ax = VisualBackbone()
        self.vis_backbone_cor = VisualBackbone()

        # 2. Shared-Latent Tabular Encoder
        self.tab_encoder = TabularEncoder(input_dim=7, latent_dim=128)

        # 3. Fusion Block
        self.fusion = FusionBlock(embed_dim=1280, latent_dim=128)

        # 4. Tri-Stream Readout Projections
        # Stream 1: Visual Context (1280 -> 128)
        self.s1_proj = nn.Sequential(nn.Linear(1280, 128), nn.GELU())

        # Stream 2: Tabular Context (1280 -> 64)
        self.s2_proj = nn.Sequential(nn.Linear(1280, 64), nn.GELU())

        # Stream 3: Clinical Prior (Raw 128) -> No projection needed

        # 5. Non-Linear Parametric Head
        # Input: 128 (Vis) + 64 (TabCtx) + 128 (Prior) = 320
        self.head = nn.Sequential(
            nn.Linear(320, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),  # alpha, sigma_base, sigma_growth
        )

    def forward(self, img_ax, img_cor, tabular):
        # Extract features
        v_ax = self.vis_backbone_ax(img_ax)  # (B, 1280)
        v_cor = self.vis_backbone_cor(img_cor)  # (B, 1280)
        t_lat = self.tab_encoder(tabular)  # (B, 128)

        # Contextualize
        v_ax_ctx, v_cor_ctx, t_ctx = self.fusion(v_ax, v_cor, t_lat)

        # Stream 1: Visual Context
        v_mean = (v_ax_ctx + v_cor_ctx) / 2.0
        h_vis = self.s1_proj(v_mean)  # (B, 128)

        # Stream 2: Tabular Context
        h_ctx = self.s2_proj(t_ctx)  # (B, 64)

        # Stream 3: Clinical Prior
        h_prior = t_lat  # (B, 128)

        # Assembly
        combined = torch.cat([h_vis, h_ctx, h_prior], dim=1)  # (B, 320)

        # Prediction
        out = self.head(combined)

        alpha = out[:, 0]
        # Enforce positive sigma
        sigma_base = F.softplus(out[:, 1])
        sigma_growth = F.softplus(out[:, 2])

        return alpha, sigma_base, sigma_growth


# ==========================================
# Loss Function
# ==========================================


def negative_laplace_log_likelihood(fvc_true, fvc_pred, sigma):
    """
    Computes the negative of the metric for minimization.
    Metric: - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    Loss = -Metric = (sqrt(2) * delta) / sigma_clipped + ln(sqrt(2) * sigma_clipped)
    """
    sigma_clipped = torch.clamp(sigma, min=70)
    delta = torch.abs(fvc_true - fvc_pred)
    delta = torch.clamp(delta, max=1000)

    sqrt_2 = 1.41421356

    term1 = (sqrt_2 * delta) / sigma_clipped
    term2 = torch.log(sqrt_2 * sigma_clipped)

    loss = term1 + term2
    return torch.mean(loss)


# ==========================================
# Training & Evaluation
# ==========================================


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    losses = AverageMeter()

    for batch in loader:
        img_ax = batch["img_ax"].to(device)
        img_cor = batch["img_cor"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["target"].to(device)
        weeks = batch["weeks"].to(device)  # Relative weeks
        base_fvc = batch["base_fvc"].to(device)

        optimizer.zero_grad()

        alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

        # Parametric Inference
        # FVC = Base + alpha * weeks
        fvc_pred = base_fvc + alpha * weeks

        # Confidence = sigma_base + sigma_growth * |weeks|
        sigma_pred = sigma_base + sigma_growth * torch.abs(weeks)

        loss = negative_laplace_log_likelihood(target, fvc_pred, sigma_pred)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), img_ax.size(0))

    return losses.avg


def validate(model, loader, device):
    model.eval()
    metric_score = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)
            weeks = batch["weeks"].to(device)
            base_fvc = batch["base_fvc"].to(device)

            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

            fvc_pred = base_fvc + alpha * weeks
            sigma_pred = sigma_base + sigma_growth * torch.abs(weeks)

            # Calculate actual metric (higher is better)
            score = LaplaceLogLikelihood(target, fvc_pred, sigma_pred)
            metric_score.update(score, img_ax.size(0))

    return metric_score.avg


def run_training(epochs=30, batch_size=16, debug=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size, debug=debug)

    # Model
    model = TSCPNet().to(device)

    # Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Tracking
    best_score = -float("inf")
    best_model_path = "./working/best_model.pth"
    patience = 8
    patience_counter = 0

    print("Starting training...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_score = validate(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Score: {val_score:.6f}"
        )

        # Save best
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"  -> New best model saved! Score: {best_score:.6f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return best_model_path


def generate_submission(model_path, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    _, _, test_loader = get_dataloaders(batch_size=batch_size, debug=False)

    # Load Model
    model = TSCPNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(
            "Warning: Model path not found. Generating predictions with untrained model."
        )

    model.eval()

    results = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            weeks = batch["weeks"].to(device)  # Relative weeks for prediction
            base_fvc = batch["base_fvc"].to(device)
            pids = batch["patient_id"]

            # Predict parameters
            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

            # Calculate FVC and Confidence
            fvc_pred = base_fvc + alpha * weeks
            sigma_pred = sigma_base + sigma_growth * torch.abs(weeks)

            # Move to CPU
            fvc_pred = fvc_pred.cpu().numpy()
            sigma_pred = sigma_pred.cpu().numpy()
            weeks_cpu = weeks.cpu().numpy()
            base_fvc_cpu = base_fvc.cpu().numpy()

            # Iterate batch
            for i in range(len(pids)):
                pid = pids[i]
                # Reconstruct Patient_Week ID
                # We need the absolute week.
                # weeks[i] = predict_week - baseline_week
                # So predict_week = weeks[i] + baseline_week
                # However, test_loader batch doesn't give absolute week directly easily without passing it.
                # But we can reconstruct the ID if we had the original df.
                # Alternatively, we can assume the order matches test.csv from metadata.
                # Actually, let's use the fact that metadata/test.csv has Patient_Week.
                # But the loader shuffles? No, test_loader shuffle=False.
                pass

            # Collect results
            # Since we can't easily reconstruct the exact Patient_Week string from just tensors inside the loop
            # without passing it through the dataset, and the dataset returns dicts...
            # The dataset in data.py doesn't return Patient_Week.
            # However, test_loader is sequential (shuffle=False).
            # We can just collect all predictions and merge with the dataframe.

            results.append(np.stack([fvc_pred, sigma_pred], axis=1))

    # Concatenate all batch results
    all_preds = np.concatenate(results, axis=0)

    # Load test metadata to map back to IDs
    test_df = pd.read_csv("./metadata/test.csv")

    # Ensure lengths match
    if len(all_preds) != len(test_df):
        print(
            f"Error: Prediction count {len(all_preds)} != Metadata count {len(test_df)}"
        )
        # Handle mismatch if necessary (e.g. drop_last=False was used?)
        # test_loader has drop_last=False by default in data.py

    # Assign
    test_df["FVC"] = all_preds[:, 0]
    test_df["Confidence"] = all_preds[:, 1]

    # Format for submission
    submission = test_df[["Patient_Week", "FVC", "Confidence"]].copy()

    # Ensure Confidence is clipped at 70 (Metric requirement, though metric does it internally,
    # submission usually expects raw or clipped. The metric definition says "confidence values are clipped at 70...".
    # It is safer to clip here to be consistent.)
    submission["Confidence"] = submission["Confidence"].clip(lower=70)

    os.makedirs("./submission", exist_ok=True)
    submission.to_csv("./submission/submission.csv", index=False)
    print("Submission saved to ./submission/submission.csv")
