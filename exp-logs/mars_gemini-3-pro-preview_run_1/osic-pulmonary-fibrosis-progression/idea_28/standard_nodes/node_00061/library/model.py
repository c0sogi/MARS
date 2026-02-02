import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
import pandas as pd
import numpy as np

# Import from provided libraries
from library.dataset import FVCDataset, get_transforms
from library.utils import AverageMeter, LaplaceLogLikelihood, seed_everything


class TabularEncoder(nn.Module):
    """
    Encodes tabular features (Age, Percent, Sex, Smoking) into a high-dimensional embedding.
    """

    def __init__(self, input_dim=6, output_dim=1280, hidden_dim=512):
        super(TabularEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.mlp(x)


class VERNet(nn.Module):
    """
    Visual-Exclusive Residual Network (VER-Net).
    Integrates multi-view CT scans with tabular data using symmetric attention
    and a visual-exclusive pooling strategy.
    """

    def __init__(self, backbone_name="tf_efficientnet_b0_ns", pretrained=True):
        super(VERNet, self).__init__()

        # 1. Independent Visual Backbones
        # Create backbones with no classifier, global pool gives (B, 1280)
        self.backbone_ax = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        self.backbone_cor = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )

        feature_dim = self.backbone_ax.num_features  # 1280 for B0

        # 2. Tabular Encoder (Projects up to feature_dim)
        self.tabular_encoder = TabularEncoder(input_dim=6, output_dim=feature_dim)

        # 3. Symmetric Attention
        # Sequence length 3: [Axial, Coronal, Tabular]
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=8, batch_first=True
        )
        self.ln_pre = nn.LayerNorm(feature_dim)

        # 4. Heads
        # Input to head: Visual_Residual (1280) + Raw_Tabular (6) = 1286
        head_input_dim = feature_dim + 6

        # Main Regression Head: Slope (alpha), Sigma_base, Sigma_growth
        self.reg_head = nn.Sequential(
            nn.Linear(head_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 3),  # alpha, sigma_base, sigma_growth
        )

        # Auxiliary Head: Baseline Estimator
        # Used to estimate baseline FVC when it is missing (0.0) in training data
        self.baseline_head = nn.Sequential(
            nn.Linear(head_input_dim, 256), nn.ReLU(), nn.Linear(256, 1)
        )

    def forward(self, img_ax, img_cor, tabular, time_delta, baseline_fvc_input):
        """
        Args:
            img_ax: (B, 3, 224, 224)
            img_cor: (B, 3, 224, 224)
            tabular: (B, 6) [Age, Percent, Sex, Smoke_0, Smoke_1, Smoke_2]
            time_delta: (B, ) Week - Baseline_Week
            baseline_fvc_input: (B, ) Provided baseline FVC (0 in training usually)
        """
        batch_size = img_ax.size(0)

        # --- 1. Feature Extraction ---
        # Visual
        v_ax = self.backbone_ax(img_ax)  # (B, 1280)
        v_cor = self.backbone_cor(img_cor)  # (B, 1280)

        # Tabular Embedding
        v_tab = self.tabular_encoder(tabular)  # (B, 1280)

        # --- 2. Symmetric Attention ---
        # Stack: [Axial, Coronal, Tabular]
        tokens = torch.stack([v_ax, v_cor, v_tab], dim=1)  # (B, 3, 1280)

        # Pre-Norm
        tokens_norm = self.ln_pre(tokens)

        # Self-Attention
        attn_out, _ = self.attention(tokens_norm, tokens_norm, tokens_norm)

        # Residual connection
        tokens = tokens + attn_out

        # Unstack
        v_ax_new = tokens[:, 0, :]
        v_cor_new = tokens[:, 1, :]
        # v_tab_new = tokens[:, 2, :] # Not used in visual residual

        # --- 3. Visual-Exclusive Readout ---
        # Average Pool only the visual tokens to isolate visual signal
        visual_residual = (v_ax_new + v_cor_new) / 2.0  # (B, 1280)

        # --- 4. Prior-Anchored Feature Construction ---
        # Concatenate Visual Residual with Raw Tabular (Skip Connection)
        combined_features = torch.cat([visual_residual, tabular], dim=1)  # (B, 1286)

        # --- 5. Prediction ---
        # Main Parameters
        params = self.reg_head(combined_features)
        alpha = params[:, 0]
        sigma_base = F.softplus(params[:, 1])
        sigma_growth = F.softplus(params[:, 2])

        # Baseline Estimation (Auxiliary)
        est_baseline = self.baseline_head(combined_features).squeeze(-1)

        # --- 6. Trajectory Calculation ---
        # Determine which baseline to use
        # If baseline_fvc_input > 100 (valid), use it. Else use estimate.
        # This allows training on dataset where baseline is 0.0, while using true baseline in test.
        use_input_mask = (baseline_fvc_input > 100.0).float()

        effective_baseline = (
            use_input_mask * baseline_fvc_input + (1.0 - use_input_mask) * est_baseline
        )

        # FVC Prediction: Base + Slope * Delta_t
        fvc_pred = effective_baseline + alpha * time_delta

        # Confidence Prediction: Sigma_base + Sigma_growth * |Delta_t|
        confidence = sigma_base + sigma_growth * torch.abs(time_delta)

        return fvc_pred, confidence


def train_model(
    epochs=20,
    batch_size=16,
    learning_rate=3e-4,
    device="cuda",
    num_workers=2,
    save_path="./working/best_model.pth",
):
    """
    Trains the VERNet model with Early Stopping and Metric Monitoring.
    """
    seed_everything(42)

    # Dataset and Loaders
    train_dataset = FVCDataset(mode="train", transform=get_transforms("train"))
    val_dataset = FVCDataset(mode="val", transform=get_transforms("val"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Model
    model = VERNet().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # Loss
    criterion = LaplaceLogLikelihood()

    best_score = -float("inf")

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = AverageMeter()

        for batch in train_loader:
            img_ax = batch["img_axial"].to(device)
            img_cor = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)
            week = batch["week"].to(device)
            base_fvc = batch["baseline_fvc"].to(device)

            optimizer.zero_grad()

            pred_fvc, pred_sigma = model(img_ax, img_cor, tabular, week, base_fvc)

            loss = criterion(pred_fvc, pred_sigma, target)

            loss.backward()
            optimizer.step()

            train_loss.update(loss.item(), img_ax.size(0))

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_metric = AverageMeter()

        with torch.no_grad():
            for batch in val_loader:
                img_ax = batch["img_axial"].to(device)
                img_cor = batch["img_coronal"].to(device)
                tabular = batch["tabular"].to(device)
                target = batch["target"].to(device)
                week = batch["week"].to(device)
                base_fvc = batch["baseline_fvc"].to(device)

                pred_fvc, pred_sigma = model(img_ax, img_cor, tabular, week, base_fvc)

                # Metric = -Loss
                batch_loss = criterion(pred_fvc, pred_sigma, target)
                val_metric.update(-batch_loss.item(), img_ax.size(0))

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss.avg:.4f} | Val Metric: {val_metric.avg:.6f}"
        )

        # Checkpoint
        if val_metric.avg > best_score:
            best_score = val_metric.avg
            torch.save(model.state_dict(), save_path)
            print(f"  -> New Best Model Saved! Score: {best_score:.6f}")

    print(f"Training complete. Best Validation Metric: {best_score:.6f}")
    return best_score


def generate_submission(
    model_path="./working/best_model.pth",
    output_path="./submission/submission.csv",
    device="cuda",
):
    """
    Generates submission file using the trained model on the Test set.
    """
    # Load Test Data
    test_dataset = FVCDataset(mode="test", transform=get_transforms("test"))
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

    # Load Model
    model = VERNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found. Using random weights.")

    model.eval()

    results = []

    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["img_axial"].to(device)
            img_cor = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            week = batch["week"].to(device)
            base_fvc = batch["baseline_fvc"].to(device)
            patient_weeks = batch["patient_week"]

            pred_fvc, pred_sigma = model(img_ax, img_cor, tabular, week, base_fvc)

            # Ensure confidence is at least 70 (though metric does this, submission should be clean)
            pred_sigma = torch.clamp(pred_sigma, min=70.0)

            pred_fvc_np = pred_fvc.cpu().numpy()
            pred_sigma_np = pred_sigma.cpu().numpy()

            for pw, fvc, conf in zip(patient_weeks, pred_fvc_np, pred_sigma_np):
                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

    df_sub = pd.DataFrame(results)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(df_sub)} rows.")
