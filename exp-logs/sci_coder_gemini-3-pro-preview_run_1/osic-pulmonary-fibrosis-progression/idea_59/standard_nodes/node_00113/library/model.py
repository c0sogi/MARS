import os
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import pandas as pd
import numpy as np
from tqdm import tqdm
from library.utils import get_device, compute_metric, seed_everything, AverageMeter
from library.dataset import get_dataloaders

# ==========================================
# Model Architecture: BCSL-Net
# ==========================================


class BCSLNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Independent Low-Capacity Visual Backbones
        # EfficientNet-B0 outputs 1280 features at the final stage.
        # num_classes=0 applies Global Average Pooling and returns the features.
        self.backbone_ax = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0
        )
        self.backbone_cor = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0
        )

        # 2. Shared-Latent Tabular Encoder
        # Input: 7 features (Age, Sex, 3xSmoking, Percent, Base_FVC_Scaled)
        # Output: 128-dim Shared Latent Vector (T_lat)
        self.tab_encoder = nn.Sequential(
            nn.Linear(7, 64), nn.GELU(), nn.Linear(64, 128), nn.GELU()
        )

        # 3. Normalized Bifurcated Flow (Flow A: Alignment)
        # Projects 128-dim latent to 1280-dim to match visual backbones.
        # LayerNorm is critical here to match the scale of pre-trained visual features.
        self.align_proj = nn.Linear(128, 1280)
        self.align_norm = nn.LayerNorm(1280)

        # 4. Pre-Norm Symmetric Attention
        # Fuses [V_ax, V_cor, T_align]
        self.attention = nn.TransformerEncoderLayer(
            d_model=1280,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-Norm for stability
        )

        # 5. Balanced-Bottleneck Head
        # Compresses the 1280-dim fused context down to 128-dim.
        self.bottleneck = nn.Sequential(
            nn.Linear(1280, 128), nn.GELU(), nn.Dropout(0.2)
        )

        # Final Prediction Layer
        # Input: 256 dims (128 from Fused Context + 128 from Original Clinical Prior)
        # Output: 3 parameters (Alpha, Sigma_Base, Sigma_Growth)
        self.head = nn.Linear(128 + 128, 3)

        # Activation for sigma (must be positive)
        self.sigma_act = nn.Softplus()

    def forward(self, axial, coronal, tabular, delta_week, base_fvc):
        # --- Visual Encoding ---
        v_ax = self.backbone_ax(axial)  # (B, 1280)
        v_cor = self.backbone_cor(coronal)  # (B, 1280)

        # --- Tabular Encoding ---
        t_lat = self.tab_encoder(tabular)  # (B, 128) -> The Clinical Prior

        # --- Flow A: Alignment ---
        t_align = self.align_proj(t_lat)  # (B, 1280)
        t_align = self.align_norm(t_align)

        # --- Tokenization ---
        # Stack tokens: [Axial, Coronal, Aligned_Tabular]
        tokens = torch.stack([v_ax, v_cor, t_align], dim=1)  # (B, 3, 1280)

        # --- Contextualization (Attention) ---
        tokens_out = self.attention(tokens)  # (B, 3, 1280)

        # --- Holistic Readout ---
        # Global Average Pooling across the sequence to capture the full context
        h_fused = tokens_out.mean(dim=1)  # (B, 1280)

        # --- Balanced Bottleneck ---
        h_compressed = self.bottleneck(h_fused)  # (B, 128)

        # --- Flow B: Balanced Concatenation ---
        # Concatenate compressed context with the original clinical prior
        # This enforces 50/50 capacity sharing between learned context and prior
        combined = torch.cat([h_compressed, t_lat], dim=1)  # (B, 256)

        # --- Parametric Prediction ---
        out = self.head(combined)

        alpha = out[:, 0]
        sigma_base = self.sigma_act(out[:, 1])
        sigma_growth = self.sigma_act(out[:, 2])

        # --- Trajectory Logic ---
        # FVC = Base + Alpha * Delta_Week
        # Sigma = Base_Sigma + Growth_Sigma * |Delta_Week|
        fvc_pred = base_fvc + alpha * delta_week
        sigma_pred = sigma_base + sigma_growth * torch.abs(delta_week)

        return fvc_pred, sigma_pred


# ==========================================
# Training & Evaluation Logic
# ==========================================


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_meter = AverageMeter()

    for batch in loader:
        # Move inputs to device
        axial = batch["axial"].to(device)
        coronal = batch["coronal"].to(device)
        tabular = batch["tabular"].to(device)
        delta_week = batch["delta_week"].to(device)
        base_fvc = batch["base_fvc"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        fvc_pred, sigma_pred = model(axial, coronal, tabular, delta_week, base_fvc)

        # Calculate Loss
        # Metric is negative log likelihood (higher is better).
        # Loss = -Metric (minimize loss).
        metric = compute_metric(target, fvc_pred, sigma_pred)
        loss = -metric

        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), axial.size(0))

    return loss_meter.avg


def evaluate(model, loader, device):
    model.eval()
    metric_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            delta_week = batch["delta_week"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            target = batch["target"].to(device)

            fvc_pred, sigma_pred = model(axial, coronal, tabular, delta_week, base_fvc)

            metric = compute_metric(target, fvc_pred, sigma_pred)
            metric_meter.update(metric.item(), axial.size(0))

    return metric_meter.avg


def run_training(
    epochs=30, batch_size=16, patience=8, save_path="./working/best_model.pth"
):
    seed_everything(42)
    device = get_device()

    # Data
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)

    # Model
    model = BCSLNet().to(device)

    # Optimization
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_metric = -float("inf")
    early_stop_counter = 0

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metric = evaluate(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Metric: {val_metric:.6f}"
        )

        # Checkpoint & Early Stopping
        if val_metric > best_metric:
            best_metric = val_metric
            early_stop_counter = 0
            torch.save(model.state_dict(), save_path)
            print("  -> New best model saved!")
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    print(f"Training complete. Best Val Metric: {best_metric:.6f}")
    return best_metric


def generate_submission(
    model_path="./working/best_model.pth", output_path="./submission/submission.csv"
):
    device = get_device()

    # Load Data
    _, _, test_loader = get_dataloaders(batch_size=16)

    # Load Model
    model = BCSLNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found. Using random weights.")

    model.eval()

    results = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)
            delta_week = batch["delta_week"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            patient_weeks = batch["patient_week"]

            fvc_pred, sigma_pred = model(axial, coronal, tabular, delta_week, base_fvc)

            # Move to CPU
            fvc_pred = fvc_pred.cpu().numpy()
            sigma_pred = sigma_pred.cpu().numpy()

            for pw, f, s in zip(patient_weeks, fvc_pred, sigma_pred):
                results.append({"Patient_Week": pw, "FVC": f, "Confidence": s})

    # Create DataFrame
    sub_df = pd.DataFrame(results)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save
    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} with {len(sub_df)} rows.")


def run_experiment():
    # Ensure working directories exist
    os.makedirs("./working", exist_ok=True)
    os.makedirs("./submission", exist_ok=True)

    # Train
    run_training(epochs=35, batch_size=16, patience=8)

    # Predict
    generate_submission()
