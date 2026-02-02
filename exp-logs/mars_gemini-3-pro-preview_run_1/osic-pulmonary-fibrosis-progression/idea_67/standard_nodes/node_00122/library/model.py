import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import from provided libraries
from library.config import Config
from library.data import OSICDataset, prepare_dataframe, TabularPreprocessor
from library.utils import AverageMeter, score_function

# ==========================================
# 1. Dataset Extension
# ==========================================


class ExtendedOSICDataset(OSICDataset):
    """
    Extends OSICDataset to include Baseline_FVC in the batch,
    which is required for the Anchored Trajectory logic.
    """

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        row = self.df.iloc[idx]
        # Add Baseline FVC for anchored prediction
        # Check if column exists (it should after prepare_dataframe)
        if "Baseline_FVC" in row:
            data["baseline_fvc"] = torch.tensor(
                row["Baseline_FVC"], dtype=torch.float32
            )
        return data


def get_extended_dataloaders(
    batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Re-implementation of get_dataloaders to use ExtendedOSICDataset.
    """
    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # 2. Prepare Dataframes
    train_df = prepare_dataframe(train_df, is_train=True)
    val_df = prepare_dataframe(val_df, is_train=True)
    # Test df already has Baseline columns from metadata generation

    if Config.DEBUG:
        train_df = train_df.head(50)
        val_df = val_df.head(20)
        test_df = test_df.head(20)

    # 3. Initialize Preprocessor
    preprocessor = TabularPreprocessor()
    preprocessor.fit(train_df)

    # 4. Define Transforms
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
            ),
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.Normalize(mean=Config.MEAN, std=Config.STD),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
            A.Normalize(mean=Config.MEAN, std=Config.STD),
            ToTensorV2(),
        ]
    )

    # 5. Create Extended Datasets
    train_ds = ExtendedOSICDataset(
        train_df, mode="train", transform=train_transform, preprocessor=preprocessor
    )
    val_ds = ExtendedOSICDataset(
        val_df, mode="val", transform=val_transform, preprocessor=preprocessor
    )
    test_ds = ExtendedOSICDataset(
        test_df, mode="test", transform=val_transform, preprocessor=preprocessor
    )

    # 6. Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ==========================================
# 2. Model Architecture
# ==========================================


class VisualBackbone(nn.Module):
    def __init__(self, model_name="efficientnet_b0", pretrained=True):
        super().__init__()
        # num_classes=0 returns the pooled features (GAP output)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )

    def forward(self, x):
        # x: (B, 3, H, W) -> (B, 1280)
        return self.backbone(x)


class TabularEncoder(nn.Module):
    def __init__(self, input_dim=6, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, latent_dim), nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class TSCGNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Independent Visual Backbones
        self.backbone_ax = VisualBackbone()
        self.backbone_cor = VisualBackbone()

        # 2. Tabular Encoder
        self.tab_encoder = TabularEncoder(input_dim=6, latent_dim=Config.LATENT_DIM)

        # 3. Alignment & Fusion
        # Project latent (128) to visual dim (1280)
        self.align_proj = nn.Linear(Config.LATENT_DIM, Config.VISUAL_BACKBONE_DIM)
        self.align_norm = nn.LayerNorm(Config.VISUAL_BACKBONE_DIM)

        # Pre-Norm Self-Attention
        # batch_first=True expects (B, Seq, Dim)
        self.attn_norm = nn.LayerNorm(Config.VISUAL_BACKBONE_DIM)
        self.attn = nn.MultiheadAttention(
            embed_dim=Config.VISUAL_BACKBONE_DIM,
            num_heads=8,
            batch_first=True,
            dropout=0.1,
        )

        # 4. Tri-Stream Projections
        # Stream 1: Visual Context (1280 -> 128)
        self.vis_ctx_proj = nn.Sequential(
            nn.Linear(Config.VISUAL_BACKBONE_DIM, Config.VISUAL_CONTEXT_DIM), nn.GELU()
        )

        # Stream 2: Tabular Context (1280 -> 64)
        self.tab_ctx_proj = nn.Sequential(
            nn.Linear(Config.VISUAL_BACKBONE_DIM, Config.TABULAR_CONTEXT_DIM), nn.GELU()
        )

        # Stream 3 is the raw latent (128)

        # 5. Non-Linear Parametric Head
        # Input: 128 (Vis) + 64 (Tab) + 128 (Prior) = 320
        self.head = nn.Sequential(
            nn.Linear(Config.FUSION_DIM, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 3),  # alpha, sigma_base, sigma_growth
        )

    def forward(self, img_ax, img_cor, tabular):
        # Step 1: Feature Extraction
        v_ax = self.backbone_ax(img_ax)  # (B, 1280)
        v_cor = self.backbone_cor(img_cor)  # (B, 1280)
        t_lat = self.tab_encoder(tabular)  # (B, 128) -> Clinical Prior

        # Step 2: Alignment
        t_align = self.align_proj(t_lat)  # (B, 1280)
        t_align = self.align_norm(t_align)

        # Step 3: Contextualization
        # Stack tokens: [Axial, Coronal, Tabular]
        tokens = torch.stack([v_ax, v_cor, t_align], dim=1)  # (B, 3, 1280)

        # Pre-Norm Attention (Residual connection is typical, but here we transform the context)
        # Standard Pre-Norm: x + Attention(Norm(x))
        tokens_norm = self.attn_norm(tokens)
        attn_out, _ = self.attn(tokens_norm, tokens_norm, tokens_norm)
        tokens = tokens + attn_out

        # Unpack
        v_ax_ctx = tokens[:, 0, :]
        v_cor_ctx = tokens[:, 1, :]
        t_align_ctx = tokens[:, 2, :]

        # Step 4: Tri-Stream Readout
        # Stream 1: Visual Context (Mean of views)
        h_vis = self.vis_ctx_proj(0.5 * (v_ax_ctx + v_cor_ctx))  # (B, 128)

        # Stream 2: Tabular Context
        h_ctx = self.tab_ctx_proj(t_align_ctx)  # (B, 64)

        # Stream 3: Clinical Prior (Raw t_lat)
        h_prior = t_lat  # (B, 128)

        # Assembly
        h_fused = torch.cat([h_vis, h_ctx, h_prior], dim=1)  # (B, 320)

        # Step 5: Prediction
        out = self.head(h_fused)

        alpha = out[:, 0]
        sigma_base = F.softplus(out[:, 1])
        sigma_growth = F.softplus(out[:, 2])

        return alpha, sigma_base, sigma_growth


# ==========================================
# 3. Training Utilities
# ==========================================


def laplace_log_likelihood_loss(y_true, y_pred, sigma):
    """
    Computes the negative modified Laplace Log Likelihood.
    We minimize this loss.
    """
    # Clipping as per metric definition
    sigma_clipped = torch.clamp(sigma, min=Config.CONFIDENCE_CLIP)

    delta = torch.abs(y_true - y_pred)
    delta = torch.clamp(delta, max=Config.MAX_ERROR)

    metric = -(np.sqrt(2) * delta) / sigma_clipped - torch.log(
        np.sqrt(2) * sigma_clipped
    )

    # Return negative metric for minimization
    return -torch.mean(metric)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    losses = AverageMeter()

    for batch in loader:
        img_ax = batch["img_ax"].to(device)
        img_cor = batch["img_cor"].to(device)
        tabular = batch["tabular"].to(device)
        delta_week = batch["delta_week"].to(device)
        target = batch["target"].to(device)
        baseline_fvc = batch["baseline_fvc"].to(device)

        optimizer.zero_grad()

        # Forward
        alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

        # Anchored Trajectory Logic
        fvc_pred = baseline_fvc + alpha * delta_week
        sigma_pred = sigma_base + sigma_growth * torch.abs(delta_week)

        # Loss
        loss = laplace_log_likelihood_loss(target, fvc_pred, sigma_pred)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), img_ax.size(0))

    return losses.avg


def validate(model, loader, device):
    model.eval()
    scores = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            delta_week = batch["delta_week"].to(device)
            target = batch["target"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)

            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

            fvc_pred = baseline_fvc + alpha * delta_week
            sigma_pred = sigma_base + sigma_growth * torch.abs(delta_week)

            score = score_function(target, fvc_pred, sigma_pred)
            scores.update(score, img_ax.size(0))

    return scores.avg


# ==========================================
# 4. Main Execution
# ==========================================


def run_training():
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader = get_extended_dataloaders()

    # Model
    model = TSCGNet().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # Training Loop
    best_score = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_score = validate(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Score: {val_score}"
        )

        # Save best
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"  -> New best model saved! Score: {best_score}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation Score: {best_score}")
    return best_model_path, test_loader


def generate_submission(model_path, test_loader):
    device = torch.device(Config.DEVICE)

    # Load model
    model = TSCGNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []

    print("Generating submission...")
    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)
            delta_week = batch["delta_week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            patient_weeks = batch["patient_week"]  # List of strings

            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

            fvc_pred = baseline_fvc + alpha * delta_week
            sigma_pred = sigma_base + sigma_growth * torch.abs(delta_week)

            # Move to CPU
            fvc_pred = fvc_pred.cpu().numpy()
            sigma_pred = sigma_pred.cpu().numpy()

            for pw, fvc, sigma in zip(patient_weeks, fvc_pred, sigma_pred):
                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": sigma})

    # Create DataFrame
    sub_df = pd.DataFrame(results)

    # Save
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    best_model_path, test_loader = run_training()
    generate_submission(best_model_path, test_loader)
