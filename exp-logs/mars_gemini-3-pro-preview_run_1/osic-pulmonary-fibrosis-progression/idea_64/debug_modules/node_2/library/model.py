import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import timm
from tqdm import tqdm

from library.config import Config
from library.data import LungDataset, get_transforms
from library.utils import calculate_metric

# ==========================================
# 1. Model Components
# ==========================================


class TabularEncoder(nn.Module):
    """
    Encodes clinical features into a Shared Latent Vector (T_lat).
    Structure: Deep MLP (Linear -> GeLU -> Linear -> GeLU).
    """

    def __init__(self, input_dim=6, latent_dim=128):
        super().__init__()
        hidden_dim = 64
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.mlp(x)


class ContextAttention(nn.Module):
    """
    Fuses visual and tabular tokens using Pre-Norm Multi-Head Self-Attention.
    """

    def __init__(self, embed_dim=1280, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, Seq_Len, Embed_Dim)

        # 1. Self Attention (Pre-Norm)
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # 2. Feed Forward (Pre-Norm)
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x


class TSBCNet(nn.Module):
    """
    Tri-Stream Balanced-Context Network.
    """

    def __init__(self):
        super().__init__()

        # --- 1. Independent Low-Capacity Visual Backbones ---
        # EfficientNet-B0. num_classes=0 returns the pooled feature vector (1280 dim).
        self.backbone_ax = timm.create_model(
            Config.BACKBONE, pretrained=True, num_classes=0
        )
        self.backbone_cor = timm.create_model(
            Config.BACKBONE, pretrained=True, num_classes=0
        )

        # --- 2. Shared-Latent Tabular Encoder ---
        self.tabular_encoder = TabularEncoder(input_dim=6, latent_dim=Config.LATENT_DIM)

        # --- 3. Normalized Bifurcated Flow ---
        # Projects T_lat (128) to T_align (1280) for attention
        self.tabular_proj = nn.Sequential(
            nn.Linear(Config.LATENT_DIM, Config.BACKBONE_OUT_DIM),
            nn.LayerNorm(Config.BACKBONE_OUT_DIM),
        )

        # --- 4. Pre-Norm Symmetric Attention ---
        self.context_attention = ContextAttention(embed_dim=Config.BACKBONE_OUT_DIM)

        # --- 5. Tri-Stream Balanced Readout ---
        # Stream 1: Visual Context (1280 -> 64)
        self.stream_vis_proj = nn.Sequential(
            nn.Linear(Config.BACKBONE_OUT_DIM, Config.HIDDEN_DIM), nn.GELU()
        )

        # Stream 2: Tabular Context (1280 -> 64)
        self.stream_ctx_proj = nn.Sequential(
            nn.Linear(Config.BACKBONE_OUT_DIM, Config.HIDDEN_DIM), nn.GELU()
        )

        # Stream 3: Clinical Prior (Raw T_lat, 128 dim) - Used directly

        # --- 6. Non-Linear Parametric Head ---
        # Input: 64 (Vis) + 64 (Ctx) + 128 (Prior) = 256
        fusion_dim = Config.HIDDEN_DIM + Config.HIDDEN_DIM + Config.LATENT_DIM
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),  # alpha, sigma_base, sigma_growth
        )

    def forward(self, img_ax, img_cor, tabular):
        """
        Returns trajectory parameters: [alpha, sigma_base, sigma_growth]
        """
        # 1. Feature Extraction
        v_ax = self.backbone_ax(img_ax)  # (B, 1280)
        v_cor = self.backbone_cor(img_cor)  # (B, 1280)
        t_lat = self.tabular_encoder(tabular)  # (B, 128)

        # 2. Alignment
        t_align = self.tabular_proj(t_lat)  # (B, 1280)

        # 3. Contextualization
        # Stack: [Axial, Coronal, Tabular]
        tokens = torch.stack([v_ax, v_cor, t_align], dim=1)  # (B, 3, 1280)
        ctx_tokens = self.context_attention(tokens)

        v_ax_ctx = ctx_tokens[:, 0, :]
        v_cor_ctx = ctx_tokens[:, 1, :]
        t_align_ctx = ctx_tokens[:, 2, :]

        # 4. Tri-Stream Processing
        # Stream 1: Visual (Mean of views)
        v_ctx_mean = (v_ax_ctx + v_cor_ctx) / 2.0
        h_vis = self.stream_vis_proj(v_ctx_mean)  # (B, 64)

        # Stream 2: Tabular Context
        h_ctx = self.stream_ctx_proj(t_align_ctx)  # (B, 64)

        # Stream 3: Clinical Prior (t_lat)

        # 5. Fusion & Prediction
        h_fused = torch.cat([h_vis, h_ctx, t_lat], dim=1)  # (B, 256)
        raw_preds = self.head(h_fused)

        # 6. Activation
        alpha = raw_preds[:, 0]  # Unbounded slope
        sigma_base = F.softplus(raw_preds[:, 1])  # Positive
        sigma_growth = F.softplus(raw_preds[:, 2])  # Positive

        return torch.stack([alpha, sigma_base, sigma_growth], dim=1)


# ==========================================
# 2. Loss Function
# ==========================================


def laplace_log_likelihood_loss(y_true, y_pred, sigma):
    """
    Differentiable approximation of the competition metric.
    Minimizes: (sqrt(2) * Delta) / Sigma_clipped + ln(sqrt(2) * Sigma_clipped)
    """
    MAX_ERROR = 1000.0
    MIN_CONFIDENCE = 70.0
    SQRT_2 = 1.41421356

    # Clip sigma
    sigma_clipped = torch.clamp(sigma, min=MIN_CONFIDENCE)

    # Calculate Delta with clipping
    abs_diff = torch.abs(y_true - y_pred)
    delta = torch.clamp(abs_diff, max=MAX_ERROR)

    # Metric terms (Negative of the score formula)
    term1 = (SQRT_2 * delta) / sigma_clipped
    term2 = torch.log(SQRT_2 * sigma_clipped)

    loss = torch.mean(term1 + term2)
    return loss


# ==========================================
# 3. Training & Inference Logic
# ==========================================


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        img_ax = batch["image_axial"].to(device)
        img_cor = batch["image_coronal"].to(device)
        tabular = batch["tabular"].to(device)
        target = batch["target"].to(device)

        # Meta info for trajectory calculation
        # Note: 'meta' is a dict of lists/tensors from collate
        week_num = torch.tensor(batch["meta"]["Week_Num"]).float().to(device)
        base_fvc = torch.tensor(batch["meta"]["Baseline_FVC"]).float().to(device)
        base_week = torch.tensor(batch["meta"]["Baseline_Week"]).float().to(device)

        optimizer.zero_grad()

        # Forward pass -> Parameters
        preds = model(img_ax, img_cor, tabular)
        alpha = preds[:, 0]
        sigma_base = preds[:, 1]
        sigma_growth = preds[:, 2]

        # Calculate Trajectory
        dt = week_num - base_week
        fvc_pred = base_fvc + alpha * dt
        sigma_pred = sigma_base + sigma_growth * torch.abs(dt)

        # Loss
        loss = laplace_log_likelihood_loss(target, fvc_pred, sigma_pred)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img_ax.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    all_true = []
    all_pred = []
    all_sigma = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].float().numpy()

            week_num = torch.tensor(batch["meta"]["Week_Num"]).float().to(device)
            base_fvc = torch.tensor(batch["meta"]["Baseline_FVC"]).float().to(device)
            base_week = torch.tensor(batch["meta"]["Baseline_Week"]).float().to(device)

            # Forward
            preds = model(img_ax, img_cor, tabular)
            alpha = preds[:, 0]
            sigma_base = preds[:, 1]
            sigma_growth = preds[:, 2]

            # Trajectory
            dt = week_num - base_week
            fvc_pred = base_fvc + alpha * dt
            sigma_pred = sigma_base + sigma_growth * torch.abs(dt)

            all_true.extend(target)
            all_pred.extend(fvc_pred.cpu().numpy())
            all_sigma.extend(sigma_pred.cpu().numpy())

    score = calculate_metric(
        np.array(all_true), np.array(all_pred), np.array(all_sigma)
    )
    return score


def predict_submission(model, loader, device):
    model.eval()
    results = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            tabular = batch["tabular"].to(device)

            # Meta
            patients = batch["meta"]["Patient"]
            week_num = torch.tensor(batch["meta"]["Week_Num"]).float().to(device)
            base_fvc = torch.tensor(batch["meta"]["Baseline_FVC"]).float().to(device)
            base_week = torch.tensor(batch["meta"]["Baseline_Week"]).float().to(device)

            # Forward
            preds = model(img_ax, img_cor, tabular)
            alpha = preds[:, 0]
            sigma_base = preds[:, 1]
            sigma_growth = preds[:, 2]

            # Trajectory
            dt = week_num - base_week
            fvc_pred = base_fvc + alpha * dt
            sigma_pred = sigma_base + sigma_growth * torch.abs(dt)

            # Collect results
            fvc_np = fvc_pred.cpu().numpy()
            sigma_np = sigma_pred.cpu().numpy()
            week_np = week_num.cpu().numpy().astype(int)

            for i in range(len(patients)):
                pid = patients[i]
                wk = week_np[i]
                fvc = fvc_np[i]
                conf = sigma_np[i]

                # Clip confidence for submission
                conf = max(conf, 70.0)

                results.append(
                    {"Patient_Week": f"{pid}_{wk}", "FVC": fvc, "Confidence": conf}
                )

    return pd.DataFrame(results)


def run_training_pipeline():
    """
    Main entry point to run training and generate submission.
    """
    Config.setup()
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading Metadata...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    if Config.DEBUG:
        print(f"DEBUG MODE: Using {Config.DEBUG_SAMPLES} samples.")
        train_df = train_df.head(Config.DEBUG_SAMPLES)
        val_df = val_df.head(Config.DEBUG_SAMPLES)
        # Test usually needs all rows for submission, but we can subset for speed check
        # test_df = test_df.head(Config.DEBUG_SAMPLES)

    # 2. Datasets & Loaders
    print("Initializing Datasets (this may take time to cache images)...")
    train_ds = LungDataset(train_df, mode="train", transform=get_transforms("train"))
    val_ds = LungDataset(val_df, mode="val", transform=get_transforms("val"))
    test_ds = LungDataset(test_df, mode="test", transform=get_transforms("test"))

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Model Setup
    model = TSBCNet().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # 4. Training Loop
    best_score = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print("Starting Training...")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_score = validate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Score: {val_score:.6f}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New Best Model Saved! Score: {best_score:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Inference
    print("Generating Submission...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    sub_df = predict_submission(model, test_loader, device)

    # Ensure correct format
    sub_df = sub_df[["Patient_Week", "FVC", "Confidence"]]
    output_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
