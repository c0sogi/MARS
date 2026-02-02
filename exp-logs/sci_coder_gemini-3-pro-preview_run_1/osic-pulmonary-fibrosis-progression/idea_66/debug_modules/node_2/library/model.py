import os
import torch
import torch.nn as nn
import torch.optim as optim
import timm
import numpy as np
import pandas as pd

from library.config import Config
from library.utils import AverageMeter, metric_function
from library.data import get_dataloaders

# ==========================================
# 1. Model Architecture
# ==========================================


class TabularEncoder(nn.Module):
    """
    Encodes the 8-dim tabular metadata into a shared latent vector.
    Structure: Linear(8->64) -> GeLU -> Linear(64->128) -> GeLU
    """

    def __init__(self, input_dim=8, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.GELU(), nn.Linear(64, output_dim), nn.GELU()
        )

    def forward(self, x):
        return self.net(x)


class TSCPNet(nn.Module):
    """
    Tri-Stream Context-Prior Network (TSCP-Net).

    Integrates two independent visual backbones with a shared latent tabular encoder
    via a Pre-Norm Symmetric Attention block. The readout is split into three
    balanced streams (Visual Context, Tabular Context, Clinical Prior) to prevent
    signal dilution before entering a non-linear parametric head.
    """

    def __init__(self):
        super().__init__()

        # 1. Independent Visual Backbones (EfficientNet-B0)
        # num_classes=0 with global_pool='avg' returns the 1280-dim feature vector
        self.backbone_ax = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )
        self.backbone_cor = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        # 2. Tabular Encoder
        self.tab_encoder = TabularEncoder(
            input_dim=8, output_dim=Config.SHARED_LATENT_DIM
        )

        # 3. Fusion Alignment
        # Project T_lat (128) to T_align (1280) + LayerNorm
        self.tab_align = nn.Linear(Config.SHARED_LATENT_DIM, Config.BACKBONE_DIM)
        self.tab_align_ln = nn.LayerNorm(Config.BACKBONE_DIM)

        # 4. Contextualization (Pre-Norm Self-Attention)
        # Sequence: [V_ax, V_cor, T_align]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.BACKBONE_DIM,
            nhead=4,
            dim_feedforward=2048,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_block = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 5. Tri-Stream Readout Projections
        # Stream 1: Visual Context (1280 -> 128)
        self.proj_vis = nn.Sequential(
            nn.Linear(Config.BACKBONE_DIM, Config.VISUAL_CONTEXT_DIM), nn.GELU()
        )
        # Stream 2: Tabular Context (1280 -> 64)
        self.proj_ctx = nn.Sequential(
            nn.Linear(Config.BACKBONE_DIM, Config.TABULAR_CONTEXT_DIM), nn.GELU()
        )
        # Stream 3: Clinical Prior (Raw 128) - No projection needed

        # 6. Non-Linear Parametric Head
        # Input: 128 (Vis) + 64 (Ctx) + 128 (Prior) = 320
        total_dim = (
            Config.VISUAL_CONTEXT_DIM
            + Config.TABULAR_CONTEXT_DIM
            + Config.SHARED_LATENT_DIM
        )
        self.head = nn.Sequential(
            nn.Linear(total_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),  # alpha, sigma_base, sigma_growth
        )

        self.softplus = nn.Softplus()

    def forward(self, img_ax, img_cor, meta):
        # A. Feature Extraction
        # Visual: (B, 1280)
        v_ax = self.backbone_ax(img_ax)
        v_cor = self.backbone_cor(img_cor)

        # Tabular: (B, 128)
        t_lat = self.tab_encoder(meta)

        # B. Alignment and Tokenization
        # Align Tabular: (B, 1280)
        t_align = self.tab_align_ln(self.tab_align(t_lat))

        # Stack: (B, 3, 1280) -> [Axial, Coronal, Tabular]
        tokens = torch.stack([v_ax, v_cor, t_align], dim=1)

        # C. Contextualization
        # (B, 3, 1280)
        tokens_prime = self.context_block(tokens)

        # Unpack
        v_ax_prime = tokens_prime[:, 0, :]
        v_cor_prime = tokens_prime[:, 1, :]
        t_align_prime = tokens_prime[:, 2, :]

        # D. Tri-Stream Readout
        # Stream 1: Visual Context (Mean of visual tokens)
        h_vis = self.proj_vis((v_ax_prime + v_cor_prime) / 2.0)  # (B, 128)

        # Stream 2: Tabular Context
        h_ctx = self.proj_ctx(t_align_prime)  # (B, 64)

        # Stream 3: Clinical Prior (Raw latent)
        # t_lat is (B, 128)

        # Assembly
        feat = torch.cat([h_vis, h_ctx, t_lat], dim=1)  # (B, 320)

        # E. Prediction
        out = self.head(feat)  # (B, 3)

        # Apply constraints
        alpha = out[:, 0]  # Slope
        sigma_base = self.softplus(out[:, 1])  # Base uncertainty
        sigma_growth = self.softplus(out[:, 2])  # Growth uncertainty

        return torch.stack([alpha, sigma_base, sigma_growth], dim=1)


# ==========================================
# 2. Loss Function
# ==========================================


def laplace_log_likelihood_loss(params, baseline_fvc, week_diff, true_fvc):
    """
    Computes the negative modified Laplace Log Likelihood.

    Formula:
        Loss = (sqrt(2) * delta_clipped) / sigma_clipped + ln(sqrt(2) * sigma_clipped)

    Args:
        params: (B, 3) -> [alpha, sigma_base, sigma_growth]
        baseline_fvc: (B,)
        week_diff: (B,)
        true_fvc: (B,)
    """
    alpha = params[:, 0]
    sigma_base = params[:, 1]
    sigma_growth = params[:, 2]

    # Predict FVC based on linear trajectory
    pred_fvc = baseline_fvc + alpha * week_diff

    # Predict Confidence (Sigma) based on linear growth
    confidence = sigma_base + sigma_growth * torch.abs(week_diff)

    # Clip Confidence (min 70 ml)
    sigma_clipped = torch.clamp(confidence, min=Config.MIN_CONFIDENCE)

    # Calculate Delta (Absolute Error)
    delta = torch.abs(true_fvc - pred_fvc)

    # Clip Delta (max 1000 ml)
    delta_clipped = torch.clamp(delta, max=Config.MAX_ERROR)

    # Metric Formula: - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
    # We want to maximize Metric, so we minimize Loss = -Metric
    sqrt_2 = np.sqrt(2)
    loss = (sqrt_2 * delta_clipped) / sigma_clipped + torch.log(sqrt_2 * sigma_clipped)

    return loss.mean()


# ==========================================
# 3. Training and Evaluation Logic
# ==========================================


def train_one_epoch(model, loader, optimizer, device):
    """
    Runs one epoch of training.
    """
    model.train()
    loss_meter = AverageMeter()

    for batch in loader:
        # Move inputs to device
        img_ax = batch["img_axial"].to(device)
        img_cor = batch["img_coronal"].to(device)
        meta = batch["meta"].to(device)
        week_diff = batch["week_diff"].to(device)
        baseline_fvc = batch["baseline_fvc"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward
        params = model(img_ax, img_cor, meta)

        # Loss
        loss = laplace_log_likelihood_loss(params, baseline_fvc, week_diff, target)

        # Backward
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), img_ax.size(0))

    return loss_meter.avg


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()
    metric_meter = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_axial"].to(device)
            img_cor = batch["img_coronal"].to(device)
            meta = batch["meta"].to(device)
            week_diff = batch["week_diff"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            target = batch["target"].to(device)

            # Forward
            params = model(img_ax, img_cor, meta)

            # Reconstruct Predictions
            alpha = params[:, 0]
            sigma_base = params[:, 1]
            sigma_growth = params[:, 2]

            pred_fvc = baseline_fvc + alpha * week_diff
            confidence = sigma_base + sigma_growth * torch.abs(week_diff)

            # Calculate Metric
            score = metric_function(target, pred_fvc, confidence)
            metric_meter.update(score, img_ax.size(0))

    return metric_meter.avg


def run_training():
    """
    Main training loop with early stopping and checkpointing.
    """
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, _ = get_dataloaders()

    # Model
    model = TSCPNet().to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # Loop
    best_score = -float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_score = validate(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Score: {val_score:.6f}"
        )

        # Checkpoint (Save if validation score improves)
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
            )
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Validation Score: {best_score:.6f}")


def generate_submission():
    """
    Generates the submission file using the best trained model.
    """
    device = torch.device(Config.DEVICE)

    # Load Data
    _, _, test_loader = get_dataloaders()

    # Load Model
    model = TSCPNet().to(device)
    weights_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(weights_path):
        print("No trained model found. Skipping submission generation.")
        return

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    results = []

    print("Generating submission...")
    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["img_axial"].to(device)
            img_cor = batch["img_coronal"].to(device)
            meta = batch["meta"].to(device)
            week_diff = batch["week_diff"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            patient_ids = batch["patient_id"]
            weeks = batch["week"]  # The target week

            # Forward
            params = model(img_ax, img_cor, meta)

            alpha = params[:, 0]
            sigma_base = params[:, 1]
            sigma_growth = params[:, 2]

            # Calculate Predictions
            pred_fvc = baseline_fvc + alpha * week_diff
            confidence = sigma_base + sigma_growth * torch.abs(week_diff)

            # Move to CPU
            pred_fvc = pred_fvc.cpu().numpy()
            confidence = confidence.cpu().numpy()
            weeks = weeks.cpu().numpy()

            for i in range(len(patient_ids)):
                pid = patient_ids[i]
                wk = int(weeks[i])
                fvc = pred_fvc[i]
                conf = confidence[i]

                patient_week = f"{pid}_{wk}"
                results.append(
                    {"Patient_Week": patient_week, "FVC": fvc, "Confidence": conf}
                )

    # Save
    df = pd.DataFrame(results)
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
