import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config
from library.utils import calculate_metric, seed_everything
from library.data import get_dataloaders, get_test_dataloader

# ==========================================
# 1. Model Components
# ==========================================


class TabularEncoder(nn.Module):
    """
    Projects raw tabular features (dim 6) up to visual dimensionality (1280).
    """

    def __init__(self, input_dim=6, output_dim=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.net(x)


class IndependentGating(nn.Module):
    """
    Generates a gating mask from raw tabular features.
    Output is strictly in [0, 1] via Sigmoid to filter visual channels.
    """

    def __init__(self, input_dim=6, output_dim=1280):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class TMIGN(nn.Module):
    """
    Tabular-Modulated Independent-Gating Network.
    Integrates orthogonal visual views with clinical priors via attention and independent gating.
    """

    def __init__(self):
        super().__init__()

        # 1. Independent Visual Backbones
        # EfficientNet-B0 outputs 1280-dim feature vector when num_classes=0
        self.backbone_ax = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0
        )
        self.backbone_cor = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0
        )

        feature_dim = Config.FEATURE_DIM  # 1280
        # Tabular input: Age_norm, Sex_enc, Smoke_Ex, Smoke_Never, Smoke_Current, Percent_norm
        tabular_dim = 6

        # 2. Tabular Encoder
        self.tab_encoder = TabularEncoder(input_dim=tabular_dim, output_dim=feature_dim)

        # 3. Modality Embeddings
        self.embed_ax = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.embed_cor = nn.Parameter(torch.zeros(1, 1, feature_dim))
        self.embed_tab = nn.Parameter(torch.zeros(1, 1, feature_dim))

        # Initialize embeddings
        nn.init.normal_(self.embed_ax, std=0.02)
        nn.init.normal_(self.embed_cor, std=0.02)
        nn.init.normal_(self.embed_tab, std=0.02)

        # 4. Symmetric Attention
        # batch_first=True -> (Batch, Seq, Feature)
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=8, batch_first=True
        )

        # 5. Independent Gating Mechanisms
        self.gate_ax = IndependentGating(input_dim=tabular_dim, output_dim=feature_dim)
        self.gate_cor = IndependentGating(input_dim=tabular_dim, output_dim=feature_dim)

        # 6. Prior-Anchored Head
        # Input: Modulated Visual (1280) + Raw Tabular (6)
        self.head = nn.Sequential(
            nn.Linear(feature_dim + tabular_dim, 512),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(512, 3),  # Alpha (Slope), Sigma_base, Sigma_growth
        )

    def forward(self, img_ax, img_cor, tab_raw):
        # --- 1. Feature Extraction ---
        # Visual features: (B, 1280)
        v_ax = self.backbone_ax(img_ax)
        v_cor = self.backbone_cor(img_cor)

        # Tabular features: (B, 1280)
        v_tab = self.tab_encoder(tab_raw)

        # --- 2. Modality Embedding Injection ---
        # Add learnable embeddings (broadcasting)
        v_ax = v_ax.unsqueeze(1) + self.embed_ax  # (B, 1, 1280)
        v_cor = v_cor.unsqueeze(1) + self.embed_cor  # (B, 1, 1280)
        v_tab = v_tab.unsqueeze(1) + self.embed_tab  # (B, 1, 1280)

        # --- 3. Symmetric Attention ---
        # Concatenate sequence: [Axial, Coronal, Tabular] -> (B, 3, 1280)
        seq = torch.cat([v_ax, v_cor, v_tab], dim=1)

        # Self-Attention
        attn_out, _ = self.attention(seq, seq, seq)

        # Split back
        v_ax_ctx = attn_out[:, 0, :]  # (B, 1280)
        v_cor_ctx = attn_out[:, 1, :]  # (B, 1280)
        # v_tab_ctx is unused for aggregation, but served as context in attention

        # --- 4. Independent Channel-Wise Recalibration ---
        # Generate masks from raw tabular data
        mask_ax = self.gate_ax(tab_raw)  # (B, 1280)
        mask_cor = self.gate_cor(tab_raw)  # (B, 1280)

        # Modulate and Aggregate
        v_vis = (v_ax_ctx * mask_ax) + (v_cor_ctx * mask_cor)  # (B, 1280)

        # --- 5. Prediction Head ---
        # Concatenate with raw tabular priors (Skip Connection)
        combined = torch.cat([v_vis, tab_raw], dim=1)  # (B, 1286)

        out = self.head(combined)

        # Parse outputs
        alpha = out[:, 0]
        sigma_base = F.softplus(out[:, 1])  # Enforce positivity
        sigma_growth = F.softplus(out[:, 2])  # Enforce positivity

        return alpha, sigma_base, sigma_growth


# ==========================================
# 2. Loss Function
# ==========================================


class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Direct optimization of the competition metric.
    Loss = -Metric (since we want to maximize metric).
    Includes robustness clipping.
    """

    def __init__(self):
        super().__init__()
        self.min_sigma = Config.MIN_SIGMA
        self.max_delta = Config.MAX_DELTA
        self.sqrt_2 = np.sqrt(2)

    def forward(self, pred_fvc, true_fvc, sigma):
        # Clip sigma
        sigma_clipped = torch.clamp(sigma, min=self.min_sigma)

        # Calculate absolute error
        delta = torch.abs(true_fvc - pred_fvc)

        # Clip error (Robustness against outliers)
        delta = torch.clamp(delta, max=self.max_delta)

        # Metric formula: - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
        # Loss = -Metric = (sqrt(2) * delta) / sigma + ln(sqrt(2) * sigma)
        loss = (self.sqrt_2 * delta) / sigma_clipped + torch.log(
            self.sqrt_2 * sigma_clipped
        )

        return torch.mean(loss)


# ==========================================
# 3. Training & Inference Logic
# ==========================================


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        img_ax = batch["img_ax"].to(device)
        img_cor = batch["img_cor"].to(device)
        tabular = batch["tabular"].to(device)

        target_fvc = batch["fvc"].to(device)
        weeks = batch["weeks"].to(device)
        base_fvc = batch["base_fvc"].to(device)
        base_week = batch["base_week"].to(device)

        optimizer.zero_grad()

        # Forward pass
        alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

        # Reconstruct prediction based on trajectory
        # FVC = Base_FVC + alpha * (Week - Base_Week)
        dt = weeks - base_week
        pred_fvc = base_fvc + alpha * dt

        # Confidence = Sigma_base + Sigma_growth * |Week - Base_Week|
        pred_sigma = sigma_base + sigma_growth * torch.abs(dt)

        # Compute loss
        loss = criterion(pred_fvc, target_fvc, pred_sigma)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img_ax.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_true = []
    all_pred = []
    all_sigma = []

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)

            target_fvc = batch["fvc"].to(device)
            weeks = batch["weeks"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            base_week = batch["base_week"].to(device)

            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

            dt = weeks - base_week
            pred_fvc = base_fvc + alpha * dt
            pred_sigma = sigma_base + sigma_growth * torch.abs(dt)

            loss = criterion(pred_fvc, target_fvc, pred_sigma)
            running_loss += loss.item() * img_ax.size(0)

            all_true.extend(target_fvc.cpu().numpy())
            all_pred.extend(pred_fvc.cpu().numpy())
            all_sigma.extend(pred_sigma.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    metric_score = calculate_metric(
        np.array(all_true), np.array(all_pred), np.array(all_sigma)
    )

    return avg_loss, metric_score


def run_training():
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Data
    train_loader, val_loader = get_dataloaders()

    # Model
    model = TMIGN().to(device)

    # Optimization
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )
    criterion = LaplaceLogLikelihoodLoss()

    # Training Loop
    best_metric = -float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {Config.EPOCHS} epochs on {device}...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metric = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Metric: {val_metric}"
        )

        # Checkpointing (Save best metric; metric is negative, higher is better)
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New Best Model Saved! Metric: {best_metric}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation Metric: {best_metric}")


def generate_submission():
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Model
    model = TMIGN().to(device)
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(best_model_path):
        print("No trained model found. Skipping submission generation.")
        return

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    # Load Test Data
    test_loader, test_df = get_test_dataloader()

    results = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tabular = batch["tabular"].to(device)

            # For test set, 'weeks' is the target prediction week
            target_weeks = batch["weeks"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            base_week = batch["base_week"].to(device)
            patient_ids = batch["patient_id"]

            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, tabular)

            dt = target_weeks - base_week
            pred_fvc = base_fvc + alpha * dt
            pred_sigma = sigma_base + sigma_growth * torch.abs(dt)

            # Move to CPU
            pred_fvc = pred_fvc.cpu().numpy()
            pred_sigma = pred_sigma.cpu().numpy()
            target_weeks = target_weeks.cpu().numpy()

            for i in range(len(patient_ids)):
                # Format: PatientID_Week
                pid = patient_ids[i]
                wk = int(target_weeks[i])
                patient_week = f"{pid}_{wk}"

                results.append(
                    {
                        "Patient_Week": patient_week,
                        "FVC": pred_fvc[i],
                        "Confidence": pred_sigma[i],
                    }
                )

    # Create DataFrame
    sub_df = pd.DataFrame(results)

    # Save
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
