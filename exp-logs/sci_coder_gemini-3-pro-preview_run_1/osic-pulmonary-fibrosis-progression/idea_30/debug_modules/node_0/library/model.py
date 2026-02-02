import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from tqdm.auto import tqdm

from library.config import Config
from library.utils import score_function


# ==========================================
# Model Architecture
# ==========================================


class GatedTabularLayer(nn.Module):
    """
    Projects low-dimensional tabular data to high-dimensional embedding
    using a Gated Linear Unit (GLU) to reduce noise.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        # GLU halves the dimension, so we project to 2 * output_dim
        self.fc = nn.Linear(input_dim, output_dim * 2)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        # x: (B, input_dim) -> (B, output_dim * 2)
        out = self.fc(x)
        # GLU(dim=-1) splits into (B, output_dim) and gates it
        return F.glu(out, dim=-1)


class CVERNet(nn.Module):
    """
    Corrected Visual-Exclusive Residual Network (C-VER-Net).

    Features:
    - Independent EfficientNet-B0 backbones for Axial and Coronal views.
    - Gated Tabular Projection.
    - Symmetric Attention (Transformer) for context.
    - Visual-Exclusive Readout (isolating visual residual).
    - Prior-Anchored Parametric Head (predicting trajectory parameters).
    """

    def __init__(self):
        super().__init__()

        # 1. Independent High-Fidelity Backbones
        # num_classes=0 returns the pooled features (1280 for B0)
        self.backbone_ax = timm.create_model(
            Config.BACKBONE, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.backbone_cor = timm.create_model(
            Config.BACKBONE, pretrained=True, num_classes=0, global_pool="avg"
        )

        # Feature dimension for EfficientNet-B0 is 1280
        self.feature_dim = Config.HIDDEN_DIM

        # 2. Gated Tabular Expansion
        # Input dim is 7 (Age:1, Percent:1, Sex:2, Smoking:3)
        self.tabular_input_dim = 7
        self.tab_projection = GatedTabularLayer(
            self.tabular_input_dim, self.feature_dim
        )

        # 3. Symmetric Attention (Contextualization)
        # Pre-Norm configuration (norm_first=True) for stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=Config.NUM_ATTENTION_HEADS,
            dim_feedforward=2048,
            dropout=Config.DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.attention = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 4. Prior-Anchored Parametric Head
        # Input: Visual Residual (1280) + Raw Tabular (7) via skip connection
        head_input_dim = self.feature_dim + self.tabular_input_dim

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(128, 3),  # Alpha, Sigma_Base, Sigma_Growth
        )

        self._init_head()

    def _init_head(self):
        # Initialize final layer to produce reasonable starting values
        # Index 0: Alpha (Slope). Init near 0.
        # Index 1: Sigma Base. Init to ~100. Softplus(100) ~ 100.
        # Index 2: Sigma Growth. Init near 0.
        final_layer = self.head[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

        # Set bias for Sigma Base to 100.0 to start with reasonable confidence
        with torch.no_grad():
            final_layer.bias[1] = 100.0

    def forward(self, img_ax, img_cor, tabular, weeks, base_fvc, base_week):
        """
        Args:
            img_ax: (B, 3, 224, 224)
            img_cor: (B, 3, 224, 224)
            tabular: (B, 7)
            weeks: (B,) Target week
            base_fvc: (B,) Baseline FVC
            base_week: (B,) Baseline week
        Returns:
            fvc_pred: (B,)
            sigma_pred: (B,)
        """
        batch_size = img_ax.size(0)

        # --- 1. Feature Extraction ---
        # (B, 1280)
        feat_ax = self.backbone_ax(img_ax)
        feat_cor = self.backbone_cor(img_cor)

        # (B, 1280)
        feat_tab = self.tab_projection(tabular)

        # --- 2. Contextualization ---
        # Stack tokens: [Axial, Coronal, Tabular] -> (B, 3, 1280)
        tokens = torch.stack([feat_ax, feat_cor, feat_tab], dim=1)

        # Apply Self-Attention
        context_tokens = self.attention(tokens)

        # --- 3. Visual-Exclusive Readout ---
        # Isolate refined visual tokens (indices 0 and 1) and pool
        visual_refined = context_tokens[:, 0:2, :]  # (B, 2, 1280)
        visual_residual = torch.mean(visual_refined, dim=1)  # (B, 1280)

        # --- 4. Parametric Prediction ---
        # Concatenate with raw tabular features (Skip Connection)
        # (B, 1287)
        combined = torch.cat([visual_residual, tabular], dim=1)

        # Predict parameters
        params = self.head(combined)

        alpha = params[:, 0]  # Slope
        sigma_base = F.softplus(params[:, 1])  # Base uncertainty
        sigma_growth = F.softplus(params[:, 2])  # Growth uncertainty

        # --- 5. Trajectory Calculation ---
        # strictly adhering to: FVC = Base + alpha * dt
        dt = weeks - base_week  # (B,)

        fvc_pred = base_fvc + alpha * dt
        sigma_pred = sigma_base + sigma_growth * torch.abs(dt)

        return fvc_pred, sigma_pred


# ==========================================
# Training Logic
# ==========================================


def laplace_log_likelihood_loss(fvc_true, fvc_pred, sigma, device):
    """
    Differentiable implementation of the competition metric for loss.
    Loss = -Metric
    """
    # Constants
    sigma_clip_val = 70.0
    delta_clip_val = 1000.0
    sq2 = torch.sqrt(torch.tensor(2.0, device=device))

    # 1. Clip Confidence (Gradient flows if sigma > 70)
    # We use max to ensure we don't divide by small numbers and adhere to metric
    sigma_clipped = torch.clamp(sigma, min=sigma_clip_val)

    # 2. Clip Delta (Error)
    delta = torch.abs(fvc_true - fvc_pred)
    delta = torch.clamp(delta, max=delta_clip_val)

    # 3. Compute Metric
    # Metric = - (sqrt(2) * delta) / sigma - ln(sqrt(2) * sigma)
    # Loss = -Metric = (sqrt(2) * delta) / sigma + ln(sqrt(2) * sigma)
    loss = (sq2 * delta) / sigma_clipped + torch.log(sq2 * sigma_clipped)

    return torch.mean(loss)


def train_model(train_loader, val_loader):
    """
    Executes the training loop with Early Stopping.
    """
    device = Config.DEVICE
    model = CVERNet().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    best_metric = -float("inf")
    patience_counter = 0
    best_model_path = Config.BEST_MODEL_PATH

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # --- Training ---
        model.train()
        train_loss_accum = 0.0

        for batch in train_loader:
            # Move inputs to device
            img_ax = batch["img_axial"].to(device)
            img_cor = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)
            weeks = batch["weeks"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            base_week = batch["base_week"].to(device)

            optimizer.zero_grad()

            # Forward
            fvc_pred, sigma_pred = model(
                img_ax, img_cor, tabular, weeks, base_fvc, base_week
            )

            # Loss
            loss = laplace_log_likelihood_loss(target, fvc_pred, sigma_pred, device)

            loss.backward()
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation ---
        model.eval()
        val_preds = []
        val_sigmas = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                img_ax = batch["img_axial"].to(device)
                img_cor = batch["img_coronal"].to(device)
                tabular = batch["tabular"].to(device)
                target = batch["target"].to(device)
                weeks = batch["weeks"].to(device)
                base_fvc = batch["base_fvc"].to(device)
                base_week = batch["base_week"].to(device)

                fvc_pred, sigma_pred = model(
                    img_ax, img_cor, tabular, weeks, base_fvc, base_week
                )

                val_preds.extend(fvc_pred.cpu().numpy())
                val_sigmas.extend(sigma_pred.cpu().numpy())
                val_targets.extend(target.cpu().numpy())

        # Calculate Metric using the official scoring function
        val_score = score_function(
            np.array(val_targets), np.array(val_preds), np.array(val_sigmas)
        )

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Score: {val_score:.10f}"
        )

        # --- Early Stopping ---
        if val_score > best_metric:
            best_metric = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"  New best model saved! Score: {best_metric:.10f}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Validation Score: {best_metric:.10f}")
    return model


# ==========================================
# Inference Logic
# ==========================================


def predict_and_submit(test_loader):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    device = Config.DEVICE
    model = CVERNet().to(device)

    # Load best model
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        print("Loaded best model for inference.")
    else:
        print("Warning: Best model not found. Using random weights.")

    model.eval()

    results = []

    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["img_axial"].to(device)
            img_cor = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            weeks = batch["weeks"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            base_week = batch["base_week"].to(device)
            patient_ids = batch["patient_id"]  # List of IDs

            fvc_pred, sigma_pred = model(
                img_ax, img_cor, tabular, weeks, base_fvc, base_week
            )

            fvc_pred = fvc_pred.cpu().numpy()
            sigma_pred = sigma_pred.cpu().numpy()
            weeks_np = weeks.cpu().numpy()

            # Collect results
            for i in range(len(patient_ids)):
                pid = patient_ids[i]
                wk = int(weeks_np[i])
                fvc = fvc_pred[i]
                conf = sigma_pred[i]

                # Clip confidence as per submission requirement (metric does this, but good to be safe)
                conf = max(conf, 70.0)

                patient_week = f"{pid}_{wk}"
                results.append(
                    {"Patient_Week": patient_week, "FVC": fvc, "Confidence": conf}
                )

    # Create DataFrame
    sub_df = pd.DataFrame(results)

    # Ensure columns are correct
    sub_df = sub_df[["Patient_Week", "FVC", "Confidence"]]

    # Save
    sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")
