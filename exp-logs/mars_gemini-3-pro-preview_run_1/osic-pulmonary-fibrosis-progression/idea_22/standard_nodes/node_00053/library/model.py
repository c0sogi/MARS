import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import get_score, seed_everything
from library.data import prepare_data


# ==========================================
# 1. Loss Function
# ==========================================
class LaplaceLogLikelihoodLoss(nn.Module):
    """
    Differentiable approximation of the competition metric.
    Metric: - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)
    We maximize metric, so we minimize -metric (Loss).
    Loss = (sqrt(2) * delta) / sigma_clipped + ln(sqrt(2) * sigma_clipped)
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred_fvc, pred_sigma, true_fvc):
        # pred_sigma is output of softplus, so it is strictly positive
        # Clip sigma to 70 as per metric requirements
        sigma_clipped = torch.clamp(pred_sigma, min=Config.CONFIDENCE_CLIP)

        # Calculate absolute error
        delta = torch.abs(true_fvc - pred_fvc)
        # Clip delta to 1000 as per metric requirements (robustness to outliers)
        delta_clipped = torch.clamp(delta, max=Config.MAX_ERROR_CLIP)

        sq2 = torch.sqrt(torch.tensor(2.0, device=pred_fvc.device))

        # Loss calculation
        loss = (sq2 * delta_clipped) / sigma_clipped + torch.log(sq2 * sigma_clipped)

        return torch.mean(loss)


# ==========================================
# 2. Model Components
# ==========================================
class EfficientNetEncoder(nn.Module):
    def __init__(self, model_name=Config.BACKBONE_NAME, pretrained=True):
        super().__init__()
        # Load EfficientNet, remove classifier
        # Global pooling is applied to get a vector output
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        # Native feature dimension (1280 for B0)
        self.out_dim = self.backbone.num_features

    def forward(self, x):
        # x: (B, 3, H, W) -> (B, 1280)
        return self.backbone(x)


class TabularEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class CASDAN(nn.Module):
    """
    Content-Adaptive Symmetric Dual-Axis Network
    """

    def __init__(self):
        super().__init__()

        # 1. Visual Backbones
        # Independent backbones for Axial and Coronal views
        self.enc_axial = EfficientNetEncoder()
        self.enc_coronal = EfficientNetEncoder()

        # 2. Tabular Encoder
        # Input dim = 7 (Age, Sex, Smoking(3), Percent, BaseFVC)
        self.enc_tabular = TabularEncoder(
            input_dim=7,
            hidden_dim=Config.TABULAR_HIDDEN_DIM,
            output_dim=Config.VISUAL_DIM,
        )

        # 3. Modality Embeddings
        # Learnable vectors to distinguish sources in the attention mechanism
        self.modality_ax = nn.Parameter(torch.randn(1, 1, Config.VISUAL_DIM))
        self.modality_cor = nn.Parameter(torch.randn(1, 1, Config.VISUAL_DIM))
        self.modality_tab = nn.Parameter(torch.randn(1, 1, Config.VISUAL_DIM))

        # 4. Symmetric Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=Config.VISUAL_DIM,
            num_heads=Config.NUM_HEADS,
            dropout=Config.DROPOUT,
            batch_first=True,
        )

        # 5. Content-Adaptive Gating
        # Input: Concat of 3 contextualized tokens (Axial, Coronal, Tabular) -> 3 * 1280
        self.gate_mlp = nn.Sequential(
            nn.Linear(3 * Config.VISUAL_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 2),  # Weights for [Axial, Coronal]
        )

        # 6. Prior-Anchored Regression Head
        # Input: Aggregated Visual(1280) + Raw Tabular(7)
        self.head = nn.Sequential(
            nn.Linear(Config.VISUAL_DIM + 7, 512),
            nn.ReLU(),
            nn.Linear(512, 3),  # Outputs: Alpha, Sigma_base, Sigma_growth
        )

    def forward(self, axial_img, coronal_img, tabular_raw):
        B = axial_img.size(0)

        # --- Feature Extraction ---
        v_ax = self.enc_axial(axial_img)  # (B, 1280)
        v_cor = self.enc_coronal(coronal_img)  # (B, 1280)
        v_tab = self.enc_tabular(tabular_raw)  # (B, 1280)

        # --- Modality Embedding ---
        # Add learnable embeddings to identify view/modality
        v_ax = v_ax.unsqueeze(1) + self.modality_ax
        v_cor = v_cor.unsqueeze(1) + self.modality_cor
        v_tab = v_tab.unsqueeze(1) + self.modality_tab

        # --- Symmetric Attention ---
        # Sequence: [Axial, Coronal, Tabular]
        seq = torch.cat([v_ax, v_cor, v_tab], dim=1)  # (B, 3, 1280)

        # Self-Attention (Contextualization)
        attn_out, _ = self.attention(seq, seq, seq)

        # Split back
        v_ax_ctx = attn_out[:, 0, :]
        v_cor_ctx = attn_out[:, 1, :]
        v_tab_ctx = attn_out[:, 2, :]

        # --- Content-Adaptive Gating ---
        # Concatenate contextualized tokens to determine weights
        h_ctx = torch.cat([v_ax_ctx, v_cor_ctx, v_tab_ctx], dim=1)  # (B, 3840)

        # Predict softmax weights for visual views
        weights = F.softmax(self.gate_mlp(h_ctx), dim=1)  # (B, 2)
        w_ax = weights[:, 0:1]
        w_cor = weights[:, 1:2]

        # Aggregate Visual Features
        v_vis = w_ax * v_ax_ctx + w_cor * v_cor_ctx  # (B, 1280)

        # --- Prior-Anchored Head ---
        # Skip connection: Concat aggregated visual with RAW tabular priors
        combined = torch.cat([v_vis, tabular_raw], dim=1)  # (B, 1287)

        out = self.head(combined)

        # Output Parameters
        alpha = out[:, 0]
        sigma_base = F.softplus(out[:, 1])  # Ensure positivity
        sigma_growth = F.softplus(out[:, 2])  # Ensure positivity

        return alpha, sigma_base, sigma_growth


# ==========================================
# 3. Training & Inference Functions
# ==========================================
def train_model():
    seed_everything(Config.SEED)

    # Load Data
    train_dataset = prepare_data("train", load_cached_data=True)
    val_dataset = prepare_data("val", load_cached_data=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model Setup
    model = CASDAN().to(Config.DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.SCHEDULER_T_MAX, eta_min=Config.SCHEDULER_MIN_LR
    )
    criterion = LaplaceLogLikelihoodLoss()

    best_score = -float("inf")
    patience_counter = 0

    print(f"Starting training on {Config.DEVICE}...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0

        # Training Loop
        for batch in train_loader:
            ax = batch["axial"].to(Config.DEVICE)
            cor = batch["coronal"].to(Config.DEVICE)
            tab = batch["tabular"].to(Config.DEVICE)
            target = batch["target"].to(Config.DEVICE)
            weeks = batch["week"].to(Config.DEVICE)

            optimizer.zero_grad()

            alpha, sigma_base, sigma_growth = model(ax, cor, tab)

            # Reconstruct Baseline FVC from normalized tabular data
            # Index 6 is BaseFVC (Age, Sex, Smk(3), Pct, BaseFVC)
            # Normalization was: (x - 2500) / 1000
            base_fvc_rec = tab[:, 6] * 1000.0 + 2500.0

            # Predict FVC: Base + alpha * delta_t
            # For training, delta_t is simply 'weeks' (relative to baseline)
            fvc_pred = base_fvc_rec + alpha * weeks

            # Predict Confidence: Base + Growth * |delta_t|
            sigma_pred = sigma_base + sigma_growth * torch.abs(weeks)

            loss = criterion(fvc_pred, sigma_pred, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Validation Loop
        model.eval()
        val_preds = []
        val_sigmas = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                ax = batch["axial"].to(Config.DEVICE)
                cor = batch["coronal"].to(Config.DEVICE)
                tab = batch["tabular"].to(Config.DEVICE)
                target = batch["target"].to(Config.DEVICE)
                weeks = batch["week"].to(Config.DEVICE)

                alpha, sigma_base, sigma_growth = model(ax, cor, tab)

                base_fvc_rec = tab[:, 6] * 1000.0 + 2500.0
                fvc_pred = base_fvc_rec + alpha * weeks
                sigma_pred = sigma_base + sigma_growth * torch.abs(weeks)

                val_preds.extend(fvc_pred.cpu().numpy())
                val_sigmas.extend(sigma_pred.cpu().numpy())
                val_targets.extend(target.cpu().numpy())

        # Calculate Competition Metric
        val_score = get_score(val_targets, val_preds, val_sigmas)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Score: {val_score:.8f}"
        )

        # Checkpointing & Early Stopping
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Best Validation Score: {best_score}")


def predict():
    seed_everything(Config.SEED)

    # Load Test Data
    test_dataset = prepare_data("test", load_cached_data=True)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Model
    model = CASDAN().to(Config.DEVICE)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        )
    else:
        print("Warning: No trained model found. Predictions will be random.")

    model.eval()

    results = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            ax = batch["axial"].to(Config.DEVICE)
            cor = batch["coronal"].to(Config.DEVICE)
            tab = batch["tabular"].to(Config.DEVICE)

            # Meta info comes as dict of lists/tensors
            meta_base_fvc = batch["meta"]["Baseline_FVC"].to(Config.DEVICE)
            meta_base_week = batch["meta"]["Baseline_Week"].to(Config.DEVICE)
            meta_pred_week = batch["meta"]["Predict_Week"].to(Config.DEVICE)
            meta_patient_week = batch["meta"]["Patient_Week"]  # List of strings

            alpha, sigma_base, sigma_growth = model(ax, cor, tab)

            # Calculate Delta T
            delta_t = meta_pred_week - meta_base_week

            # Predict
            fvc_pred = meta_base_fvc + alpha * delta_t
            sigma_pred = sigma_base + sigma_growth * torch.abs(delta_t)

            # Clip Confidence (Final Submission Requirement)
            sigma_pred = torch.clamp(sigma_pred, min=Config.CONFIDENCE_CLIP)

            # Store results
            fvc_np = fvc_pred.cpu().numpy()
            sigma_np = sigma_pred.cpu().numpy()

            for i in range(len(meta_patient_week)):
                results.append(
                    {
                        "Patient_Week": meta_patient_week[i],
                        "FVC": fvc_np[i],
                        "Confidence": sigma_np[i],
                    }
                )

    # Save Submission
    sub_df = pd.DataFrame(results)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
