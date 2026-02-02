import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, score_function
from library.data import get_dataloaders

# ==========================================
# Model Architecture
# ==========================================


class VisualBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        # EfficientNet-B0: 1280 features at the final layer (GAP)
        self.net = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0, global_pool="avg"
        )

    def forward(self, x):
        # x: (B, 3, 224, 224) -> (B, 1280)
        return self.net(x)


class TabularEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(Config.TABULAR_INPUT_DIM, Config.TABULAR_HIDDEN_DIM),
            nn.LayerNorm(Config.TABULAR_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(Config.TABULAR_HIDDEN_DIM, Config.VISUAL_DIM),
            nn.LayerNorm(Config.VISUAL_DIM),
        )

    def forward(self, x):
        # x: (B, 4) -> (B, 1280)
        return self.mlp(x)


class VisuallyContextualizedNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Independent Visual Backbones
        self.axial_backbone = VisualBackbone()
        self.coronal_backbone = VisualBackbone()

        # 2. Up-Projected Tabular Embedding
        self.tabular_encoder = TabularEncoder()

        # 3. Symmetric Attention Fusion
        self.attn = nn.MultiheadAttention(
            embed_dim=Config.VISUAL_DIM,
            num_heads=Config.ATTENTION_HEADS,
            batch_first=True,
            dropout=0.1,
        )

        # 4. Parametric Head
        # Input: Visual Residual (1280) + Raw Tabular (4) via Skip Connection
        head_in_dim = Config.VISUAL_DIM + Config.TABULAR_INPUT_DIM

        self.head_hidden = nn.Sequential(
            nn.Linear(head_in_dim, 512), nn.GELU(), nn.Dropout(0.1)
        )

        # Predicts: [FVC_Base, Alpha, Sigma_Base_Raw, Sigma_Growth_Raw]
        self.head_out = nn.Linear(512, 4)

        self._init_weights()

    def _init_weights(self):
        # Initialize head to reasonable starting values
        # FVC_Base: ~2500 (Bias 2500)
        # Alpha: ~-5 (Bias -5)
        # Sigma_Base: ~100 (Softplus^-1(100) approx 100)
        # Sigma_Growth: ~1 (Softplus^-1(1) approx 0.55)

        nn.init.constant_(self.head_out.bias[0], 2500.0)
        nn.init.constant_(self.head_out.bias[1], Config.INIT_ALPHA)
        nn.init.constant_(self.head_out.bias[2], 100.0)
        nn.init.constant_(self.head_out.bias[3], 0.55)

        # Small random weights for the linear layer to allow learning
        nn.init.normal_(self.head_out.weight, mean=0.0, std=0.01)

    def forward(self, img_axial, img_coronal, tabular, weeks):
        batch_size = img_axial.shape[0]

        # --- Feature Extraction ---
        # (B, 1280)
        v_ax = self.axial_backbone(img_axial)
        v_cor = self.coronal_backbone(img_coronal)

        # (B, 1280)
        v_tab = self.tabular_encoder(tabular)

        # --- Fusion (Symmetric Attention) ---
        # Stack tokens: [Axial, Coronal, Tabular] -> (B, 3, 1280)
        tokens = torch.stack([v_ax, v_cor, v_tab], dim=1)

        # Self-Attention
        # attn_output: (B, 3, 1280)
        attn_output, _ = self.attn(tokens, tokens, tokens)

        # --- Visual-Selective Readout ---
        # Extract contextualized visual tokens (indices 0 and 1) and pool
        # We ignore the tabular token (index 2) for the residual calculation
        v_res = torch.mean(attn_output[:, 0:2, :], dim=1)  # (B, 1280)

        # --- Skip Connection & Head ---
        # Concatenate Visual Residual with RAW tabular features
        # (B, 1280 + 4)
        combined = torch.cat([v_res, tabular], dim=1)

        hidden = self.head_hidden(combined)
        out = self.head_out(hidden)

        # Unpack predictions
        fvc_base = out[:, 0]
        alpha = out[:, 1]
        sigma_base_raw = out[:, 2]
        sigma_growth_raw = out[:, 3]

        # Constraints
        sigma_base = F.softplus(sigma_base_raw)
        sigma_growth = F.softplus(sigma_growth_raw)

        # --- Parametric Inference ---
        # FVC = Base + Alpha * Weeks
        # Confidence = Base_Conf + Growth_Conf * |Weeks|

        weeks = weeks.view(-1)  # Ensure flat

        fvc_pred = fvc_base + alpha * weeks
        confidence = sigma_base + sigma_growth * torch.abs(weeks)

        return {
            "fvc": fvc_pred,
            "confidence": confidence,
            "alpha": alpha,
            "sigma_base": sigma_base,
            "sigma_growth": sigma_growth,
        }


# ==========================================
# Training & Evaluation Logic
# ==========================================


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch in loader:
        img_axial = batch["img_axial"].to(device)
        img_coronal = batch["img_coronal"].to(device)
        tabular = batch["tabular"].to(device)
        weeks = batch["weeks"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        output = model(img_axial, img_coronal, tabular, weeks)
        fvc_pred = output["fvc"]
        confidence = output["confidence"]

        # Metric is negative, we want to maximize it.
        # Loss = -Metric (minimize negative metric)
        # Note: score_function returns a scalar numpy value usually, but here we need differentiable loss.
        # We re-implement the metric logic with torch tensors for the loss.

        sigma_clipped = torch.clamp(confidence, min=Config.MIN_CONFIDENCE)
        abs_error = torch.abs(target - fvc_pred)
        delta = torch.clamp(abs_error, max=Config.MAX_ERROR)

        sqrt_2 = 1.41421356
        metric = -(sqrt_2 * delta) / sigma_clipped - torch.log(sqrt_2 * sigma_clipped)
        loss = -torch.mean(metric)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * img_axial.size(0)

    if scheduler:
        scheduler.step()

    return total_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    all_targets = []
    all_preds = []
    all_sigmas = []

    with torch.no_grad():
        for batch in loader:
            img_axial = batch["img_axial"].to(device)
            img_coronal = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            weeks = batch["weeks"].to(device)
            target = batch["target"].to(device)

            output = model(img_axial, img_coronal, tabular, weeks)

            all_targets.append(target.cpu().numpy())
            all_preds.append(output["fvc"].cpu().numpy())
            all_sigmas.append(output["confidence"].cpu().numpy())

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)
    all_sigmas = np.concatenate(all_sigmas)

    score = score_function(all_targets, all_preds, all_sigmas)
    return score


def train_model(load_cached_data=True):
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Data
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=load_cached_data
    )

    # Model
    model = VisuallyContextualizedNet().to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # Training Loop
    best_score = -float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training on {device}...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_score = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Score: {val_score:.8f}"
        )

        # Checkpoint
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print("  -> New Best Model Saved!")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training Complete. Best Validation Score: {best_score:.8f}")

    # Load best model for inference
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    return model, test_loader


def generate_submission(model, test_loader, device):
    model.eval()
    results = []

    print("Generating predictions for test set...")
    with torch.no_grad():
        for batch in test_loader:
            img_axial = batch["img_axial"].to(device)
            img_coronal = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            weeks = batch["weeks"].to(device)
            patient_weeks = batch["patient_week"]

            output = model(img_axial, img_coronal, tabular, weeks)

            fvc_preds = output["fvc"].cpu().numpy()
            conf_preds = output["confidence"].cpu().numpy()

            # Clip confidence as per submission requirement (though metric handles it, submission should be clean)
            conf_preds = np.maximum(conf_preds, Config.MIN_CONFIDENCE)

            for pw, fvc, conf in zip(patient_weeks, fvc_preds, conf_preds):
                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

    df_sub = pd.DataFrame(results)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main(load_cached_data=True):
    # Ensure working directory exists
    os.makedirs(Config.IDEA_DIR, exist_ok=True)

    model, test_loader = train_model(load_cached_data=load_cached_data)
    generate_submission(model, test_loader, torch.device(Config.DEVICE))
