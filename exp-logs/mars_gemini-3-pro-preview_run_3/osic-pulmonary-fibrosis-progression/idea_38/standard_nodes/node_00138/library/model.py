import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm

from library.config import Config
from library.utils import laplace_log_likelihood

# =========================================================================
# Model Architecture
# =========================================================================


class ClinicalStream(nn.Module):
    """
    Stream A: Supervised Clinical Anchor.
    Processes tabular clinical data and outputs a latent vector and a base prediction.
    """

    def __init__(self):
        super().__init__()
        # Input: Age, Sex, Smoking, RelativeTime (4 dims)
        self.net = nn.Sequential(
            nn.Linear(Config.CLINICAL_INPUT_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, Config.CLINICAL_LATENT_DIM),
        )
        # Auxiliary Head for Stream A
        # Outputs mu_base_raw and sigma_base_raw
        self.aux_head = nn.Linear(Config.CLINICAL_LATENT_DIM, 2)

    def forward(self, x):
        h_clin = self.net(x)
        base_raw = self.aux_head(h_clin)
        return h_clin, base_raw


class VisualStream(nn.Module):
    """
    Stream B: Visual Feature Extractor.
    Uses EfficientNet-B2 backbone with specific freezing logic.
    """

    def __init__(self):
        super().__init__()
        # Backbone: EfficientNet-B2
        # num_classes=0 and global_pool='avg' returns the pooled feature vector
        self.backbone = timm.create_model(
            Config.BACKBONE_ARCH,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="avg",
        )

        # Feature dimension of EfficientNet-B2 is 1408
        self.vis_dim = self.backbone.num_features

        # Freezing Logic: Unfreeze top two stages (blocks 5 and 6), freeze rest.
        # Structure: conv_stem -> bn1 -> blocks (0..6) -> conv_head -> bn2 -> classifier

        # 1. Freeze everything initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze specific layers
        for name, child in self.backbone.named_children():
            if name == "blocks":
                # child is a nn.Sequential of blocks
                for i, block in enumerate(child):
                    if i >= 5:  # Unfreeze blocks 5 and 6
                        for param in block.parameters():
                            param.requires_grad = True
            elif name in ["conv_head", "bn2"]:
                for param in child.parameters():
                    param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


class SCOSRNet(nn.Module):
    """
    Supervised Cascaded Output-Space Residual Network.
    Fuses Clinical Anchor and Visual Residuals.
    """

    def __init__(self):
        super().__init__()
        self.clinical_stream = ClinicalStream()
        self.visual_stream = VisualStream()

        # Fusion and Residual Head
        # Input: Visual Features (1408) + Clinical Latent (64) = 1472
        in_features = self.visual_stream.vis_dim + Config.CLINICAL_LATENT_DIM

        self.residual_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(128, 64),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(64, 2),
        )

        # Zero Initialization for the final layer of residual head
        # This ensures the model starts as the Clinical Anchor
        nn.init.constant_(self.residual_head[-1].weight, 0)
        nn.init.constant_(self.residual_head[-1].bias, 0)

    def forward(self, image, clinical):
        # 1. Stream A: Clinical Anchor
        h_clin, base_raw = self.clinical_stream(clinical)
        mu_base_raw = base_raw[:, 0].unsqueeze(1)
        sigma_base_raw = base_raw[:, 1].unsqueeze(1)

        # 2. Stream B: Visual Features
        vis_feat = self.visual_stream(image)

        # 3. Fusion
        fused = torch.cat([vis_feat, h_clin], dim=1)

        # 4. Residual Prediction
        residual = self.residual_head(fused)
        delta_mu = residual[:, 0].unsqueeze(1)
        delta_sigma = residual[:, 1].unsqueeze(1)

        # 5. Combination (Cascaded Output-Space Summation)
        mu_final = mu_base_raw + delta_mu

        # Sigma calculation:
        # We model log-variance dynamics. The residual adjusts the raw logit.
        # Softplus ensures positivity. Epsilon prevents numerical instability.
        sigma_final = F.softplus(sigma_base_raw + delta_sigma) + 1e-6

        # Aux outputs for loss (Clinical Stream trained independently)
        sigma_base = F.softplus(sigma_base_raw) + 1e-6

        return mu_final, sigma_final, mu_base_raw, sigma_base


# =========================================================================
# Training & Evaluation Logic
# =========================================================================


def evaluate_model(model, loader, device):
    """
    Evaluates the model on the validation set using the competition metric.
    Performs denormalization to ensure metric is calculated on original scale.
    """
    model.eval()
    total_metric = 0.0

    # Retrieve normalization stats from dataset
    fvc_mean = loader.dataset.stats.get("fvc_mean", 2500.0)
    fvc_std = loader.dataset.stats.get("fvc_std", 500.0)

    with torch.no_grad():
        for images, clinical, targets, _ in loader:
            images = images.to(device)
            clinical = clinical.to(device)
            targets = targets.to(device)  # Normalized targets

            mu_final, sigma_final, _, _ = model(images, clinical)

            # Denormalize predictions and targets
            mu_denorm = mu_final * fvc_std + fvc_mean
            sigma_denorm = sigma_final * fvc_std
            targets_denorm = targets * fvc_std + fvc_mean

            # Calculate metric (LLL)
            batch_metric = laplace_log_likelihood(
                targets_denorm, mu_denorm, sigma_denorm
            )
            total_metric += batch_metric.item() * images.size(0)

    return total_metric / len(loader.dataset)


def train_model(train_loader, val_loader):
    """
    Main training loop with differential learning rates and auxiliary supervision.
    """
    device = torch.device(Config.DEVICE)
    model = SCOSRNet().to(device)

    # Differential Learning Rates
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # Stats for denormalization in loss
    fvc_mean = train_loader.dataset.stats.get("fvc_mean", 2500.0)
    fvc_std = train_loader.dataset.stats.get("fvc_std", 500.0)

    best_metric = -float("inf")

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0

        for images, clinical, targets, _ in train_loader:
            images = images.to(device)
            clinical = clinical.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            mu_final, sigma_final, mu_base, sigma_base = model(images, clinical)

            # --- Loss Calculation ---
            # We must denormalize to use the competition metric as loss (due to fixed clipping)

            # Denormalize Final Predictions
            mu_final_dn = mu_final * fvc_std + fvc_mean
            sigma_final_dn = sigma_final * fvc_std

            # Denormalize Base Predictions
            mu_base_dn = mu_base * fvc_std + fvc_mean
            sigma_base_dn = sigma_base * fvc_std

            # Denormalize Targets
            targets_dn = targets * fvc_std + fvc_mean

            # Calculate LLL (Metric returns negative value, higher is better)
            # We want to minimize Loss, so Loss = -Metric
            metric_main = laplace_log_likelihood(
                targets_dn, mu_final_dn, sigma_final_dn
            )
            metric_aux = laplace_log_likelihood(targets_dn, mu_base_dn, sigma_base_dn)

            loss = -metric_main - (Config.AUX_LOSS_WEIGHT * metric_aux)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        val_metric = evaluate_model(model, val_loader, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Metric: {val_metric:.8f}"
        )

        # Save Best Model
        if val_metric > best_metric:
            best_metric = val_metric
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved! Metric: {val_metric:.8f}")

    return model


def predict(test_loader):
    """
    Generates predictions for the test set using the best saved model.
    Saves the result to submission.csv.
    """
    device = torch.device(Config.DEVICE)
    model = SCOSRNet().to(device)

    # Load best model
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        print("Loaded best model for inference.")
    else:
        print("Warning: Best model not found. Using current initialized weights.")

    model.eval()
    results = []

    # Stats for inverse transform
    fvc_mean = test_loader.dataset.stats.get("fvc_mean", 2500.0)
    fvc_std = test_loader.dataset.stats.get("fvc_std", 500.0)

    with torch.no_grad():
        for images, clinical, _, meta in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            clinical = clinical.to(device)

            mu_final, sigma_final, _, _ = model(images, clinical)

            mu_np = mu_final.cpu().numpy().flatten()
            sigma_np = sigma_final.cpu().numpy().flatten()

            # Inverse Transform
            pred_fvc = mu_np * fvc_std + fvc_mean
            pred_sigma = sigma_np * fvc_std

            # Extract metadata
            # meta is a dict where values are lists/tensors
            pat_weeks = meta["Patient_Week"]

            for i in range(len(pat_weeks)):
                pid_week = pat_weeks[i]
                fvc = pred_fvc[i]
                conf = pred_sigma[i]

                # Post-processing clip for submission
                conf = max(conf, 70)

                results.append(
                    {"Patient_Week": pid_week, "FVC": fvc, "Confidence": conf}
                )

    df_sub = pd.DataFrame(results)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
