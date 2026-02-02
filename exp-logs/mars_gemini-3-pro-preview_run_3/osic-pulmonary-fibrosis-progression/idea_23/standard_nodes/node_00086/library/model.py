import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
import os
import numpy as np
from library.config import Config
from library.utils import AverageMeter, score_function, seed_everything

# -------------------------------------------------------------------------
# Model Architecture
# -------------------------------------------------------------------------


class MCDSRNet(nn.Module):
    def __init__(self):
        super(MCDSRNet, self).__init__()

        # 1. Backbone: EfficientNet-B2
        # Native resolution 260x260, Output channels 1408
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME, pretrained=True, num_classes=0, global_pool=""
        )

        # Freezing Strategy: Freeze all, then unfreeze top 2 stages
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze specific layers: conv_head, bn2, and last 2 blocks
        # EfficientNet implementation in timm typically has: conv_stem, bn1, blocks, conv_head, bn2
        for param in self.backbone.conv_head.parameters():
            param.requires_grad = True
        for param in self.backbone.bn2.parameters():
            param.requires_grad = True

        # Unfreeze last 2 blocks (indices 5 and 6 for B2 which has 7 blocks)
        # We iterate through the blocks container
        num_blocks = len(self.backbone.blocks)
        for i in range(num_blocks - 2, num_blocks):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = True

        # Feature dimensions
        self.n_features = self.backbone.num_features  # 1408 for B2

        # Image Projection
        self.img_projector = nn.Linear(self.n_features, Config.PROJECTION_DIM)

        # 2. Stream A: Over-Parameterized Clinical Anchor
        # Input: 6 features
        self.stream_a = nn.Sequential(
            nn.Linear(Config.N_TABULAR_FEATURES, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.PROJECTION_DIM),
        )

        # 3. Stream B: Visual Interaction Stream
        # Input: Image Projection (64) + Clinical (6) = 70
        self.stream_b = nn.Sequential(
            nn.Linear(
                Config.PROJECTION_DIM + Config.N_TABULAR_FEATURES, Config.HIDDEN_DIM
            ),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.PROJECTION_DIM),
        )

        # 4. Shared Head
        # Projects fused latent (64) to mu (1) and raw_sigma (1)
        self.head = nn.Linear(Config.PROJECTION_DIM, 2)

    def forward(self, img, tab):
        # --- Image Branch ---
        # img: [B, 3, 260, 260]
        x = self.backbone.forward_features(img)  # [B, 1408, H, W]
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)  # [B, 1408]
        img_emb = self.img_projector(x)  # [B, 64]

        # --- Stream A (Clinical) ---
        # tab: [B, 6]
        out_a = self.stream_a(tab)  # [B, 64]

        # --- Stream B (Interaction) ---
        # Concatenate image embedding and tabular features
        combined_input = torch.cat([img_emb, tab], dim=1)  # [B, 70]
        out_b = self.stream_b(combined_input)  # [B, 64]

        # --- Fusion (Residual Summation) ---
        h_final = out_a + out_b  # [B, 64]

        # --- Head ---
        out = self.head(h_final)  # [B, 2]

        mu = out[:, 0]
        raw_sigma = out[:, 1]

        # Uncertainty constraint: softplus + epsilon
        sigma = F.softplus(raw_sigma) + 1e-6

        return mu, sigma


# -------------------------------------------------------------------------
# Loss Function
# -------------------------------------------------------------------------


def metric_aligned_loss(mu_pred, sigma_pred, target):
    """
    Metric-Aligned Laplace Log Likelihood Loss.
    L = (sqrt(2) * |y_true - y_pred|) / sigma + ln(sqrt(2) * sigma)
    Computed on scaled values for stability.
    """
    delta = torch.abs(target - mu_pred)
    sqrt_2 = math.sqrt(2)

    # Note: We do not clip delta or sigma here for the optimization objective,
    # allowing gradients to flow naturally.
    loss = (sqrt_2 * delta) / sigma_pred + torch.log(sqrt_2 * sigma_pred)
    return loss.mean()


# -------------------------------------------------------------------------
# Training & Evaluation
# -------------------------------------------------------------------------


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_meter = AverageMeter()

    for batch in loader:
        img = batch["image"].to(device)
        tab = batch["tabular"].to(device)
        target = batch["target"].to(device).squeeze(-1)  # [B]

        optimizer.zero_grad()

        mu, sigma = model(img, tab)

        loss = metric_aligned_loss(mu, sigma, target)
        loss.backward()

        optimizer.step()

        loss_meter.update(loss.item(), img.size(0))

    return loss_meter.avg


def validate(model, loader, device, stats):
    model.eval()
    loss_meter = AverageMeter()
    metric_meter = AverageMeter()

    # Stats for inverse transform
    fvc_mean = stats["FVC_mean"]
    fvc_std = stats["FVC_std"]

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            tab = batch["tabular"].to(device)
            target_scaled = batch["target"].to(device).squeeze(-1)
            target_raw = batch["FVC_raw"].to(device)  # Raw true FVC

            mu_scaled, sigma_scaled = model(img, tab)

            # Compute Loss on Scaled values
            loss = metric_aligned_loss(mu_scaled, sigma_scaled, target_scaled)
            loss_meter.update(loss.item(), img.size(0))

            # Inverse Transform for Metric Calculation
            mu_raw = mu_scaled * fvc_std + fvc_mean
            sigma_raw = sigma_scaled * fvc_std

            # Compute Metric
            score = score_function(target_raw, mu_raw, sigma_raw)
            metric_meter.update(score, img.size(0))

    return loss_meter.avg, metric_meter.avg


# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------


class ModelTrainer:
    def __init__(self, train_loader, val_loader, stats):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.stats = stats
        self.device = torch.device(Config.DEVICE)

        self.model = MCDSRNet().to(self.device)

        # Differential Learning Rates
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": Config.LR_BACKBONE},
                {"params": head_params, "lr": Config.LR_HEAD},
            ],
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS
        )

    def train(self):
        best_metric = -float("inf")
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

        print(f"Starting training on {self.device} for {Config.EPOCHS} epochs...")

        for epoch in range(Config.EPOCHS):
            train_loss = train_one_epoch(
                self.model, self.train_loader, self.optimizer, self.device
            )
            val_loss, val_metric = validate(
                self.model, self.val_loader, self.device, self.stats
            )

            self.scheduler.step()

            # Checkpoint
            if val_metric > best_metric:
                best_metric = val_metric
                torch.save(self.model.state_dict(), best_model_path)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Metric: {val_metric:.10f}"
            )

        print(f"Training complete. Best Metric: {best_metric:.10f}")
        return best_model_path

    def predict(self, test_loader, model_path=None):
        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        results = []
        fvc_mean = self.stats["FVC_mean"]
        fvc_std = self.stats["FVC_std"]

        with torch.no_grad():
            for batch in test_loader:
                img = batch["image"].to(self.device)
                tab = batch["tabular"].to(self.device)
                patient_weeks = batch["patient_week"]

                mu_scaled, sigma_scaled = self.model(img, tab)

                # Inverse Transform
                mu_raw = (mu_scaled * fvc_std + fvc_mean).cpu().numpy()
                sigma_raw = (sigma_scaled * fvc_std).cpu().numpy()

                # Post-processing: Clip sigma strictly for submission
                sigma_raw = np.maximum(sigma_raw, Config.MIN_UNCERTAINTY)

                for pw, fvc, conf in zip(patient_weeks, mu_raw, sigma_raw):
                    results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

        return results
