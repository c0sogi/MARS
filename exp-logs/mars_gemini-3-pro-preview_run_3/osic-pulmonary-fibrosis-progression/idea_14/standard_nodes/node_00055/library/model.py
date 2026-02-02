import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import AverageMeter, laplace_log_likelihood_metric, seed_everything
from library.data import get_dataloaders, get_test_dataloader


class DDSRNet(nn.Module):
    """
    Decoupled Dual-Stream Residual Network (DDSR-Net).

    Architecture:
    1. Image Backbone: EfficientNet-V2-S (Top 2 stages unfrozen).
    2. Stream A: Linear Residual (BaseFVC + Time -> Trend).
    3. Stream B: Deep Interaction (Image + Tabular -> Residual Correction & Uncertainty).
    4. Decoupled Heads: Separate MLPs for Mu and Sigma.
    """

    def __init__(self):
        super(DDSRNet, self).__init__()

        # --- 1. Backbone: EfficientNet-V2-S ---
        # Load pretrained model, removing the classification head (num_classes=0)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=Config.BACKBONE_PRETRAINED,
            num_classes=0,
            in_chans=Config.NUM_SLICES,
        )
        self.backbone_dim = self.backbone.num_features

        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze top two convolutional stages
        # In timm's EfficientNet implementation, stages are stored in .blocks
        # We unfreeze the last 2 blocks (stages) and the conv_head/bn2 if present
        if hasattr(self.backbone, "blocks"):
            for block in self.backbone.blocks[-2:]:
                for param in block.parameters():
                    param.requires_grad = True

        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True

        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # Projection for Image Features
        self.img_projector = nn.Linear(self.backbone_dim, Config.IMG_EMBED_DIM)

        # --- 2. Stream A: Linear Residual ---
        # Inputs: [BaseFVC_norm, RelWeek_scaled] -> 2 features
        # Models the autoregressive trend: BaseFVC + decay * t
        self.stream_a = nn.Linear(2, 1, bias=False)

        # --- 3. Stream B: Deep Interaction ---
        # Inputs: Image_Embed (128) + Tabular (5: BaseFVC, Age, Sex, Smoke, RelWeek)
        input_dim_b = Config.IMG_EMBED_DIM + 5
        self.stream_b_mlp = nn.Sequential(
            nn.Linear(input_dim_b, Config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(Config.HIDDEN_DIM, Config.HIDDEN_DIM),
            nn.ReLU(),
        )

        # --- 4. Decoupled Heads ---
        # Mu Head: Predicts residual correction to the linear trend
        self.head_mu = nn.Linear(Config.HIDDEN_DIM, 1)
        # Sigma Head: Predicts uncertainty
        self.head_sigma = nn.Linear(Config.HIDDEN_DIM, 1)

    def forward(self, image, tabular):
        # image: [B, 3, H, W]
        # tabular: [B, 5] -> [BaseFVC, Age, Sex, Smoke, RelWeek]

        # --- Image Processing ---
        img_feats = self.backbone(image)  # [B, 1280]
        img_emb = self.img_projector(img_feats)  # [B, 128]

        # --- Stream A (Linear Trend) ---
        # Select BaseFVC (idx 0) and RelWeek (idx 4)
        stream_a_in = tabular[:, [0, 4]]
        trend = self.stream_a(stream_a_in)  # [B, 1]

        # --- Stream B (Deep Interaction) ---
        # Concatenate Image Embedding and All Tabular Features
        stream_b_in = torch.cat([img_emb, tabular], dim=1)  # [B, 133]
        features_b = self.stream_b_mlp(stream_b_in)  # [B, 128]

        # --- Prediction Heads ---
        mu_resid = self.head_mu(features_b)  # [B, 1]
        sigma_raw = self.head_sigma(features_b)  # [B, 1]

        # --- Final Combination ---
        # Mu = Linear Trend + Deep Residual Correction
        mu = trend + mu_resid

        # Sigma = Softplus(raw) + epsilon (to ensure positivity)
        sigma = F.softplus(sigma_raw) + 1e-6

        return mu, sigma


def get_optimizer_and_scheduler(model):
    """
    Configures AdamW with differential learning rates:
    - Lower LR for the backbone.
    - Higher LR for the heads/MLP.
    """
    backbone_ids = list(map(id, model.backbone.parameters()))
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if id(param) in backbone_ids:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEADS},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    return optimizer, scheduler


def loss_fn(mu, sigma, target):
    """
    Modified Laplace Log Likelihood Loss for training.
    L = |y - mu| / sigma + ln(sigma)
    """
    # Squeeze to ensure shape matching [B]
    mu = mu.squeeze()
    sigma = sigma.squeeze()
    target = target.squeeze()

    delta = torch.abs(target - mu)
    loss = (delta / sigma) + torch.log(sigma)
    return torch.mean(loss)


def train_model(train_loader, val_loader):
    """
    Main training loop.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    model = DDSRNet().to(device)
    optimizer, scheduler = get_optimizer_and_scheduler(model)

    best_metric = -float("inf")

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # --- Training ---
        model.train()
        train_loss_meter = AverageMeter()

        for batch in train_loader:
            image = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            mu, sigma = model(image, tabular)

            loss = loss_fn(mu, sigma, target)
            loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item(), image.size(0))

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_metric_meter = AverageMeter()

        with torch.no_grad():
            for batch in val_loader:
                image = batch["image"].to(device)
                tabular = batch["tabular"].to(device)
                raw_fvc = batch["raw_fvc"].numpy()  # Ground truth in ml

                mu, sigma = model(image, tabular)

                # Inverse Transform (Un-scale)
                mu_np = mu.cpu().numpy().flatten()
                sigma_np = sigma.cpu().numpy().flatten()

                mu_unscaled = mu_np * Config.TARGET_STD + Config.TARGET_MEAN
                sigma_unscaled = sigma_np * Config.TARGET_STD

                # Calculate Metric
                score = laplace_log_likelihood_metric(
                    raw_fvc, mu_unscaled, sigma_unscaled
                )
                val_metric_meter.update(score, image.size(0))

        print(
            f"Epoch {epoch+1:02d}/{Config.EPOCHS} | Train Loss: {train_loss_meter.avg:.6f} | Val Metric: {val_metric_meter.avg:.10f}"
        )

        # Save Best Model
        if val_metric_meter.avg > best_metric:
            best_metric = val_metric_meter.avg
            torch.save(
                model.state_dict(),
                os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
            )

    print(f"Training complete. Best Validation Metric: {best_metric:.10f}")
    return model


def predict(test_loader):
    """
    Inference pipeline for generating submission.
    """
    device = torch.device(Config.DEVICE)
    model = DDSRNet().to(device)

    # Load Best Checkpoint
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("Loaded best model checkpoint.")
    else:
        print("Warning: No checkpoint found! Using random initialization.")

    model.eval()
    results = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            patient_weeks = batch["patient_week"]

            mu, sigma = model(image, tabular)

            # Inverse Transform
            mu_np = mu.cpu().numpy().flatten()
            sigma_np = sigma.cpu().numpy().flatten()

            mu_unscaled = mu_np * Config.TARGET_STD + Config.TARGET_MEAN
            sigma_unscaled = sigma_np * Config.TARGET_STD

            # Post-Processing: Hard clip sigma at 70ml
            sigma_final = np.maximum(sigma_unscaled, 70)

            for pw, fvc, conf in zip(patient_weeks, mu_unscaled, sigma_final):
                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

    # Save Submission
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    return submission_df


def run():
    """
    Entry point for the module logic.
    """
    Config.setup()
    train_loader, val_loader = get_dataloaders()
    train_model(train_loader, val_loader)

    test_loader = get_test_dataloader()
    predict(test_loader)
