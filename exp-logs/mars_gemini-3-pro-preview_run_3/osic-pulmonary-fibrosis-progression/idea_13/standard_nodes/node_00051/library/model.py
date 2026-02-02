import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, MetricMonitor, TargetScaler
from library.data import get_dataloaders, get_submission_loader

# --------------------------------------------------------------------------
# Model Architecture
# --------------------------------------------------------------------------


class ImageBranch(nn.Module):
    """
    Fine-Tuned Content-Adaptive 2.5D Image Branch.
    Uses EfficientNet-B2 with top blocks unfrozen.
    """

    def __init__(self):
        super().__init__()
        # Load Pretrained Backbone
        # num_classes=0 returns the global pool features (flat vector)
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=Config.PRETRAINED, num_classes=0
        )

        # 1. Freeze entire backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze top two convolutional blocks (Stages 5 and 6 for EffNet-B2)
        # EfficientNet in timm stores blocks in .blocks (nn.Sequential)
        # We unfreeze the last 2 stages of blocks
        if hasattr(self.backbone, "blocks"):
            num_stages = len(self.backbone.blocks)
            for i in range(num_stages - 2, num_stages):
                for param in self.backbone.blocks[i].parameters():
                    param.requires_grad = True

        # 3. Unfreeze Head and BN if present (standard in timm effnet)
        if hasattr(self.backbone, "conv_head"):
            for param in self.backbone.conv_head.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, "bn2"):
            for param in self.backbone.bn2.parameters():
                param.requires_grad = True

        # Projection Layer: 1408 (B2) -> 128
        self.projection = nn.Linear(
            self.backbone.num_features, Config.IMAGE_EMBEDDING_DIM
        )

    def forward(self, x):
        # x shape: (Batch, 3, H, W)
        features = self.backbone(x)  # (Batch, 1408)
        embedding = self.projection(features)  # (Batch, 128)
        return embedding


class DSPRNet(nn.Module):
    """
    Dual-Stream Point-Wise Residual Network.
    Combines a deep interaction stream with a linear residual stream.
    """

    def __init__(self):
        super().__init__()
        self.img_branch = ImageBranch()

        # Dimensions
        # Tabular: [Age_Norm, Sex_Code, Smoking_Code, Baseline_FVC_Norm] -> 4 features
        # Time: t_rel -> 1 feature
        n_tab = len(Config.TABULAR_FEATURES)  # 4
        n_time = 1
        n_img = Config.IMAGE_EMBEDDING_DIM  # 256

        # Increased capacity as per Lesson 27 (Cite solution_lesson_node_00027)
        self.fusion_dim = 512

        # Stream A: Deep Interaction Stream
        # Input: [Image(256), Tabular(4), Time(1)]
        self.stream_a = nn.Sequential(
            nn.Linear(n_img + n_tab + n_time, self.fusion_dim),
            nn.ReLU(),
            nn.Linear(self.fusion_dim, self.fusion_dim),
        )

        # Stream B: Linear Residual Stream
        # Input: [Baseline_FVC_Norm(1), Time(1)]
        # This learns the linear decay and autoregressive coef directly.
        self.stream_b = nn.Linear(2, self.fusion_dim, bias=True)

        # Final Head
        # Projects summed stream outputs to [Mu, Raw_Sigma]
        self.head = nn.Linear(self.fusion_dim, 2)

    def forward(self, img, tab, t_rel):
        """
        Args:
            img: (B, 3, H, W)
            tab: (B, 4) -> [Age, Sex, Smoke, BaseFVC]
            t_rel: (B, 1)
        """
        # 1. Image Embedding
        img_emb = self.img_branch(img)  # (B, 128)

        # 2. Stream A (Deep)
        # Concat all inputs
        stream_a_in = torch.cat([img_emb, tab, t_rel], dim=1)
        out_a = self.stream_a(stream_a_in)  # (B, 64)

        # 3. Stream B (Linear Residual)
        # Extract Baseline_FVC_Norm (Index 3 in tabular)
        base_fvc = tab[:, 3:4]
        stream_b_in = torch.cat([base_fvc, t_rel], dim=1)
        out_b = self.stream_b(stream_b_in)  # (B, 64)

        # 4. Fusion (Summation)
        fused = out_a + out_b

        # 5. Prediction
        logits = self.head(fused)  # (B, 2)

        mu = logits[:, 0]
        raw_sigma = logits[:, 1]

        # Enforce positivity for sigma using softplus + epsilon
        sigma = F.softplus(raw_sigma) + 1e-6

        return mu, sigma


# --------------------------------------------------------------------------
# Training Utilities
# --------------------------------------------------------------------------


def laplace_nll_loss(true, pred_mu, pred_sigma):
    """
    Negative Log Likelihood for Laplace Distribution.
    L = log(sqrt(2) * sigma) + (sqrt(2) * |true - mu|) / sigma
    Computed in the scaled space.
    """
    # Clamp sigma for numerical stability in loss
    sigma = torch.clamp(pred_sigma, min=1e-6)

    delta = torch.abs(true - pred_mu)
    sqrt_2 = np.sqrt(2)

    loss = torch.log(sqrt_2 * sigma) + (sqrt_2 * delta) / sigma
    return torch.mean(loss)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        img = batch["image"].to(device)
        tab = batch["tabular"].to(device)
        t_rel = batch["t_rel"].to(device)
        target = batch["target"].to(device).squeeze(-1)  # (B,)

        optimizer.zero_grad()

        pred_mu, pred_sigma = model(img, tab, t_rel)

        loss = laplace_nll_loss(target, pred_mu, pred_sigma)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, device, target_scaler):
    model.eval()
    monitor = MetricMonitor()

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            tab = batch["tabular"].to(device)
            t_rel = batch["t_rel"].to(device)
            target_scaled = batch["target"].to(device).squeeze(-1)

            pred_mu_scaled, pred_sigma_scaled = model(img, tab, t_rel)

            # Inverse transform for metric calculation (ml units)
            pred_mu, pred_sigma = target_scaler.inverse_transform(
                pred_mu_scaled, pred_sigma_scaled
            )
            true_fvc = target_scaler.inverse_transform(target_scaled)

            monitor.update(true_fvc, pred_mu, pred_sigma)

    return monitor.avg


# --------------------------------------------------------------------------
# Main Execution Functions
# --------------------------------------------------------------------------


def train_model():
    """
    Main training loop.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Data
    train_loader, val_loader, target_scaler, _ = get_dataloaders(load_cached_data=True)

    # Model
    model = DSPRNet().to(device)

    # Differential Learning Rates
    # Backbone params get LR_BACKBONE, Heads get LR_HEAD
    backbone_params = list(model.img_branch.backbone.parameters())
    head_params = (
        list(model.img_branch.projection.parameters())
        + list(model.stream_a.parameters())
        + list(model.stream_b.parameters())
        + list(model.head.parameters())
    )

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

    best_score = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_score = validate(model, val_loader, device, target_scaler)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Score: {val_score:.6f}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            # print(f"  New best score! Saved model.")

    print(f"Training complete. Best Val Score: {best_score:.6f}")
    return best_model_path


def generate_submission():
    """
    Loads the best model and generates the submission file.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # We need the preprocessor and scaler fitted on train data
    # Calling get_dataloaders is the most robust way to get them state-matched
    _, _, target_scaler, tab_preprocessor = get_dataloaders(load_cached_data=True)

    # Submission Loader
    sub_loader = get_submission_loader(tab_preprocessor, load_cached_data=True)

    # Load Model
    model = DSPRNet().to(device)
    model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_path}. Run training first."
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []

    with torch.no_grad():
        for batch in sub_loader:
            img = batch["image"].to(device)
            tab = batch["tabular"].to(device)
            t_rel = batch["t_rel"].to(device)
            patient_weeks = batch["patient_week"]

            pred_mu_scaled, pred_sigma_scaled = model(img, tab, t_rel)

            # Inverse Transform
            pred_mu, pred_sigma = target_scaler.inverse_transform(
                pred_mu_scaled, pred_sigma_scaled
            )

            # Apply final clipping for submission
            pred_sigma = torch.clamp(pred_sigma, min=Config.CONFIDENCE_CLIP)

            pred_mu = pred_mu.cpu().numpy()
            pred_sigma = pred_sigma.cpu().numpy()

            for pw, fvc, conf in zip(patient_weeks, pred_mu, pred_sigma):
                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

    # Save
    sub_df = pd.DataFrame(results)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
