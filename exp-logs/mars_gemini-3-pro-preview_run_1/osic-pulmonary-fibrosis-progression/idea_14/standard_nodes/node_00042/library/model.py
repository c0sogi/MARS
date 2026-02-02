import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import timm

# Import from provided libraries
from library.utils import seed_everything, AverageMeter, laplace_log_likelihood_metric
from library.data import LungDataset, LungDataProcessor, get_transforms


class HighFidelityDualNet(nn.Module):
    """
    High-Fidelity Holistic Dual-Axis Network.

    Features:
    - Two independent EfficientNet-B0 backbones for Axial and Coronal views.
    - No dimensionality reduction on visual features (maintains 1280-dim).
    - Up-projected tabular embedding (matches 1280-dim).
    - Symmetric Self-Attention fusion.
    - Holistic Mean Pooling readout.
    - Prior-Preserving Skip Connection.
    - Parametric regression head (Alpha, Sigma_Base, Sigma_Growth).
    """

    def __init__(self, tab_input_dim=6, embed_dim=1280):
        super(HighFidelityDualNet, self).__init__()

        # 1. Independent Visual Backbones
        # num_classes=0 returns the global pool output (1280 for B0)
        self.backbone_ax = timm.create_model(
            "tf_efficientnet_b0_ns", pretrained=True, num_classes=0
        )
        self.backbone_cor = timm.create_model(
            "tf_efficientnet_b0_ns", pretrained=True, num_classes=0
        )

        # 2. Up-Projected Tabular Embedding
        self.tab_mlp = nn.Sequential(
            nn.Linear(tab_input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 512),
            nn.GELU(),
            nn.Linear(512, embed_dim),
        )

        # 3. Symmetric Attention
        # batch_first=True expects (Batch, Seq, Feature)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=8, batch_first=True
        )

        # 4. Regression Head
        # Input: Pooled (1280) + Tabular Raw (6) + Baseline FVC Scaled (1) = 1287
        head_input_dim = embed_dim + tab_input_dim + 1

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 3),  # alpha, sigma_base, sigma_growth
        )

    def forward(
        self, img_ax, img_cor, tab_vec, rel_week, baseline_fvc, baseline_fvc_sc
    ):
        """
        Args:
            img_ax: (B, 3, 224, 224)
            img_cor: (B, 3, 224, 224)
            tab_vec: (B, 6)
            rel_week: (B, 1) or (B,)
            baseline_fvc: (B,) - Raw baseline FVC for calculation
            baseline_fvc_sc: (B,) - Scaled baseline FVC for skip connection
        """
        # Ensure shapes
        if rel_week.dim() == 1:
            rel_week = rel_week.unsqueeze(1)
        if baseline_fvc.dim() == 1:
            baseline_fvc = baseline_fvc.unsqueeze(1)
        if baseline_fvc_sc.dim() == 1:
            baseline_fvc_sc = baseline_fvc_sc.unsqueeze(1)

        # 1. Visual Feature Extraction
        # Output: (B, 1280)
        feat_ax = self.backbone_ax(img_ax)
        feat_cor = self.backbone_cor(img_cor)

        # 2. Tabular Embedding
        # Output: (B, 1280)
        feat_tab = self.tab_mlp(tab_vec)

        # 3. Sequence Construction & Fusion
        # Stack: (B, 3, 1280)
        seq = torch.stack([feat_ax, feat_cor, feat_tab], dim=1)

        # Self-Attention
        attn_out, _ = self.attention(seq, seq, seq)

        # Holistic Readout (Mean Pooling)
        # Output: (B, 1280)
        pooled = torch.mean(attn_out, dim=1)

        # 4. Skip Connection & Head
        # Concatenate: (B, 1280 + 6 + 1)
        combined = torch.cat([pooled, tab_vec, baseline_fvc_sc], dim=1)

        # Predict parameters
        out = self.head(combined)

        alpha = out[:, 0:1]
        sigma_base = F.softplus(out[:, 1:2])
        sigma_growth = F.softplus(out[:, 2:3])

        # 5. Calculate Final Predictions
        # FVC = Base + Alpha * Week
        fvc_pred = baseline_fvc + alpha * rel_week

        # Confidence = Sigma_Base + Sigma_Growth * |Week|
        confidence = sigma_base + sigma_growth * torch.abs(rel_week)

        return fvc_pred.squeeze(1), confidence.squeeze(1)


def train_model(train_df, val_df, config):
    """
    Handles the training loop, validation, and early stopping.
    """
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Setup
    processor = LungDataProcessor(cache_dir="./working/idea_14")
    train_transforms = get_transforms("train")
    val_transforms = get_transforms("val")

    train_dataset = LungDataset(
        train_df, processor, transforms=train_transforms, mode="train"
    )
    val_dataset = LungDataset(val_df, processor, transforms=val_transforms, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model Setup
    model = HighFidelityDualNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler()

    # Tracking
    best_metric = -float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0

    print(f"Starting training on {device} for {config['epochs']} epochs...")

    for epoch in range(config["epochs"]):
        # --- Training ---
        model.train()
        train_loss_meter = AverageMeter()

        for batch in train_loader:
            # Move to device
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tab_vec = batch["tab_vec"].to(device)
            rel_week = batch["rel_week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            baseline_fvc_sc = batch["baseline_fvc_sc"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                pred_fvc, pred_conf = model(
                    img_ax, img_cor, tab_vec, rel_week, baseline_fvc, baseline_fvc_sc
                )
                # Metric is negative, higher is better. Loss is -Metric.
                metric_val = laplace_log_likelihood_metric(target, pred_fvc, pred_conf)
                loss = -metric_val

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_meter.update(loss.item(), img_ax.size(0))

        scheduler.step()

        # --- Validation ---
        model.eval()
        val_metric_meter = AverageMeter()

        with torch.no_grad():
            for batch in val_loader:
                img_ax = batch["img_ax"].to(device)
                img_cor = batch["img_cor"].to(device)
                tab_vec = batch["tab_vec"].to(device)
                rel_week = batch["rel_week"].to(device)
                baseline_fvc = batch["baseline_fvc"].to(device)
                baseline_fvc_sc = batch["baseline_fvc_sc"].to(device)
                target = batch["target"].to(device)

                pred_fvc, pred_conf = model(
                    img_ax, img_cor, tab_vec, rel_week, baseline_fvc, baseline_fvc_sc
                )
                metric_val = laplace_log_likelihood_metric(target, pred_fvc, pred_conf)

                val_metric_meter.update(metric_val.item(), img_ax.size(0))

        # --- Logging & Checkpointing ---
        current_metric = val_metric_meter.avg
        print(
            f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {train_loss_meter.avg:.4f} | Val Metric: {current_metric:.6f}"
        )

        if current_metric > best_metric:
            best_metric = current_metric
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            # Save best model immediately
            torch.save(model.state_dict(), "./working/best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= config["patience"]:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Validation Metric: {best_metric:.6f}")

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model


def predict(test_df, model_path="./working/best_model.pth"):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Setup
    processor = LungDataProcessor(cache_dir="./working/idea_14")
    test_transforms = get_transforms("test")  # No augmentation

    test_dataset = LungDataset(
        test_df, processor, transforms=test_transforms, mode="test"
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    # Model Setup
    model = HighFidelityDualNet().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: Model path {model_path} not found. Using random weights.")

    model.eval()

    results = []

    with torch.no_grad():
        for batch in test_loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            tab_vec = batch["tab_vec"].to(device)
            rel_week = batch["rel_week"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            baseline_fvc_sc = batch["baseline_fvc_sc"].to(device)
            patient_weeks = batch["patient_week"]  # List of strings

            pred_fvc, pred_conf = model(
                img_ax, img_cor, tab_vec, rel_week, baseline_fvc, baseline_fvc_sc
            )

            # Clip confidence as per metric requirement (though metric func does it, we should output valid conf)
            # The metric function clips at 70, so we should ensure our output is reasonable.
            # The model outputs softplus, so it's > 0.

            pred_fvc = pred_fvc.cpu().numpy().flatten()
            pred_conf = pred_conf.cpu().numpy().flatten()

            for pw, fvc, conf in zip(patient_weeks, pred_fvc, pred_conf):
                results.append({"Patient_Week": pw, "FVC": fvc, "Confidence": conf})

    # Save Submission
    sub_df = pd.DataFrame(results)
    os.makedirs("./submission", exist_ok=True)
    sub_path = "./submission/submission.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path} with {len(sub_df)} rows.")
    return sub_df
