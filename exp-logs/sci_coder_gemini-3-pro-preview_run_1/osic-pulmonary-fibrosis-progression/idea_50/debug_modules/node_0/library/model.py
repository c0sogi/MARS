import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

# Import from provided libraries
from library.utils import seed_everything, LaplaceLogLikelihoodLoss, AverageMeter
from library.data import LungDataset, get_transforms


class NSHDAN(nn.Module):
    """
    Normalized Shared-Latent Holistic Dual-Axis Network (NSH-DAN).
    """

    def __init__(self, latent_dim=128, visual_dim=1280, dropout=0.1):
        super(NSHDAN, self).__init__()

        # 1. Independent Low-Capacity Visual Backbones (EfficientNet-B0)
        # num_classes=0 ensures we get the pooled feature vector (1280-dim)
        self.backbone_ax = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0
        )
        self.backbone_cor = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0
        )

        # 2. Shared-Latent Tabular Encoder
        # Input: Age, Sex, Smoking, Percent (4 dims)
        self.tab_encoder = nn.Sequential(
            nn.Linear(4, 64), nn.GELU(), nn.Linear(64, latent_dim), nn.GELU()
        )

        # 3. Normalized Bifurcated Flow
        # Projection for alignment with visual features
        self.proj_align = nn.Linear(latent_dim, visual_dim)
        # Layer Normalization immediately after projection
        self.norm_align = nn.LayerNorm(visual_dim)

        # 4. Pre-Norm Symmetric Attention (Contextualization Phase)
        # norm_first=True implements Pre-Normalization
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=visual_dim,
            nhead=4,
            dim_feedforward=2048,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 5. Bottleneck Prior-Anchored Head
        # Concatenates Holistic Fused Vector (1280) + Shared Latent Vector (128)
        self.head = nn.Sequential(
            nn.Linear(visual_dim + latent_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 3),  # alpha, sigma_base, sigma_growth
        )

    def forward(self, img_ax, img_cor, meta):
        # Extract visual features
        v_ax = self.backbone_ax(img_ax)  # (B, 1280)
        v_cor = self.backbone_cor(img_cor)  # (B, 1280)

        # Encode metadata to latent
        t_lat = self.tab_encoder(meta)  # (B, 128)

        # Bifurcated Flow A: Alignment
        t_align = self.proj_align(t_lat)
        t_align = self.norm_align(t_align)  # (B, 1280)

        # Stack tokens: [Axial, Coronal, Aligned_Tabular]
        seq = torch.stack([v_ax, v_cor, t_align], dim=1)  # (B, 3, 1280)

        # Attention Fusion
        seq_out = self.fusion(seq)  # (B, 3, 1280)

        # Holistic Readout: Global Average Pooling across sequence
        h_fused = seq_out.mean(dim=1)  # (B, 1280)

        # Bifurcated Flow B: Prior Preservation (Concatenation)
        combined = torch.cat([h_fused, t_lat], dim=1)  # (B, 1408)

        # Prediction Head
        out = self.head(combined)

        # Unpack outputs
        alpha = out[:, 0]
        # Enforce positivity for sigmas using Softplus
        sigma_base = nn.functional.softplus(out[:, 1])
        sigma_growth = nn.functional.softplus(out[:, 2])

        return alpha, sigma_base, sigma_growth


def get_baseline_weeks(mode="train"):
    """
    Creates a dictionary mapping Patient ID to their Baseline Week.
    This is necessary because the LungDataset provides the current week and
    baseline FVC, but we need the baseline week to calculate delta_t accurately.
    """
    if mode == "train":
        df = pd.read_csv("./metadata/train.csv")
    elif mode == "val":
        df = pd.read_csv("./metadata/val.csv")
    elif mode == "test":
        df = pd.read_csv("./metadata/test.csv")
        # For test set, metadata already has Baseline_Week column
        return dict(zip(df["Patient"], df["Baseline_Week"]))
    else:
        return {}

    # For train/val, find the week corresponding to the first visit (baseline)
    baseline_map = {}
    for patient, group in df.groupby("Patient"):
        # Sort by weeks to find the earliest one
        sorted_group = group.sort_values("Weeks")
        base_week = sorted_group.iloc[0]["Weeks"]
        baseline_map[patient] = base_week

    return baseline_map


def train_one_epoch(model, loader, criterion, optimizer, device, baseline_map):
    model.train()
    losses = AverageMeter()

    for batch in tqdm(loader, desc="Training", leave=False):
        img_ax = batch["image_axial"].to(device)
        img_cor = batch["image_coronal"].to(device)
        meta = batch["meta"].to(device)
        target = batch["target"].to(device)
        week = batch["week"].to(device)
        base_fvc = batch["base_fvc"].to(device)
        patient_weeks = batch["patient_week"]

        # Calculate delta_t
        # Parse patient IDs to look up baseline week
        # Note: Doing this in the loop is slightly inefficient but safe
        # batch['patient_week'] format: "ID..._Week" or just ID for train?
        # LungDataset returns "Patient_Week" from csv or constructs it.
        # For train/val, we can extract Patient ID from the string if needed,
        # but LungDataset doesn't explicitly return Patient ID separate from Patient_Week in the dict.
        # However, we can infer delta_t if we assume the loader logic.
        # Let's extract patient ID from patient_week string (format: ID_Week)

        base_weeks = []
        for pw in patient_weeks:
            # patient_week is ID_Week. Split on last underscore.
            pid = pw.rsplit("_", 1)[0]
            base_weeks.append(baseline_map.get(pid, 0))

        base_weeks = torch.tensor(base_weeks, device=device, dtype=torch.float32)
        dt = week - base_weeks

        optimizer.zero_grad()

        alpha, sigma_base, sigma_growth = model(img_ax, img_cor, meta)

        # Parametric Prediction
        pred_fvc = base_fvc + alpha * dt
        pred_sigma = sigma_base + sigma_growth * torch.abs(dt)

        loss = criterion(pred_fvc, pred_sigma, target)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), img_ax.size(0))

    return losses.avg


def validate(model, loader, criterion, device, baseline_map):
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            meta = batch["meta"].to(device)
            target = batch["target"].to(device)
            week = batch["week"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            patient_weeks = batch["patient_week"]

            base_weeks = []
            for pw in patient_weeks:
                pid = pw.rsplit("_", 1)[0]
                base_weeks.append(baseline_map.get(pid, 0))

            base_weeks = torch.tensor(base_weeks, device=device, dtype=torch.float32)
            dt = week - base_weeks

            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, meta)

            pred_fvc = base_fvc + alpha * dt
            pred_sigma = sigma_base + sigma_growth * torch.abs(dt)

            loss = criterion(pred_fvc, pred_sigma, target)
            losses.update(loss.item(), img_ax.size(0))

    return losses.avg


def train_model(epochs=30, batch_size=32, lr=1e-4, patience=8, limit_size=None):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Loaders
    train_dataset = LungDataset(
        mode="train", transform=get_transforms("train"), limit_size=limit_size
    )
    val_dataset = LungDataset(
        mode="val", transform=get_transforms("val"), limit_size=limit_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Baseline Maps for Delta T calculation
    train_base_map = get_baseline_weeks("train")
    val_base_map = get_baseline_weeks("val")
    # Merge maps for convenience
    full_base_map = {**train_base_map, **val_base_map}

    # Model Setup
    model = NSHDAN().to(device)
    criterion = LaplaceLogLikelihoodLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    best_loss = float("inf")
    patience_counter = 0
    save_path = "./working/best_model.pth"

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, full_base_map
        )
        val_loss = validate(model, val_loader, criterion, device, full_base_map)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        )

        # Checkpointing & Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  -> Model saved! Best Val Loss: {best_loss:.6f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    return best_loss


def generate_submission(batch_size=32):
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    model = NSHDAN().to(device)
    model_path = "./working/best_model.pth"
    if not os.path.exists(model_path):
        print("Model file not found. Skipping submission generation.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Test Data
    test_dataset = LungDataset(mode="test", transform=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Baseline Map
    test_base_map = get_baseline_weeks("test")

    predictions = []

    print("Generating submission...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            img_ax = batch["image_axial"].to(device)
            img_cor = batch["image_coronal"].to(device)
            meta = batch["meta"].to(device)
            week = batch["week"].to(device)  # Predict_Week
            base_fvc = batch["base_fvc"].to(device)
            patient_weeks = batch["patient_week"]

            # Calculate delta_t
            base_weeks = []
            for pw in patient_weeks:
                pid = pw.rsplit("_", 1)[0]
                base_weeks.append(test_base_map.get(pid, 0))

            base_weeks = torch.tensor(base_weeks, device=device, dtype=torch.float32)
            dt = week - base_weeks

            alpha, sigma_base, sigma_growth = model(img_ax, img_cor, meta)

            pred_fvc = base_fvc + alpha * dt
            pred_sigma = sigma_base + sigma_growth * torch.abs(dt)

            # Clip confidence as per metric requirement
            pred_sigma = torch.clamp(pred_sigma, min=70)

            # Collect results
            for i in range(len(patient_weeks)):
                predictions.append(
                    {
                        "Patient_Week": patient_weeks[i],
                        "FVC": pred_fvc[i].item(),
                        "Confidence": pred_sigma[i].item(),
                    }
                )

    # Save to CSV
    sub_df = pd.DataFrame(predictions)
    os.makedirs("./submission", exist_ok=True)
    sub_df.to_csv("./submission/submission.csv", index=False)
    print(f"Submission saved to ./submission/submission.csv with {len(sub_df)} rows.")
