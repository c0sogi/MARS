import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import timm
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import (
    seed_everything,
    AverageMeter,
    laplace_log_likelihood_loss,
    score_metric,
)
from library.data import get_dataloaders, prepare_submission_df, LungDataset


class MultiScaleEfficientNet(nn.Module):
    """
    Extracts features from both intermediate (texture) and final (semantic) layers
    of an EfficientNet-B0 backbone.
    """

    def __init__(self):
        super().__init__()
        # Load EfficientNet-B0, extracting features from intermediate and final layers
        # out_indices=(3, 4) corresponds to stride 16 (approx block 11) and stride 32 (final)
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=Config.PRETRAINED,
            features_only=True,
            out_indices=(3, 4),
            in_chans=Config.NUM_SLICES,
        )

        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Determine feature channels dynamically
        # Create a dummy input to trace shapes
        dummy = torch.randn(1, Config.NUM_SLICES, Config.IMG_SIZE, Config.IMG_SIZE)
        with torch.no_grad():
            feats = self.backbone(dummy)
            # feats[0] is texture (stride 16), feats[1] is semantic (stride 32)
            dim_texture = feats[0].shape[1]
            dim_semantic = feats[1].shape[1]

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Projections to compact dimension
        self.proj_texture = nn.Linear(dim_texture, Config.PROJECTION_DIM)
        self.proj_semantic = nn.Linear(dim_semantic, Config.PROJECTION_DIM)

    def forward(self, x):
        # x: (B, 3, 224, 224)
        feats = self.backbone(x)

        # Texture features
        t = self.pool(feats[0]).flatten(1)
        t = self.proj_texture(t)

        # Semantic features
        s = self.pool(feats[1]).flatten(1)
        s = self.proj_semantic(s)

        # Concatenate: (B, 64+64=128)
        return torch.cat([t, s], dim=1)


class FusedNet(nn.Module):
    """
    Fused Multimodal Network.
    Concatenates all features (Image + Metadata + Temporal) into a single MLP.
    Cite solution_lesson_node_00022: Avoid isolating time variables in additive branches.
    """

    def __init__(self):
        super().__init__()

        # --- Image Branch ---
        self.image_encoder = MultiScaleEfficientNet()
        self.img_vec_dim = Config.PROJECTION_DIM * 2

        # --- Tabular Embeddings ---
        self.sex_embed = nn.Embedding(2, Config.EMBED_DIMS["Sex"])
        self.smoke_embed = nn.Embedding(3, Config.EMBED_DIMS["SmokingStatus"])

        # Input Dimension
        # Image (128) + Sex (2) + Smoke (3) + Age (1) + BaseFVC (1) + Weeks (1)
        self.input_dim = (
            self.img_vec_dim
            + Config.EMBED_DIMS["Sex"]
            + Config.EMBED_DIMS["SmokingStatus"]
            + 1  # Age
            + 1  # BaseFVC
            + 1  # Weeks
        )

        # Latent Dimension
        self.latent_dim = 128

        # MLP (No BatchNorm as per solution_lesson_node_00010)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
        )

        # --- Heads ---
        self.head_mu = nn.Linear(self.latent_dim, 1)
        self.head_sigma = nn.Linear(self.latent_dim, 1)

    def forward(self, img, weeks, base_fvc, age, sex, smoke):
        # --- Feature Extraction ---
        img_vec = self.image_encoder(img)  # (B, 128)
        sex_vec = self.sex_embed(sex)  # (B, 2)
        smoke_vec = self.smoke_embed(smoke)  # (B, 3)

        # --- Concatenation ---
        # Concatenate all features early to allow interaction learning
        x = torch.cat([img_vec, sex_vec, smoke_vec, age, base_fvc, weeks], dim=1)

        # --- MLP ---
        feat = self.mlp(x)

        # --- Output ---
        mu = self.head_mu(feat)
        sigma_raw = self.head_sigma(feat)

        # Sigma must be positive. Softplus + small epsilon for stability.
        sigma = F.softplus(sigma_raw) + 1e-3

        return mu, sigma


def train_epoch(model, loader, optimizer, device):
    model.train()
    losses = AverageMeter()

    for batch in loader:
        # Move inputs to device
        img = batch["image"].to(device)
        weeks = batch["weeks"].to(device)
        base_fvc = batch["baseline_fvc"].to(device)
        age = batch["age"].to(device)
        sex = batch["sex"].to(device)
        smoke = batch["smoke"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()

        # Forward pass
        mu, sigma = model(img, weeks, base_fvc, age, sex, smoke)

        # Calculate loss (Modified Laplace Log Likelihood)
        loss = laplace_log_likelihood_loss(target, mu, sigma)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), img.size(0))

    return losses.avg


def validate(model, loader, device):
    model.eval()
    scores = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            weeks = batch["weeks"].to(device)
            base_fvc = batch["baseline_fvc"].to(device)
            age = batch["age"].to(device)
            sex = batch["sex"].to(device)
            smoke = batch["smoke"].to(device)
            target_z = batch["target"].to(device)

            mu_z, sigma_z = model(img, weeks, base_fvc, age, sex, smoke)

            # Inverse Transform to original scale (ml) for metric calculation
            mu_ml = mu_z * Config.FVC_STD + Config.FVC_MEAN
            sigma_ml = sigma_z * Config.FVC_STD
            target_ml = target_z * Config.FVC_STD + Config.FVC_MEAN

            # Move to CPU numpy for scoring
            mu_np = mu_ml.cpu().numpy().flatten()
            sigma_np = sigma_ml.cpu().numpy().flatten()
            target_np = target_ml.cpu().numpy().flatten()

            metric = score_metric(target_np, mu_np, sigma_np)
            scores.update(metric, img.size(0))

    return scores.avg


def run_experiment():
    seed_everything(Config.SEED)

    print(f"Starting Experiment: {Config.PROJECT_NAME}")
    print(f"Device: {Config.DEVICE}")

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_META_PATH)
    val_df = pd.read_csv(Config.VAL_META_PATH)

    if Config.DEBUG:
        print(f"Debug Mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 2. DataLoaders
    train_loader, val_loader = get_dataloaders(train_df, val_df)

    # 3. Model Setup
    model = FusedNet().to(Config.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # 4. Training Loop
    best_score = -float("inf")
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, Config.DEVICE)
        val_score = validate(model, val_loader, Config.DEVICE)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Score: {val_score:.5f}"
        )

        # Save best model
        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best Score! Model Saved.")

    print(f"Training Complete. Best Validation Score: {best_score:.5f}")

    # 5. Inference / Submission
    print("Generating Submission...")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=Config.DEVICE))
    model.eval()

    # Prepare submission data
    test_df = pd.read_csv(Config.TEST_META_PATH)
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Expand test set to required Patient_Week rows
    sub_df = prepare_submission_df(test_df, sample_sub)

    # Create Dataset/Loader for submission
    sub_ds = LungDataset(sub_df, mode="submission")
    sub_loader = DataLoader(
        sub_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        drop_last=False,
    )

    results = []

    with torch.no_grad():
        for batch in sub_loader:
            img = batch["image"].to(Config.DEVICE)
            weeks = batch["weeks"].to(Config.DEVICE)
            base_fvc = batch["baseline_fvc"].to(Config.DEVICE)
            age = batch["age"].to(Config.DEVICE)
            sex = batch["sex"].to(Config.DEVICE)
            smoke = batch["smoke"].to(Config.DEVICE)
            patient_ids = batch["patient_id"]  # List of patient IDs

            # Predict
            mu_z, sigma_z = model(img, weeks, base_fvc, age, sex, smoke)

            # Inverse Transform
            mu_ml = mu_z * Config.FVC_STD + Config.FVC_MEAN
            sigma_ml = sigma_z * Config.FVC_STD

            # Post-process Sigma (Metric requirement: clipped at 70)
            # Note: We apply this clip for the final CSV, though the metric function does it internally too.
            sigma_ml = torch.clamp(sigma_ml, min=Config.MIN_CONFIDENCE)

            mu_np = mu_ml.cpu().numpy().flatten()
            sigma_np = sigma_ml.cpu().numpy().flatten()

            # Reconstruct Patient_Week ID
            # In submission mode, LungDataset returns 'weeks' as relative weeks scaled by 100.
            # We need to map back to the Patient_Week string.
            # However, the DataLoader batches might shuffle if we didn't set shuffle=False.
            # A safer way is to rely on the order since shuffle=False.
            # But we can also reconstruct if needed.
            # Actually, we can just zip with the sub_df slice corresponding to this batch index?
            # Easier: Just collect results and assign to sub_df later since order is preserved.

            for m, s in zip(mu_np, sigma_np):
                results.append((m, s))

    # Assign results back to dataframe
    sub_df["FVC"] = [r[0] for r in results]
    sub_df["Confidence"] = [r[1] for r in results]

    # Format for submission
    submission = sub_df[["Patient_Week", "FVC", "Confidence"]]
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
