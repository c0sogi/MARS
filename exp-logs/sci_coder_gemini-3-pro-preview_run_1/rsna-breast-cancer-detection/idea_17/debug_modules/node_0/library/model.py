import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
from tqdm.auto import tqdm

from library.config import Config
from library.utils import get_logger, probabilistic_f1, seed_everything
from library.dataset import SiameseBreastCancerDataset

# Initialize logger
logger = get_logger("model")


class PyramidSiameseEfficientNet(nn.Module):
    """
    Pyramid Symmetry-Difference Siamese Network using EfficientNet-B2 backbone.

    Extracts features at multiple scales (P3, P4, P5), computes differences
    between target and contralateral views to cancel out symmetric signals (like Age),
    and fuses them for classification.
    """

    def __init__(self, backbone_name=Config.BACKBONE, pretrained=True):
        super(PyramidSiameseEfficientNet, self).__init__()

        # Load backbone with feature extraction enabled
        # Indices (2, 3, 4) typically correspond to strides 8, 16, 32 (P3, P4, P5)
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4),
            in_chans=Config.CHANNELS,
        )

        # Determine feature channel dimensions dynamically
        # Create a dummy input to trace shapes
        dummy_input = torch.randn(1, Config.CHANNELS, 256, 256)
        with torch.no_grad():
            features = self.encoder(dummy_input)

        # Calculate total embedding dimension
        # For each scale, we have: GAP(Target) + GAP(Target - Contra)
        # So we take 2 * channels for each scale
        self.feature_channels = [f.shape[1] for f in features]
        total_dim = sum(self.feature_channels) * 2

        self.head = nn.Linear(total_dim, 1)

    def forward_features(self, x):
        """Passes one image through the backbone."""
        return self.encoder(x)

    def forward(self, x_target, x_contra):
        """
        Args:
            x_target: Tensor (B, C, H, W) - Candidate image
            x_contra: Tensor (B, C, H, W) - Contralateral image
        """
        # 1. Extract Multi-Scale Features
        feats_target = self.forward_features(x_target)  # List of tensors [P3, P4, P5]
        feats_contra = self.forward_features(x_contra)

        pooled_vectors = []

        # 2. Process each scale
        for ft, fc in zip(feats_target, feats_contra):
            # Signed Feature Difference
            # Captures asymmetry while cancelling symmetric background (Age/Density)
            diff = ft - fc

            # Global Average Pooling
            # (B, C, H, W) -> (B, C)
            ft_pool = ft.mean(dim=(2, 3))
            diff_pool = diff.mean(dim=(2, 3))

            pooled_vectors.append(ft_pool)
            pooled_vectors.append(diff_pool)

        # 3. Concatenate all vectors
        # Shape: (B, Sum(C_i * 2))
        global_representation = torch.cat(pooled_vectors, dim=1)

        # 4. Classification
        logits = self.head(global_representation)

        return logits


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    y_true_all = []
    y_pred_all = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for batch in pbar:
        # Move inputs to device
        x_target = batch["target"].to(device)
        x_contra = batch["contra"].to(device)
        y = batch["label"].to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        # Forward
        logits = model(x_target, x_contra)
        loss = criterion(logits, y)

        # Backward
        loss.backward()

        # Gradient Clipping logic (Disabled per strategy, but kept as option if config changes)
        if Config.CLIP_GRADIENTS:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Metrics
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels = y.detach().cpu().numpy()

        running_loss += loss.item() * x_target.size(0)
        y_true_all.extend(labels)
        y_pred_all.extend(probs)

        pbar.set_postfix({"loss": loss.item()})

    epoch_loss = running_loss / len(loader.dataset)
    epoch_pf1 = probabilistic_f1(y_true_all, y_pred_all)

    return epoch_loss, epoch_pf1


def validate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    y_true_all = []
    y_pred_all = []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)

    with torch.no_grad():
        for batch in pbar:
            x_target = batch["target"].to(device)
            x_contra = batch["contra"].to(device)
            y = batch["label"].to(device).unsqueeze(1)

            logits = model(x_target, x_contra)
            loss = criterion(logits, y)

            probs = torch.sigmoid(logits).cpu().numpy()
            labels = y.cpu().numpy()

            running_loss += loss.item() * x_target.size(0)
            y_true_all.extend(labels)
            y_pred_all.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_pf1 = probabilistic_f1(y_true_all, y_pred_all)

    return epoch_loss, epoch_pf1


def run_training(debug=False):
    """
    Main training orchestration function.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")

    # 1. Prepare Datasets
    logger.info("Initializing datasets...")
    train_dataset = SiameseBreastCancerDataset(
        Config.TRAIN_METADATA, mode="train", debug=debug
    )
    val_dataset = SiameseBreastCancerDataset(
        Config.VAL_METADATA, mode="val", debug=debug
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Initialize Model
    logger.info(f"Initializing model: {Config.BACKBONE}")
    model = PyramidSiameseEfficientNet(backbone_name=Config.BACKBONE, pretrained=True)
    model.to(device)

    # 3. Setup Optimization
    # Aggressive positive weighting for imbalance
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # 4. Training Loop
    best_pf1 = 0.0
    best_epoch = 0

    logger.info("Starting training loop...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        train_loss, train_pf1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_pf1 = validate(model, val_loader, criterion, device, epoch)

        scheduler.step()

        elapsed = time.time() - start_time

        logger.info(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Train pF1: {train_pf1:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val pF1: {val_pf1:.10f} | "  # Full precision for Val pF1
            f"Time: {elapsed:.1f}s"
        )

        # Checkpoint
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            best_epoch = epoch
            torch.save(model.state_dict(), Config.MODEL_PATH)
            logger.info(
                f"New best model saved at epoch {epoch} with pF1 {best_pf1:.10f}"
            )

    logger.info(f"Training finished. Best pF1: {best_pf1:.10f} at epoch {best_epoch}")
    return best_pf1


def predict_and_submit(debug=False):
    """
    Runs inference on the test set and generates the submission file.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Model
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {Config.MODEL_PATH}. Train first."
        )

    logger.info(f"Loading model from {Config.MODEL_PATH}")
    model = PyramidSiameseEfficientNet(backbone_name=Config.BACKBONE, pretrained=False)
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 2. Prepare Test Data
    logger.info("Initializing test dataset...")
    test_dataset = SiameseBreastCancerDataset(
        Config.TEST_METADATA, mode="test", debug=debug
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Inference Loop
    results = []
    logger.info("Starting inference...")

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            x_target = batch["target"].to(device)
            x_contra = batch["contra"].to(device)
            prediction_ids = batch["prediction_id"]

            logits = model(x_target, x_contra)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            for pid, prob in zip(prediction_ids, probs):
                results.append({"prediction_id": pid, "cancer": prob})

    # 4. Aggregation
    # Multiple images (views) share the same prediction_id.
    # We take the MAX probability for each prediction_id.
    df_results = pd.DataFrame(results)

    if df_results.empty:
        logger.warning(
            "No predictions generated. Creating empty submission based on sample."
        )
        # Fallback for empty test set in some environments
        df_sub = pd.read_csv(Config.SAMPLE_SUBMISSION)
        df_sub["cancer"] = 0.0
    else:
        # Group by prediction_id and take max
        df_agg = df_results.groupby("prediction_id")["cancer"].max().reset_index()

        # Ensure all prediction_ids from sample submission are present
        df_sample = pd.read_csv(Config.SAMPLE_SUBMISSION)

        # Merge to ensure order and completeness
        # Left merge on sample to keep all required IDs
        df_sub = pd.merge(
            df_sample[["prediction_id"]], df_agg, on="prediction_id", how="left"
        )

        # Fill missing values (if any) with a low probability
        df_sub["cancer"] = df_sub["cancer"].fillna(0.0)

    # 5. Save Submission
    logger.info(f"Saving submission to {Config.SUBMISSION_PATH}")
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info("Submission saved successfully.")
