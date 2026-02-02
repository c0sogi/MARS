import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import ContrailDataset, get_transforms
from library.model import AttentionGatedConvNeXtUNet
from library.loss import HybridBCEDiceLoss
from library.engine import train_model, valid_one_epoch
from library.inference import run_inference


def analyze_failures(model, dataloader, device):
    """
    Performs failure analysis by correlating error magnitude with metadata features.
    """
    print("Running Failure Analysis...")
    model.eval()

    sample_dices = []
    gt_areas = []

    # Iterate over validation set to get sample-wise metrics
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Forward
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > Config.THRESHOLD).float()

            # Compute Dice per sample (Batch size B)
            # masks shape: (B, 1, H, W)
            intersection = (preds * masks).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))

            # Add epsilon to avoid div by zero for empty masks
            dice = (2.0 * intersection) / (union + 1e-6)

            sample_dices.extend(dice.cpu().numpy())
            gt_areas.extend(masks.sum(dim=(1, 2, 3)).cpu().numpy())

    # Calculate Error Magnitude (1 - Dice)
    errors = 1.0 - np.array(sample_dices)
    gt_areas = np.array(gt_areas)

    # Retrieve Metadata from the dataset's dataframe
    # The dataloader is sequential (shuffle=False), so indices align
    df = dataloader.dataset.df

    # Extract features
    # 1. Temporal: Hour of day
    if "timestamp" in df.columns:
        timestamps = df["timestamp"].values
        dt = pd.to_datetime(timestamps, unit="s")
        hours = dt.hour
    else:
        hours = np.zeros(len(errors))

    # 2. Spatial: Row Min (Latitude proxy), Col Min (Longitude proxy)
    row_mins = (
        df["row_min"].values if "row_min" in df.columns else np.zeros(len(errors))
    )
    col_mins = (
        df["col_min"].values if "col_min" in df.columns else np.zeros(len(errors))
    )

    features = {
        "Hour of Day": hours,
        "Row Min (Lat)": row_mins,
        "Col Min (Lon)": col_mins,
        "GT Contrail Area": gt_areas,
    }

    print("-" * 30)
    print("Failure Analysis: Correlation with Error (1 - Dice)")
    print("-" * 30)

    for name, feat_vals in features.items():
        # Check for constant values or NaNs
        if np.all(feat_vals == feat_vals[0]) or np.isnan(feat_vals).any():
            continue

        # Pearson Correlation
        corr, _ = pearsonr(feat_vals, errors)
        print(f"{name:<20}: {corr:.4f}")
    print("-" * 30)


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for Fast Baseline
    Config.EPOCHS = 10  # Reduced epochs for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 5000  # Train on 5000 samples for robust baseline
    Config.BATCH_SIZE = 32

    # Setup directories and seeds
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running on device: {device}")
    print(
        f"Training with {Config.DEBUG_SAMPLE_SIZE} samples for {Config.EPOCHS} epochs."
    )

    # ==========================================
    # 2. Data Loading
    # ==========================================
    train_transforms = get_transforms("train")
    valid_transforms = get_transforms("validation")

    # Train Dataset: Subsampled via debug=True
    train_dataset = ContrailDataset(
        split="train", transform=train_transforms, debug=True, load_cached_data=True
    )

    # Validation Dataset: Full set (debug=False) for valid metric
    valid_dataset = ContrailDataset(
        split="validation",
        transform=valid_transforms,
        debug=False,
        load_cached_data=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    model = AttentionGatedConvNeXtUNet()
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    criterion = HybridBCEDiceLoss()

    # ==========================================
    # 4. Training
    # ==========================================
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        epochs=Config.EPOCHS,
    )

    # ==========================================
    # 5. Final Validation
    # ==========================================
    # Load best model weights
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Compute metric on full validation set
    _, final_dice = valid_one_epoch(
        model, valid_loader, criterion, device, threshold=Config.THRESHOLD
    )

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_dice}")

    # ==========================================
    # 6. Failure Analysis
    # ==========================================
    analyze_failures(model, valid_loader, device)

    # ==========================================
    # 7. Submission
    # ==========================================
    threshold_score = 0.5910660985501295

    if final_dice > threshold_score:
        print(
            f"Validation score ({final_dice}) exceeds threshold ({threshold_score}). Generating submission..."
        )
        run_inference(
            batch_size=Config.BATCH_SIZE,
            num_workers=Config.NUM_WORKERS,
            device=device,
            threshold=Config.THRESHOLD,
        )
    else:
        print(
            f"Validation score ({final_dice}) did not exceed threshold ({threshold_score}). Submission skipped."
        )


if __name__ == "__main__":
    main()
