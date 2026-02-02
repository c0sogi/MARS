import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

# Import from library
from library.config import Config
from library.dataset import SaltDataset
from library.model import ResNet34WideLinkNet
from library.engine import train_model, validate, predict_and_submit
from library.utils import unpad_image, calc_iou

# -------------------------------------------------------------------------
# Helper Classes & Functions
# -------------------------------------------------------------------------


class SafeSaltDataset(SaltDataset):
    """
    Subclass of SaltDataset that handles missing depths (NaNs) in the test set
    by filling them with the mean of valid depths. This prevents NaN propagation
    during Multi-Task Loss calculation in Stage 3.
    """

    def _load_depths_from_disk(self):
        # Load raw depths
        vals = self.df["z"].values.astype(np.float32)

        # Check for NaNs
        mask = np.isnan(vals)
        if mask.any():
            # Calculate mean of valid values
            valid_mean = np.nanmean(vals)
            if np.isnan(valid_mean):
                valid_mean = 0.0

            # Fill NaNs
            vals[mask] = valid_mean

        return vals


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def run_pipeline():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration
    # -------------------------------------------------------------------------
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # Override Config for runtime constraints
    Config.EPOCHS = 50
    Config.BATCH_SIZE = 32
    PATIENCE = 10

    # Ensure output directories exist
    os.makedirs(Config.CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Supervised Training (Single Model)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(" Supervised Training (ResNet34 + Depth Injection)")
    print("=" * 40)

    # Load datasets
    train_ds = SaltDataset(mode="train", load_cached_data=True)
    val_ds = SaltDataset(mode="val", load_cached_data=True)

    # Dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    model = ResNet34WideLinkNet().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
    )

    save_path = os.path.join(Config.CHECKPOINTS_DIR, "best_model.pth")

    # Train
    best_threshold = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        epochs=Config.EPOCHS,
        patience=PATIENCE,
        save_path=save_path,
    )

    # -------------------------------------------------------------------------
    # Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(" Validation & Failure Analysis")
    print("=" * 40)

    # Load best model
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    # Get predictions on validation set
    val_loss, val_map, val_probs, val_masks = validate(
        model, val_loader, device, return_probs=True
    )

    print(f"Final Validation Metric: {val_map:.10f}")

    # Failure Analysis: Correlation between IoU and Features
    val_df = val_ds.df
    ious = []
    depths = val_df["z"].values
    coverages = val_df["salt_coverage"].values

    for prob, mask in zip(val_probs, val_masks):
        pred_bin = (prob > best_threshold).astype(np.uint8)
        iou = calc_iou(pred_bin, mask)
        ious.append(iou)

    ious = np.array(ious)

    # Handle potential NaNs in correlation calculation
    if len(ious) > 1:
        corr_depth, _ = pearsonr(depths, ious)
        corr_cov, _ = pearsonr(coverages, ious)
    else:
        corr_depth, corr_cov = 0.0, 0.0

    print("-" * 30)
    print("Failure Analysis Report")
    print("-" * 30)
    print(f"Correlation (IoU vs Depth): {corr_depth:.4f}")
    print(f"Correlation (IoU vs Salt Coverage): {corr_cov:.4f}")

    # -------------------------------------------------------------------------
    # Submission
    # -------------------------------------------------------------------------
    if val_map > 0.7985:
        print("\n=== Generating Submission ===")

        # Load Test Set (SafeSaltDataset handles missing depths if any, though standard SaltDataset might be fine if test.csv has depths)
        # Using SafeSaltDataset just in case test depths are missing
        test_ds_final = SafeSaltDataset(mode="test", load_cached_data=True)
        test_loader_final = DataLoader(
            test_ds_final,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        predict_and_submit(
            model,
            test_loader_final,
            device,
            best_threshold,
            Config.SUBMISSION_PATH,
        )
    else:
        print(
            f"\nValidation metric {val_map:.4f} did not meet threshold 0.7985. Submission skipped."
        )


if __name__ == "__main__":
    run_pipeline()
