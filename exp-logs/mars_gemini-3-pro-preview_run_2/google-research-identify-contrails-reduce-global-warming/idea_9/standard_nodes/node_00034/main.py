import os
import sys
import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Import from the provided library
from library.config import Config
from library.utils import set_seed, dice_coef_metric
from library.dataset import get_dataloader
from library.model import HybridResNetTransformerUNet
from library.train import train_model
from library.predict import predict_and_submit


def perform_failure_analysis(model, val_loader, device):
    """
    Evaluates the model on the validation set, computes the Global Dice,
    and performs failure analysis by correlating errors with metadata.
    """
    print("Starting Failure Analysis and Final Evaluation...")

    model.eval()

    # Metrics accumulators
    total_intersection = 0.0
    total_union = 0.0

    # Per-sample storage
    sample_dices = []

    # Ensure no gradients are computed
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)

            # Inference
            logits = model(images)
            preds = (torch.sigmoid(logits) > Config.THRESHOLD).float()

            # 1. Update Global Dice Stats
            preds_flat = preds.view(-1)
            targets_flat = masks.view(-1)

            intersection = (preds_flat * targets_flat).sum().item()
            union = preds_flat.sum().item() + targets_flat.sum().item()

            total_intersection += intersection
            total_union += union

            # 2. Compute Per-Sample Dice for Failure Analysis
            # shape: (B, 1, H, W)
            batch_size = images.size(0)
            for i in range(batch_size):
                p = preds[i].view(-1)
                t = masks[i].view(-1)

                inter = (p * t).sum().item()
                uni = p.sum().item() + t.sum().item()

                # Smooth dice per sample
                d = (2.0 * inter) / (uni + 1e-6)
                sample_dices.append(d)

    # Compute Final Global Dice
    global_dice = (2.0 * total_intersection) / (total_union + 1e-6)

    # Print required metric format
    print(f"Final Validation Metric: {global_dice:.16f}")

    # --- Failure Analysis ---
    # Get metadata from the dataset
    # The loader preserves order because shuffle=False for validation
    df_val = val_loader.dataset.df.copy()

    # Ensure lengths match (drop last batch issues handled by loader, but val usually doesn't drop)
    # If sizes differ slightly due to drop_last (unlikely for val), truncate df
    if len(sample_dices) != len(df_val):
        print(
            f"Warning: Mismatch in samples ({len(sample_dices)} vs {len(df_val)}). Truncating for analysis."
        )
        min_len = min(len(sample_dices), len(df_val))
        sample_dices = sample_dices[:min_len]
        df_val = df_val.iloc[:min_len]

    df_val["dice"] = sample_dices
    df_val["error"] = 1.0 - df_val["dice"]

    # Extract features from timestamp
    if "timestamp" in df_val.columns:
        df_val["datetime"] = pd.to_datetime(df_val["timestamp"], unit="s")
        df_val["hour"] = df_val["datetime"].dt.hour
        df_val["month"] = df_val["datetime"].dt.month

    # Features to correlate
    features = ["row_min", "col_min", "hour", "month"]
    correlations = {}

    print("\nFailure Analysis - Correlation with Error (1 - Dice):")
    for feat in features:
        if feat in df_val.columns:
            corr = df_val[feat].corr(df_val["error"])
            correlations[feat] = corr
            print(f"  {feat}: {corr:.4f}")

    return global_dice


def main():
    # 1. Setup
    set_seed(Config.SEED)

    # Override Config for Fast Baseline
    # We use full data but fewer epochs to ensure speed while maintaining performance
    Config.EPOCHS = 5
    Config.MAX_TRAIN_SAMPLES = None  # Use full dataset
    Config.MAX_VAL_SAMPLES = None  # Use full validation set

    print(f"Configuration set: Epochs={Config.EPOCHS}, Device={Config.DEVICE}")

    # 2. Train Model
    # This handles training loop and saves best_model.pth
    train_model(
        max_train_samples=Config.MAX_TRAIN_SAMPLES,
        max_val_samples=Config.MAX_VAL_SAMPLES,
        epochs=Config.EPOCHS,
    )

    # 3. Load Best Model for Analysis
    print(f"\nLoading best model from {Config.MODEL_PATH}...")
    model = HybridResNetTransformerUNet()
    model.to(Config.DEVICE)

    if os.path.exists(Config.MODEL_PATH):
        state_dict = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE)
        model.load_state_dict(state_dict)
    else:
        print("Error: Model checkpoint not found!")
        return

    # 4. Validation & Failure Analysis
    val_loader = get_dataloader(split="validation", batch_size=Config.BATCH_SIZE)
    final_metric = perform_failure_analysis(model, val_loader, Config.DEVICE)

    # 5. Submission Logic
    # Threshold from requirements
    THRESHOLD_SCORE = 0.5676456935477064

    if final_metric > THRESHOLD_SCORE:
        print(
            f"\nMetric ({final_metric:.6f}) > Threshold ({THRESHOLD_SCORE:.6f}). Generating submission..."
        )
        predict_and_submit(max_samples=None)
    else:
        print(
            f"\nMetric ({final_metric:.6f}) <= Threshold ({THRESHOLD_SCORE:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
