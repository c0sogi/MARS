import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import ContrailDataset
from library.model import ResNetUNet
from library.train import run_training
from library.predict import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for Fast Baseline execution
    Config.EPOCHS = 5
    Config.DEBUG_SAMPLE_SIZE = 5000  # Limit dataset size for speed
    Config.BATCH_SIZE = 32

    # Ensure reproducible results
    Config.setup()
    seed_everything(Config.SEED)

    print(
        f"Configuration: Epochs={Config.EPOCHS}, Sample Size={Config.DEBUG_SAMPLE_SIZE}"
    )

    # ==========================================
    # 2. Training
    # ==========================================
    # run_training handles data loading, model init, training loop, and saving best model
    print("\n--- Starting Training ---")
    best_model_path = run_training()

    # ==========================================
    # 3. Validation & Failure Analysis
    # ==========================================
    print("\n--- Starting Validation & Failure Analysis ---")
    device = torch.device(Config.DEVICE)

    # Load validation metadata
    val_df = pd.read_csv(Config.VALIDATION_METADATA_PATH)
    if Config.DEBUG_SAMPLE_SIZE is not None:
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Initialize Validation Dataset and Loader
    val_dataset = ContrailDataset(val_df, split="validation")
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,  # Important: Must be False to align with metadata for analysis
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # Load the best trained model
    model = ResNetUNet(in_channels=Config.IN_CHANNELS, num_classes=1)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    # Accumulators for Global Dice
    intersection_sum = 0.0
    union_sum = 0.0

    # Accumulator for Per-Sample Failure Analysis
    sample_errors = []

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            # Inference
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds_bin = (probs > Config.THRESHOLD).float()

            # --- Global Dice Calculation ---
            preds_flat = preds_bin.view(-1)
            masks_flat = masks.view(-1)

            intersection_sum += (preds_flat * masks_flat).sum().item()
            union_sum += preds_flat.sum().item() + masks_flat.sum().item()

            # --- Per-Sample Error Calculation for Failure Analysis ---
            # Iterate over batch to calculate individual Dice scores
            batch_size = images.size(0)
            for i in range(batch_size):
                p = preds_bin[i].view(-1)
                t = masks[i].view(-1)

                inter = (p * t).sum().item()
                union = p.sum().item() + t.sum().item()

                # Dice = 2*Inter / (Union + smooth)
                # Error = 1 - Dice
                dice = (2.0 * inter) / (union + 1e-6)
                sample_errors.append(1.0 - dice)

    # Compute and Print Final Validation Metric
    smooth = 1e-6
    final_global_dice = (2.0 * intersection_sum) / (union_sum + smooth)
    print(f"Final Validation Metric: {final_global_dice}")

    # --- Failure Analysis ---
    # Add error magnitude to the dataframe
    # Since shuffle=False, indices align
    analysis_df = val_df.iloc[: len(sample_errors)].copy()
    analysis_df["error_magnitude"] = sample_errors

    # Calculate correlations with metadata features
    features_to_analyze = ["timestamp", "row_min", "col_min", "row_size", "col_size"]
    correlations = {}

    print("\nFailure Analysis (Correlation of Input Features with Error Magnitude):")
    for feat in features_to_analyze:
        if feat in analysis_df.columns:
            # Ensure numeric
            try:
                corr = (
                    analysis_df[feat].astype(float).corr(analysis_df["error_magnitude"])
                )
                correlations[feat] = corr
                print(f"  {feat}: {corr:.4f}")
            except Exception as e:
                print(f"  {feat}: Could not calculate correlation ({e})")

    # ==========================================
    # 4. Submission
    # ==========================================
    print("\n--- Generating Submission ---")
    generate_submission(best_model_path, device=device)


if __name__ == "__main__":
    main()
