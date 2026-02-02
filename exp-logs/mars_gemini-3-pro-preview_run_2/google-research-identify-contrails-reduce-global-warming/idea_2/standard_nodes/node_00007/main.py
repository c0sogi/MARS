import pandas as pd
import torch
import numpy as np
import os
import sys

# Import necessary components from the provided library
from library.config import (
    DEVICE,
    MODEL_SAVE_PATH,
    VALIDATION_METADATA_PATH,
    BATCH_SIZE,
    SUBMISSION_FILE_PATH,
)
from library.utils import set_seed
from library.dataset import get_dataloader
from library.model import UNet
from library.train import train_model
from library.predict import predict_and_submit


def main():
    # 1. Setup
    set_seed(42)
    print("Initializing Fast Baseline Pipeline...")

    # 2. Training
    # We use debug=False to train on the full dataset for better performance.
    # We increase epochs to 12 to ensure convergence (Cite solution_lesson_node_00002).
    print("\n=== Starting Training ===")
    train_model(debug=False, epochs=12, batch_size=BATCH_SIZE)

    # 3. Validation & Metric Calculation
    print("\n=== Starting Validation & Failure Analysis ===")

    # Load the best model
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Model file not found at {MODEL_SAVE_PATH}")
        return

    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    val_loader = get_dataloader("validation", batch_size=BATCH_SIZE, debug=False)
    val_df = val_loader.dataset.df

    # Accumulators for Global Dice (Metric)
    total_intersection = 0.0
    total_union = 0.0

    # List to store per-sample metrics for failure analysis
    sample_analysis_data = []

    with torch.no_grad():
        for i, (images, masks) in enumerate(val_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            # Inference
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds_bin = (probs > 0.5).float()

            # --- Global Dice Calculation Updates ---
            # Flatten to 1D for accurate global counts
            preds_flat = preds_bin.view(-1)
            masks_flat = masks.view(-1)

            intersection = (preds_flat * masks_flat).sum().item()
            union = preds_flat.sum().item() + masks_flat.sum().item()

            total_intersection += intersection
            total_union += union

            # --- Per-Sample Analysis ---
            # We need to compute dice per image to correlate with metadata
            batch_size_curr = images.size(0)
            start_idx = i * BATCH_SIZE

            for j in range(batch_size_curr):
                # Extract single sample
                p_s = preds_bin[j].view(-1)
                t_s = masks[j].view(-1)

                inter_s = (p_s * t_s).sum().item()
                union_s = p_s.sum().item() + t_s.sum().item()

                # Dice per sample (smooth to avoid div by zero)
                dice_s = (2.0 * inter_s + 1e-6) / (union_s + 1e-6)

                # Get record_id
                # val_loader.dataset.df is the dataframe used by the dataset
                record_id = val_df.iloc[start_idx + j]["record_id"]

                sample_analysis_data.append(
                    {
                        "record_id": str(record_id),
                        "dice": dice_s,
                        "error": 1.0 - dice_s,  # Error magnitude
                    }
                )

    # Compute Final Global Dice
    # Formula: 2 * |X n Y| / (|X| + |Y|)
    global_dice = (2.0 * total_intersection) / (total_union + 1e-6)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {global_dice}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")

    # Load metadata
    if os.path.exists(VALIDATION_METADATA_PATH):
        meta_df = pd.read_csv(VALIDATION_METADATA_PATH)
        meta_df["record_id"] = meta_df["record_id"].astype(str)

        # Create analysis dataframe
        analysis_df = pd.DataFrame(sample_analysis_data)

        # Merge metrics with metadata
        merged_df = analysis_df.merge(meta_df, on="record_id", how="left")

        # Feature Engineering for Time
        if "timestamp" in merged_df.columns:
            merged_df["datetime"] = pd.to_datetime(merged_df["timestamp"], unit="s")
            merged_df["hour"] = merged_df["datetime"].dt.hour
            merged_df["month"] = merged_df["datetime"].dt.month

        # Calculate Correlations
        features_to_check = ["row_min", "col_min", "hour", "month"]
        print("Correlation between Error Magnitude (1 - Dice) and Features:")

        for feat in features_to_check:
            if feat in merged_df.columns:
                corr = merged_df[feat].corr(merged_df["error"])
                print(f"  {feat}: {corr:.6f}")
    else:
        print("Warning: Validation metadata not found. Skipping correlation analysis.")

    # 5. Submission
    # Threshold check
    THRESHOLD = 0.5454606988733747

    if global_dice > THRESHOLD:
        print(
            f"\nMetric ({global_dice:.6f}) exceeds threshold ({THRESHOLD:.6f}). Generating submission..."
        )
        predict_and_submit(debug=False)
    else:
        print(
            f"\nMetric ({global_dice:.6f}) does not exceed threshold ({THRESHOLD:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()
