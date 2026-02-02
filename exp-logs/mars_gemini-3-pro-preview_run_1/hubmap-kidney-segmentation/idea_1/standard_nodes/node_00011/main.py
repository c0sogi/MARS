import os
import sys
import pandas as pd
import numpy as np
import torch
import gc

# Import from the provided library
from library.config import Config
from library.utils import set_seed, rle_decode, dice_coeff
from library.train import train_model
from library.inference import generate_submission, predict_sliding_window
from library.model import LinkNetResNet34


def run_failure_analysis(val_df, model, device):
    """
    Performs validation on full images and calculates correlations between
    error magnitude and metadata features.
    """
    print("Starting Validation and Failure Analysis...")

    results = []
    dice_scores = []

    model.eval()

    for idx, row in val_df.iterrows():
        img_id = row["id"]
        img_path = row["image_path"]
        rle_mask = row["encoding"]

        # 1. Predict
        try:
            # Predict full mask using sliding window
            pred_mask = predict_sliding_window(model, img_path, device)

            # 2. Load Ground Truth
            # We need the image dimensions to decode the RLE
            # predict_sliding_window returns a mask of the correct shape (H, W)
            h, w = pred_mask.shape
            gt_mask = rle_decode(rle_mask, (h, w))

            # 3. Calculate Dice
            # dice_coeff expects tensors or numpy arrays
            score = dice_coeff(pred_mask, gt_mask)
            dice_scores.append(score)

            # Store result for analysis
            results.append({"id": img_id, "dice": score, "error": 1.0 - score})

        except Exception as e:
            print(f"Error validating image {img_id}: {e}")
            dice_scores.append(0.0)
            results.append({"id": img_id, "dice": 0.0, "error": 1.0})

        # Cleanup
        gc.collect()

    # Calculate Global Metric
    mean_dice = np.mean(dice_scores) if dice_scores else 0.0
    print(f"Final Validation Metric: {mean_dice}")

    # --- Failure Analysis ---
    results_df = pd.DataFrame(results)

    # Merge with metadata to get features
    analysis_df = pd.merge(results_df, val_df, on="id", how="left")

    # Select numerical features for correlation
    feature_cols = [
        "age",
        "weight_kilograms",
        "bmi_kg/m^2",
        "percent_cortex",
        "percent_medulla",
    ]

    print("\nFailure Analysis (Correlation with Error Magnitude):")
    for col in feature_cols:
        if col in analysis_df.columns:
            # Drop NaNs for correlation calculation
            valid_data = analysis_df[["error", col]].dropna()
            if len(valid_data) > 1:
                corr = valid_data["error"].corr(valid_data[col])
                print(f"  {col}: {corr:.4f}")
            else:
                print(f"  {col}: Not enough data")


def main():
    # 1. Configuration
    set_seed(Config.SEED)

    print("--- Step 1: Training Model ---")
    # Train the model
    # This saves the best model to Config.MODEL_CHECKPOINT_PATH
    # Using Config.EPOCHS (20) defined in config.py
    best_dice = train_model(epochs=Config.EPOCHS, load_cached_data=True)

    print("\n--- Step 2: Validation & Failure Analysis ---")
    # Load Validation Metadata
    if not os.path.exists(Config.VAL_METADATA_PATH):
        print("Validation metadata not found. Skipping analysis.")
    else:
        val_df = pd.read_csv(Config.VAL_METADATA_PATH)

        # Load the best model
        device = torch.device(Config.DEVICE)
        model = LinkNetResNet34(in_channels=Config.IN_CHANNELS, classes=Config.CLASSES)

        if os.path.exists(Config.MODEL_CHECKPOINT_PATH):
            state_dict = torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)

            # Run analysis
            run_failure_analysis(val_df, model, device)
        else:
            print("Model checkpoint not found. Skipping analysis.")

    print("\n--- Step 3: Generating Submission ---")
    # Generate submission for the test set only if validation metric meets threshold
    if best_dice > 0.8836:
        print(f"Validation Dice ({best_dice:.4f}) > 0.8836. Generating submission...")
        generate_submission()
    else:
        print(f"Validation Dice ({best_dice:.4f}) <= 0.8836. Skipping submission.")


if __name__ == "__main__":
    main()
