import os
import sys
import numpy as np
import pandas as pd
import torch
import gc
from scipy.stats import pearsonr

# Ensure library is in path
sys.path.append(".")

from library.config import Config, seed_everything
from library.train import run_training
from library.inference import generate_submission_csv, predict_wsi
from library.model import FPNResNet18
from library.utils import rle_decode


def compute_numpy_dice(mask1, mask2):
    """Computes Dice coefficient between two binary numpy arrays."""
    intersection = np.sum(mask1 * mask2)
    sum_areas = np.sum(mask1) + np.sum(mask2)
    if sum_areas == 0:
        return 1.0
    return (2.0 * intersection) / sum_areas


def run_validation_and_analysis():
    """
    Runs inference on the validation set, computes metrics, and performs failure analysis.
    """
    print("\n--- Starting Validation & Failure Analysis ---")

    # 1. Load Validation Metadata
    val_df = pd.read_csv(Config.VAL_METADATA)

    # 2. Load Model
    device = torch.device(Config.DEVICE)
    model = FPNResNet18()

    if os.path.exists(Config.MODEL_SAVE_PATH):
        state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded best model from {Config.MODEL_SAVE_PATH}")
    else:
        print("Warning: Model checkpoint not found. Using random weights.")

    model.to(device)
    model.eval()

    # 3. Inference and Metric Calculation
    dice_scores = []
    errors = []

    print(f"Evaluating on {len(val_df)} validation images...")

    for _, row in val_df.iterrows():
        image_id = row["id"]
        image_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        anat_path = os.path.join(Config.INPUT_DIR, row["anatomical_json_path"])
        gt_rle = row["encoding"]
        height = row["height_pixels"]
        width = row["width_pixels"]

        # Generate Prediction
        try:
            pred_rle = predict_wsi(
                model=model,
                image_path=image_path,
                anat_path=anat_path,
                device=device,
                threshold=0.5,
            )
        except Exception as e:
            print(f"Error predicting {image_id}: {e}")
            pred_rle = ""

        # Decode Masks
        shape = (height, width)
        pred_mask = rle_decode(pred_rle, shape)

        # Handle NaN GT (though unlikely in train/val)
        if pd.isna(gt_rle):
            gt_mask = np.zeros(shape, dtype=np.uint8)
        else:
            gt_mask = rle_decode(gt_rle, shape)

        # Compute Dice
        dice = compute_numpy_dice(pred_mask, gt_mask)
        dice_scores.append(dice)
        errors.append(1.0 - dice)

        # Clean up memory
        del pred_mask, gt_mask
        gc.collect()

    final_metric = np.mean(dice_scores)
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Add error to dataframe for correlation
    val_df["error"] = errors

    # Select numerical features for correlation
    features = [
        "age",
        "weight_kilograms",
        "bmi_kg/m^2",
        "percent_cortex",
        "percent_medulla",
        "height_centimeters",
    ]

    print("Correlation between Model Error (1-Dice) and Features:")
    for feature in features:
        if feature in val_df.columns:
            # Drop NaNs for calculation
            valid_data = val_df[[feature, "error"]].dropna()
            if len(valid_data) > 1:
                corr, _ = pearsonr(valid_data[feature], valid_data["error"])
                print(f"  {feature}: {corr:.4f}")
            else:
                print(f"  {feature}: Not enough data")
        else:
            print(f"  {feature}: Feature not found")


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Training
    # Using 5 epochs for a fast baseline as per requirements
    print("\n--- Starting Training ---")
    run_training(epochs=5, batch_size=Config.BATCH_SIZE)

    # 3. Validation & Failure Analysis
    run_validation_and_analysis()

    # 4. Submission
    print("\n--- Generating Submission ---")
    generate_submission_csv()

    print("\nPipeline execution complete.")


if __name__ == "__main__":
    main()
