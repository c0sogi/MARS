import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings

# Add library to path just in case, though imports assume it's accessible
sys.path.append(os.path.abspath("."))

from library.config import Config
from library.train import train_model
from library.inference import predict_sliding_window, make_submission
from library.utils import rle_decode, set_seed
from library.model import UnetPlusPlus

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def compute_dice(mask1, mask2):
    """Computes Dice coefficient between two binary masks."""
    smooth = 1e-6
    intersection = (mask1 * mask2).sum()
    union = mask1.sum() + mask2.sum()
    return (2.0 * intersection + smooth) / (union + smooth)


def run_full_validation(model, device):
    """
    Runs sliding window inference on the full validation set to get accurate metrics.
    Returns a DataFrame containing IDs, Dice scores, and Metadata.
    """
    print("\n--- Starting Full Validation Inference ---")

    # Load validation metadata
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    results = []

    for _, row in val_df.iterrows():
        img_id = row["id"]
        h = row["height_pixels"]
        w = row["width_pixels"]

        # Paths
        # Image path in metadata is relative to input dir (e.g., "train/id.tiff")
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        anat_path = row["anatomical_json_path"]  # Relative path handled by utils

        print(f"Validating {img_id} ({w}x{h})...")

        try:
            # 1. Generate Prediction
            pred_mask = predict_sliding_window(
                model=model,
                image_path=img_path,
                image_id=img_id,
                anatomical_json_path=anat_path,
                height=h,
                width=w,
                device=device,
            )

            # 2. Load Ground Truth
            if pd.notna(row["encoding"]):
                gt_mask = rle_decode(row["encoding"], (h, w))
            else:
                gt_mask = np.zeros((h, w), dtype=np.uint8)

            # 3. Compute Metric
            dice = compute_dice(pred_mask, gt_mask)

            # Store result
            res = row.to_dict()
            res["dice"] = dice
            res["error"] = 1.0 - dice
            results.append(res)

        except Exception as e:
            print(f"Error validating {img_id}: {e}")

    return pd.DataFrame(results)


def perform_failure_analysis(results_df):
    """
    Correlates model error (1 - Dice) with metadata features.
    """
    print("\n--- Failure Analysis ---")

    if results_df.empty:
        print("No validation results to analyze.")
        return

    # Features to analyze
    features = [
        "age",
        "weight_kilograms",
        "bmi_kg/m^2",
        "percent_cortex",
        "percent_medulla",
        "height_centimeters",
    ]

    # Filter for existing columns
    existing_features = [f for f in features if f in results_df.columns]

    if not existing_features:
        print("No metadata features available for correlation analysis.")
        return

    # Compute correlations with 'error'
    # Use 'float' type to avoid object dtype issues
    analysis_df = results_df[existing_features + ["error"]].apply(
        pd.to_numeric, errors="coerce"
    )

    correlations = analysis_df.corr()["error"].drop("error")

    print("Correlation between Error (1 - Dice) and Features:")
    print(correlations.sort_values(ascending=False))

    # Identify worst cases
    print("\nWorst Performing Images:")
    worst = results_df.sort_values("dice").head(3)
    for _, row in worst.iterrows():
        print(f"ID: {row['id']}, Dice: {row['dice']:.4f}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running pipeline on {device}")

    # 2. Train Model
    # Fast baseline settings: 5 epochs, 500 samples/epoch
    print("\n--- Starting Training ---")
    train_model(
        num_epochs=5, batch_size=Config.BATCH_SIZE, samples_per_epoch=500, patience=3
    )

    # 3. Load Best Model for Validation
    print("\n--- Loading Best Model ---")
    model = UnetPlusPlus(
        backbone_name=Config.BACKBONE,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
    )

    if os.path.exists(Config.CHECKPOINT_PATH):
        state_dict = torch.load(Config.CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(
            "Error: Checkpoint not found. Using random weights (Metrics will be poor)."
        )

    model.to(device)
    model.eval()

    # 4. Full Validation
    val_results = run_full_validation(model, device)

    if val_results.empty:
        print("Validation failed to produce results.")
        return

    final_metric = val_results["dice"].mean()

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    perform_failure_analysis(val_results)

    # 6. Submission
    THRESHOLD = 0.9132
    if final_metric > THRESHOLD:
        print(
            f"\nValidation Metric ({final_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )
        make_submission(
            checkpoint_path=Config.CHECKPOINT_PATH, output_path=Config.SUBMISSION_PATH
        )
    else:
        print(
            f"\nValidation Metric ({final_metric}) <= Threshold ({THRESHOLD}). Skipping Submission."
        )


if __name__ == "__main__":
    main()
