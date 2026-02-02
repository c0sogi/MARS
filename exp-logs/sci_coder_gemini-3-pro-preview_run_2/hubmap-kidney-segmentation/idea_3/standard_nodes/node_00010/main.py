"""
Implementation of the end-to-end pipeline for HuBMAP glomerulus detection.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import rasterio

# Import library components
from library.config import Config
from library.utils import seed_everything, rle_decode, polygons_to_mask
from library.model import AttentionUNetResNet34
from library.train import run_training
from library.inference import predict_sliding_window, generate_submission


def compute_image_dice(pred_mask, gt_mask):
    """
    Computes the Dice coefficient for a single image.
    """
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)

    # Handle edge case where both are empty (perfect prediction of background)
    if union == 0:
        return 1.0

    return (2.0 * intersection) / union


def run_validation_and_analysis():
    """
    Runs full-image inference on the validation set, computes the official metric,
    and performs failure analysis by correlating errors with metadata.
    """
    print("Starting Validation and Failure Analysis...")

    # Check paths
    if not os.path.exists(Config.VAL_METADATA_PATH):
        print(f"Error: Validation metadata not found at {Config.VAL_METADATA_PATH}")
        return 0.0

    if not os.path.exists(Config.MODEL_CHECKPOINT_PATH):
        print(f"Error: Model checkpoint not found at {Config.MODEL_CHECKPOINT_PATH}")
        return 0.0

    # Load Metadata
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    print(f"Validation set size: {len(val_df)} images")

    # Load Model
    device = torch.device(Config.DEVICE)
    model = AttentionUNetResNet34(
        in_channels=Config.IN_CHANNELS, num_classes=Config.NUM_CLASSES, pretrained=False
    )
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()

    dice_scores = []

    # Iterate through validation images
    for idx, row in val_df.iterrows():
        img_id = row["id"]
        img_path = os.path.join(Config.INPUT_DIR, row["image_path"])
        anat_path = os.path.join(Config.INPUT_DIR, row["anatomical_json_path"])

        # 1. Load Image
        try:
            with rasterio.open(img_path) as src:
                image = src.read().transpose(1, 2, 0)  # (H, W, C)
                if image.shape[2] > 3:
                    image = image[:, :, :3]
        except Exception as e:
            print(f"Error loading {img_id}: {e}")
            continue

        h, w = image.shape[:2]

        # 2. Load Ground Truth Mask
        if "encoding" in row and pd.notna(row["encoding"]):
            gt_mask = rle_decode(row["encoding"], (h, w))
        else:
            gt_mask = np.zeros((h, w), dtype=np.uint8)

        # 3. Load Anatomical Mask (Context)
        anat_mask = polygons_to_mask(anat_path, (h, w), label_name="Cortex")

        # 4. Prepare Input (4 Channels)
        anat_exp = np.expand_dims(anat_mask, axis=-1)
        input_image = np.concatenate([image, anat_exp], axis=2)

        # 5. Inference
        prob_map = predict_sliding_window(
            input_image,
            model,
            device,
            tile_size=Config.TILE_SIZE,
            overlap=Config.OVERLAP_STRIDE,
        )

        # 6. Post-Processing (Anatomical Filter)
        prob_map = prob_map * anat_mask
        pred_mask = (prob_map > Config.PREDICTION_THRESHOLD).astype(np.uint8)

        # 7. Compute Metric
        dice = compute_image_dice(pred_mask, gt_mask)
        dice_scores.append(dice)

        # Record error for analysis
        val_df.at[idx, "dice"] = dice
        val_df.at[idx, "error"] = 1.0 - dice

        print(f"  Image {img_id}: Dice = {dice:.4f}")

    # Compute Final Metric
    final_metric = np.mean(dice_scores) if dice_scores else 0.0
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\n=== Failure Analysis ===")
    print("Correlation between Error (1 - Dice) and Metadata features:")

    # Define features to check
    # Numerical features
    features = [
        "age",
        "weight_kilograms",
        "height_centimeters",
        "bmi_kg/m^2",
        "percent_cortex",
        "percent_medulla",
    ]

    # Encode 'sex' if present
    if "sex" in val_df.columns:
        val_df["sex_encoded"] = val_df["sex"].map({"Male": 0, "Female": 1})
        features.append("sex_encoded")

    # Calculate correlations
    correlations = {}
    for feat in features:
        if feat in val_df.columns:
            # Create a clean subset (drop NaNs for this pair)
            subset = val_df[[feat, "error"]].dropna()

            # Need at least 2 points for correlation
            if len(subset) > 1:
                try:
                    # Ensure numeric types
                    x = pd.to_numeric(subset[feat])
                    y = pd.to_numeric(subset["error"])

                    if x.std() > 0 and y.std() > 0:
                        corr = np.corrcoef(x, y)[0, 1]
                        correlations[feat] = corr
                    else:
                        correlations[feat] = 0.0  # No variance
                except Exception:
                    pass

    # Sort and print
    sorted_corrs = sorted(
        correlations.items(), key=lambda item: abs(item[1]), reverse=True
    )

    if not sorted_corrs:
        print("  No sufficient data for correlation analysis.")
    else:
        for feat, corr in sorted_corrs:
            print(f"  {feat}: {corr:.4f}")

    return final_metric


def main():
    # 1. Setup & Configuration Override
    # We override epochs to ensure the baseline runs quickly within the time limit.
    Config.EPOCHS = 5
    Config.WARMUP_EPOCHS = 2
    Config.EARLY_STOPPING_PATIENCE = 2

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # 2. Training
    print("=== Starting Training Pipeline ===")
    run_training()

    # 3. Validation & Analysis
    print("\n=== Starting Validation Pipeline ===")
    final_metric = run_validation_and_analysis()

    # 4. Submission
    # Threshold defined in task description logic (implied by "If and only if... > 0.8873")
    SUBMISSION_THRESHOLD = 0.8873

    if final_metric > SUBMISSION_THRESHOLD:
        print(
            f"\nValidation Metric ({final_metric:.6f}) exceeds threshold ({SUBMISSION_THRESHOLD})."
        )
        print("Generating submission file...")
        generate_submission()
    else:
        print(
            f"\nValidation Metric ({final_metric:.6f}) does not exceed threshold ({SUBMISSION_THRESHOLD})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
