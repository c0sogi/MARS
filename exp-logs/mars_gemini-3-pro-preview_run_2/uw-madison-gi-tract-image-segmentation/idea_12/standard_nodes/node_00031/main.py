import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings

# Import from provided library
from library.config import Config
from library.trainer import Trainer
from library.inference import InferenceEngine, run_inference
from library.utils import set_seed, rle_decode, calculate_dice, calculate_hausdorff
from library.data_loader import preprocess_metadata

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print("Setting up configuration for fast baseline run...")

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Override Config for time constraints (Fast Baseline)
    # We use DEBUG mode to limit dataset size, but increase sample size for better learning
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 6000  # Sufficient for a baseline

    # Reduce epochs to ensure completion within 2 hours
    Config.COARSE_EPOCHS = 5
    Config.FINE_EPOCHS = 5

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # =========================================================================
    # 2. Training
    # =========================================================================
    print("\n" + "=" * 40)
    print("STAGE 1: Training Coarse Model")
    print("=" * 40)
    trainer_coarse = Trainer("coarse")
    trainer_coarse.fit()

    print("\n" + "=" * 40)
    print("STAGE 2: Training Fine Model")
    print("=" * 40)
    trainer_fine = Trainer("fine")
    trainer_fine.fit()

    # Clear GPU memory after training
    del trainer_coarse, trainer_fine
    torch.cuda.empty_cache()

    # =========================================================================
    # 3. Validation & Metric Calculation
    # =========================================================================
    print("\n" + "=" * 40)
    print("VALIDATION & METRIC CALCULATION")
    print("=" * 40)

    # Load validation metadata (pivoted to have all classes in one row)
    val_df = preprocess_metadata(Config.VAL_META_PATH, load_cached_data=False)

    # If in debug mode, we might need to filter val_df to match what was used or just use all
    # Since validation is fast, we can use the full validation set (16 cases) for accurate metrics
    # provided in the metadata.

    # Group by Case+Day to form 3D volumes
    val_df["group_key"] = val_df["case"].astype(str) + "_" + val_df["day"].astype(str)
    groups = val_df.groupby("group_key")

    # Initialize Inference Engine with trained models
    engine = InferenceEngine()

    dice_scores = []
    hausdorff_scores = []

    # For Failure Analysis
    failure_data = []

    print(f"Validating on {len(groups)} cases...")

    for group_key, group_df in groups:
        # Sort by slice to ensure 3D consistency
        group_df = group_df.sort_values("slice").reset_index(drop=True)

        # 1. Get Ground Truth 3D Volume
        # Shape: (Depth, C, H, W)
        depth = len(group_df)
        h, w = group_df.iloc[0]["img_height"], group_df.iloc[0]["img_width"]

        gt_volume = np.zeros((depth, Config.NUM_CLASSES, h, w), dtype=np.uint8)

        for idx, row in group_df.iterrows():
            for c_idx, cls in enumerate(Config.CLASS_LABELS):
                rle = row[cls]
                mask = rle_decode(rle, (h, w))
                gt_volume[idx, c_idx, :, :] = mask

        # 2. Run Prediction
        # predict_case returns {slice_id: {class: rle}}
        preds_dict = engine.predict_case(group_df)

        pred_volume = np.zeros((depth, Config.NUM_CLASSES, h, w), dtype=np.uint8)

        # Map predictions back to volume
        for idx, row in group_df.iterrows():
            sid = row["id"]
            if sid in preds_dict:
                for c_idx, cls in enumerate(Config.CLASS_LABELS):
                    rle = preds_dict[sid].get(cls, "")
                    mask = rle_decode(rle, (h, w))
                    pred_volume[idx, c_idx, :, :] = mask

        # 3. Calculate Metrics per Class
        case_dices = []
        case_hausdorffs = []

        for c_idx in range(Config.NUM_CLASSES):
            gt_c = gt_volume[:, c_idx, :, :]
            pred_c = pred_volume[:, c_idx, :, :]

            # Dice (Pixel-wise agreement on volume)
            # Flatten for Dice calculation as per formula
            d = calculate_dice(gt_c.flatten(), pred_c.flatten())
            case_dices.append(d)

            # Hausdorff (3D)
            h_dist = calculate_hausdorff(gt_c, pred_c)
            # Normalize Hausdorff to 0-1 score?
            # The prompt says: "The expected / predicted pixel locations are normalized by image size to create a bounded 0-1 score."
            # calculate_hausdorff in utils.py already normalizes coordinates.
            # However, Hausdorff is a distance (lower is better).
            # The metric combination is 0.4*Dice + 0.6*Hausdorff.
            # Usually, competition metrics use (1 - Hausdorff) or similar if they want to maximize.
            # But the prompt says "Metric: Mean Dice coefficient and 3D Hausdorff distance... combined with a weight of 0.4 for Dice and 0.6 for Hausdorff".
            # It doesn't explicitly say "1 - Hausdorff".
            # However, typically "Score" implies higher is better.
            # If we assume the standard Kaggle UW-Madison metric: Score = 0.4*Dice + 0.6*(1 - Hausdorff).
            # Given the "bounded 0-1 score" comment, the distance itself is 0-1.
            # I will assume the goal is to MAXIMIZE the metric, so I will use (1 - Hausdorff).

            # If Hausdorff > 1.0 (penalty for empty vs non-empty), clip it?
            # utils.py returns 1.0 for empty vs non-empty.
            h_score = 1.0 - h_dist
            h_score = max(0.0, h_score)  # Clip at 0
            case_hausdorffs.append(h_score)

        mean_dice = np.mean(case_dices)
        mean_hausdorff = np.mean(case_hausdorffs)

        dice_scores.append(mean_dice)
        hausdorff_scores.append(mean_hausdorff)

        # 4. Collect Data for Failure Analysis
        # We'll use slice-level Dice for correlation analysis
        # Re-calculate slice-wise dice for granularity
        for idx, row in group_df.iterrows():
            slice_dices = []
            for c_idx in range(Config.NUM_CLASSES):
                d = calculate_dice(gt_volume[idx, c_idx], pred_volume[idx, c_idx])
                slice_dices.append(d)

            avg_slice_dice = np.mean(slice_dices)

            failure_data.append(
                {
                    "error": 1.0 - avg_slice_dice,  # Error magnitude
                    "slice_idx": row["slice"],
                    "img_width": row["img_width"],
                    "img_height": row["img_height"],
                    "pixel_spacing_w": row["pixel_spacing_w"],
                    "pixel_spacing_h": row["pixel_spacing_h"],
                    "case": row["case"],
                    "day": row["day"],
                }
            )

    # Aggregate Metrics
    final_dice = np.mean(dice_scores)
    final_hausdorff = np.mean(hausdorff_scores)

    # Combined Metric
    final_metric = 0.4 * final_dice + 0.6 * final_hausdorff

    print(f"\nValidation Results:")
    print(f"  Mean Dice: {final_dice:.5f}")
    print(f"  Mean 3D Hausdorff Score (1-Dist): {final_hausdorff:.5f}")
    print(f"Final Validation Metric: {final_metric:.18f}")

    # =========================================================================
    # 4. Failure Analysis
    # =========================================================================
    print("\n" + "=" * 40)
    print("FAILURE ANALYSIS")
    print("=" * 40)

    if len(failure_data) > 0:
        fa_df = pd.DataFrame(failure_data)

        # Calculate correlations with Error
        # We select numerical columns
        cols_to_corr = ["slice_idx", "img_width", "pixel_spacing_w", "day"]
        correlations = fa_df[cols_to_corr].corrwith(fa_df["error"])

        print("Correlation between Error (1-Dice) and Metadata features:")
        print(correlations.to_string())

        # Identify worst cases
        worst_slices = fa_df.sort_values("error", ascending=False).head(5)
        print("\nTop 5 Worst Slices (Highest Error):")
        print(
            worst_slices[["case", "day", "slice_idx", "error"]].to_string(index=False)
        )
    else:
        print("No validation data available for failure analysis.")

    # =========================================================================
    # 5. Submission
    # =========================================================================
    print("\n" + "=" * 40)
    print("SUBMISSION GENERATION")
    print("=" * 40)

    THRESHOLD = 0.438

    if final_metric > THRESHOLD:
        print(
            f"Metric ({final_metric:.5f}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        # Free memory before inference
        del engine
        torch.cuda.empty_cache()

        run_inference()
    else:
        print(
            f"Metric ({final_metric:.5f}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()
