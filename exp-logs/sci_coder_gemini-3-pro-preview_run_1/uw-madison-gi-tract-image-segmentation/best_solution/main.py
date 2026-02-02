import os
import sys
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr

# Ensure library modules are importable
sys.path.append(os.getcwd())

from library.config import (
    SEED,
    VAL_CSV,
    CHECKPOINT_DIR,
    SUBMISSION_PATH,
    WORKING_DIR,
    set_seed,
)
from library.train import train_model
from library.evaluate import (
    load_model,
    predict_dataset,
    process_volume,
    generate_submission,
)


def main():
    # Set reproducibility
    set_seed(SEED)

    print("=== Step 1: Model Training ===")
    # Train for 5 epochs to ensure completion within the time limit while establishing a strong baseline.
    # The library handles data caching and GPU utilization automatically.
    try:
        train_model(debug=False, epochs=5)
    except Exception as e:
        print(f"Critical Error during training: {e}")
        sys.exit(1)

    print("\n=== Step 2: Validation & Metric Calculation ===")
    # Load validation metadata
    if not os.path.exists(VAL_CSV):
        print(f"Error: Validation metadata not found at {VAL_CSV}")
        sys.exit(1)

    df_val = pd.read_csv(VAL_CSV, keep_default_na=False)

    # Load the best model checkpoint saved during training
    model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        sys.exit(1)

    model = load_model(model_path)

    # Run inference on the validation set
    # predict_dataset returns a map of {slice_id: probability_mask}
    print("Running inference on validation set...")
    preds_map = predict_dataset(model, df_val, debug=False)

    # Aggregate predictions into 3D volumes and compute metrics
    print("Reconstructing 3D volumes and computing metrics...")
    groups = df_val.groupby(["case", "day"])

    volume_metrics = []
    dice_scores_all = []
    hd_scores_all = []

    for (case, day), group_df in groups:
        # Extract metadata features for failure analysis
        # We take the first row's attributes as they are constant for the volume
        num_slices = len(group_df)
        width = group_df.iloc[0]["width"]
        height = group_df.iloc[0]["height"]
        spacing_x = group_df.iloc[0]["spacing_x"]
        spacing_y = group_df.iloc[0]["spacing_y"]

        # Process the volume: reconstruct 3D, post-process (CCA), compute Dice/HD
        res = process_volume(group_df, preds_map, is_validation=True)

        if res:
            # res['dice'] and res['hd'] are lists containing scores for each of the 3 classes
            mean_vol_dice = np.mean(res["dice"])
            mean_vol_hd = np.mean(res["hd"])

            dice_scores_all.append(mean_vol_dice)
            hd_scores_all.append(mean_vol_hd)

            # Calculate composite score for this volume
            # Metric: 0.4 * Dice + 0.6 * (1 - Hausdorff)
            # Note: Hausdorff is normalized to 0-1 range in the metric definition
            vol_score = 0.4 * mean_vol_dice + 0.6 * (1.0 - mean_vol_hd)

            # Store data for failure analysis
            volume_metrics.append(
                {
                    "case": case,
                    "day": day,
                    "num_slices": num_slices,
                    "width": width,
                    "height": height,
                    "spacing_x": spacing_x,
                    "spacing_y": spacing_y,
                    "dice": mean_vol_dice,
                    "hd": mean_vol_hd,
                    "score": vol_score,
                    "error": 1.0 - vol_score,  # Error magnitude
                }
            )

    # Compute Global Weighted Score
    final_mean_dice = np.mean(dice_scores_all)
    final_mean_hd = np.mean(hd_scores_all)
    final_score = 0.4 * final_mean_dice + 0.6 * (1.0 - final_mean_hd)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_score:.16f}")
    print(f"  Mean Dice: {final_mean_dice:.6f}")
    print(f"  Mean Hausdorff: {final_mean_hd:.6f}")

    print("\n=== Step 3: Failure Analysis ===")
    # Analyze correlation between error magnitude and input features
    if not volume_metrics:
        print("No validation metrics available for analysis.")
    else:
        analysis_df = pd.DataFrame(volume_metrics)

        # Features to analyze
        features = ["num_slices", "width", "spacing_x"]
        print(f"Analyzing {len(analysis_df)} validation volumes.")
        print("Correlation between Model Error (1 - Score) and Input Features:")

        for feat in features:
            if feat in analysis_df.columns and analysis_df[feat].nunique() > 1:
                # Calculate Pearson correlation
                corr, p_val = pearsonr(analysis_df["error"], analysis_df[feat])
                print(f"  Feature '{feat}': Correlation = {corr:.4f} (p={p_val:.4f})")
            else:
                print(f"  Feature '{feat}': Insufficient variance or missing data.")

    print("\n=== Step 4: Submission Generation ===")
    THRESHOLD = 0.5745916532174952

    if final_score > THRESHOLD:
        print(
            f"Validation score ({final_score:.6f}) exceeds threshold ({THRESHOLD:.6f})."
        )
        print("Generating submission file...")
        generate_submission(debug=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"Validation score ({final_score:.6f}) did not meet threshold ({THRESHOLD:.6f})."
        )
        print("Skipping submission generation.")


if __name__ == "__main__":
    main()
