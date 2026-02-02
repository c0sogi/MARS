import os
import sys
import pandas as pd
import numpy as np
import cv2
import torch
from scipy.stats import spearmanr

# Import from the provided library
from library.config import Config
from library.train import run_training
from library.inference import predict
from library.utils import seed_everything, quadratic_weighted_kappa
from library.dataset import load_dataframe


def main():
    # 1. Setup and Configuration
    seed_everything(Config.seed)

    # Override Config for fast baseline execution within time limits
    # 5 epochs * 5 folds * 2 models is well within 2 hours on A100
    Config.epochs = 5
    Config.n_folds = 5
    Config.debug = False

    print("=== Configuration ===")
    print(f"Device: {Config.device}")
    print(f"Epochs: {Config.epochs}")
    print(f"Folds: {Config.n_folds}")
    print(f"Debug Mode: {Config.debug}")
    print(f"Models: {Config.model_archs}")
    print("=====================")

    # 2. Training Loop
    # We iterate manually to pass the explicit 'epochs' argument
    for arch in Config.model_archs:
        run_training(model_name=arch, epochs=Config.epochs)

    # 3. Validation Inference & Metric Calculation
    print("\n=== Running Validation Inference ===")

    # Temporarily point test_csv_path to val_csv_path to reuse the predict function
    original_test_path = Config.test_csv_path
    Config.test_csv_path = Config.val_csv_path

    val_preds_path = os.path.join(Config.working_dir, "val_predictions.csv")

    try:
        # Generate predictions on validation set
        # This uses the ensemble of all trained models
        val_submission = predict(load_cached_data=True, output_path=val_preds_path)

        # Load Ground Truth
        val_df = load_dataframe(Config.val_csv_path, "val_df_ground_truth")

        # Merge predictions with ground truth
        # Ensure alignment by id_code
        val_merged = pd.merge(
            val_df, val_submission, on="id_code", suffixes=("_true", "_pred")
        )

        y_true = val_merged["diagnosis_true"].values
        y_pred = val_merged["diagnosis_pred"].values

        # Calculate Metric
        val_kappa = quadratic_weighted_kappa(y_true, y_pred)

        # REQUIRED OUTPUT FORMAT
        print(f"Final Validation Metric: {val_kappa}")

    finally:
        # Restore configuration
        Config.test_csv_path = original_test_path

    # 4. Failure Analysis
    print("\n=== Failure Analysis ===")

    # Calculate Error Magnitude
    val_merged["error"] = (
        val_merged["diagnosis_true"] - val_merged["diagnosis_pred"]
    ).abs()

    # Extract Metadata Features for Correlation Analysis
    # We calculate these on the fly for the validation set
    meta_stats = []

    print("Extracting metadata for failure analysis...")
    for idx, row in val_merged.iterrows():
        full_path = os.path.join(Config.input_dir, row["file_path"])
        try:
            # File size
            file_size = os.path.getsize(full_path)

            # Image dimensions (read header only if possible, but cv2 reads full)
            img = cv2.imread(full_path)
            if img is not None:
                h, w = img.shape[:2]
                aspect_ratio = w / h if h > 0 else 0
            else:
                h, w, aspect_ratio = 0, 0, 0

            meta_stats.append(
                {
                    "file_size": file_size,
                    "width": w,
                    "height": h,
                    "aspect_ratio": aspect_ratio,
                }
            )
        except Exception:
            meta_stats.append(
                {"file_size": 0, "width": 0, "height": 0, "aspect_ratio": 0}
            )

    df_stats = pd.DataFrame(meta_stats)
    val_analysis = pd.concat([val_merged.reset_index(drop=True), df_stats], axis=1)

    # Calculate Correlations
    features_to_check = ["file_size", "width", "height", "aspect_ratio"]
    print("\nSpearman Correlation with Prediction Error:")
    for feat in features_to_check:
        if val_analysis[feat].std() > 0:
            corr, _ = spearmanr(val_analysis["error"], val_analysis[feat])
            print(f"{feat}: {corr:.4f}")
        else:
            print(f"{feat}: NaN (No variance)")

    # 5. Submission Generation
    # Conditional execution based on metric threshold
    THRESHOLD = 0.9207435978935975

    if val_kappa > THRESHOLD:
        print(
            f"\nValidation metric ({val_kappa}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        submission_path = "./submission/submission.csv"
        predict(load_cached_data=True, output_path=submission_path)
    else:
        print(
            f"\nValidation metric ({val_kappa}) did not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()
