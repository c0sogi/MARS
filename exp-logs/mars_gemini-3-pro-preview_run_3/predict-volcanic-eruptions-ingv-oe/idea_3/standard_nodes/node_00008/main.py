import os
import sys
import pandas as pd
import numpy as np
import torch

# Import config first
import library.config as config

# --- Configuration Overrides for Fast Baseline ---
# We modify the variables in the imported modules to ensure fast execution
import library.trainer_cnn
import library.trainer_lgbm

# Reduce epochs for CNN to ensure it completes quickly (Fast Baseline)
library.trainer_cnn.EPOCHS = 5
# Reduce max estimators for LightGBM (modifying the dictionary object updates it globally)
config.LGBM_PARAMS["n_estimators"] = 2000

# Import other necessary functions from the provided library
from library.utils import seed_everything, calculate_mae
from library.data_processing import prepare_tabular_dataset, get_spectrogram_loaders
from library.trainer_lgbm import train_lgbm_model, predict_lgbm
from library.trainer_cnn import run_cnn_training, predict_cnn


def main():
    # Set seeds for reproducibility
    seed_everything(config.SEED)

    print("Starting Orchestration Script...")

    # =========================================================================
    # STREAM A: LightGBM (Tabular Features)
    # =========================================================================
    print("\n=== Stream A: LightGBM ===")

    # 1. Prepare Data
    # Load cached data if available to save time and compute
    train_df, val_df, test_df = prepare_tabular_dataset(load_cached_data=True)

    # 2. Train Model
    # train_lgbm_model returns the model and the validation predictions (aligned with val_df)
    lgbm_model, lgbm_val_preds = train_lgbm_model(train_df, val_df)

    # Store predictions in val_df for ensemble alignment
    val_df["lgbm_pred"] = lgbm_val_preds

    # =========================================================================
    # STREAM B: CNN (Spectrograms)
    # =========================================================================
    print("\n=== Stream B: CNN ===")

    # 1. Train Model
    # run_cnn_training handles loading, training loop, early stopping, and saving best model
    cnn_model, cnn_best_mae = run_cnn_training(debug=False)

    # 2. Generate Validation Predictions for Ensemble
    # We need to run inference on the validation set to get predictions aligned by segment_id
    print("Generating CNN validation predictions...")
    _, val_loader, _ = get_spectrogram_loaders(
        batch_size=config.BATCH_SIZE, debug=False
    )

    # predict_cnn returns a DataFrame with segment_id and time_to_eruption (prediction)
    # We pass the validation loader here
    cnn_val_preds_df = predict_cnn(cnn_model, test_loader=val_loader)

    # Rename column to avoid collision during merge
    cnn_val_preds_df = cnn_val_preds_df.rename(columns={"time_to_eruption": "cnn_pred"})

    # =========================================================================
    # ENSEMBLE & EVALUATION
    # =========================================================================
    print("\n=== Ensemble ===")

    # Merge CNN predictions into val_df based on segment_id
    # val_df already has lgbm_pred. We use left join to preserve val_df structure.
    ensemble_df = val_df.merge(cnn_val_preds_df, on="segment_id", how="left")

    # Calculate Weighted Average
    alpha = config.ENSEMBLE_WEIGHT
    ensemble_df["ensemble_pred"] = (alpha * ensemble_df["lgbm_pred"]) + (
        (1 - alpha) * ensemble_df["cnn_pred"]
    )

    # Calculate Final Metric
    y_true = ensemble_df["time_to_eruption"]
    y_pred = ensemble_df["ensemble_pred"]
    final_mae = calculate_mae(y_true, y_pred)

    # REQUIRED OUTPUT: Print the final validation metric
    print(f"Final Validation Metric: {final_mae}")

    # =========================================================================
    # FAILURE ANALYSIS
    # =========================================================================
    print("\n=== Failure Analysis ===")

    # Calculate absolute error for each sample
    ensemble_df["error"] = np.abs(
        ensemble_df["time_to_eruption"] - ensemble_df["ensemble_pred"]
    )

    # Select feature columns (exclude metadata, targets, and predictions)
    exclude_cols = [
        "segment_id",
        "time_to_eruption",
        "file_path",
        "lgbm_pred",
        "cnn_pred",
        "ensemble_pred",
        "error",
    ]
    feature_cols = [c for c in ensemble_df.columns if c not in exclude_cols]

    # Calculate correlation between input features and the error magnitude
    correlations = ensemble_df[feature_cols].corrwith(ensemble_df["error"]).abs()

    print("Top 10 features correlated with prediction error:")
    print(correlations.sort_values(ascending=False).head(10))

    # =========================================================================
    # SUBMISSION
    # =========================================================================
    THRESHOLD = 3398603.6592843872

    if final_mae < THRESHOLD:
        print("\nMetric passed threshold. Generating submission...")

        # 1. LightGBM Inference on Test Set
        lgbm_test_preds = predict_lgbm(lgbm_model, test_df)
        test_df["lgbm_pred"] = lgbm_test_preds

        # 2. CNN Inference on Test Set
        # predict_cnn creates the test loader internally from TEST_META_PATH if not provided
        cnn_test_preds_df = predict_cnn(cnn_model)
        cnn_test_preds_df = cnn_test_preds_df.rename(
            columns={"time_to_eruption": "cnn_pred"}
        )

        # 3. Ensemble Test Predictions
        submission_df = test_df[["segment_id", "lgbm_pred"]].merge(
            cnn_test_preds_df, on="segment_id", how="left"
        )

        submission_df["time_to_eruption"] = (alpha * submission_df["lgbm_pred"]) + (
            (1 - alpha) * submission_df["cnn_pred"]
        )

        # 4. Save Submission
        final_submission = submission_df[["segment_id", "time_to_eruption"]]
        save_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        final_submission.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nMetric {final_mae} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()
