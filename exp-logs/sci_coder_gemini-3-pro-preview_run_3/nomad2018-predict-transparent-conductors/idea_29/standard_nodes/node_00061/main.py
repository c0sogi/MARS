import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Import from provided libraries
import library.config as config
from library.data_manager import build_dataset
from library.model_handler import EnergyPredictor, save_submission

# Set random seeds for reproducibility
np.random.seed(config.NP_SEED)


def calculate_rmsle(y_true, y_pred):
    """
    Calculates Root Mean Squared Logarithmic Error.
    Since models predict in log1p space, this is just RMSE of the log-transformed values.
    However, if y_pred are raw values, we need to log1p them.
    The EnergyPredictor trains on log1p targets.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def perform_failure_analysis(df, y_true_log, y_pred_log, target_name, feature_cols):
    """
    Correlates absolute error with features to find sources of error.
    """
    print(f"\n--- Failure Analysis for {target_name} ---")

    # Calculate absolute error in log space (which corresponds to ratio error in linear space)
    errors = np.abs(y_true_log - y_pred_log)

    # Create a dataframe for correlation analysis
    analysis_df = df[feature_cols].copy()
    analysis_df["error"] = errors

    # Calculate correlation
    correlations = analysis_df.corr()["error"].drop("error")

    # Sort by absolute correlation
    top_correlations = correlations.abs().sort_values(ascending=False).head(10)

    print("Top 10 features correlated with error magnitude:")
    for feature, corr in top_correlations.items():
        print(f"  {feature}: {correlations[feature]:.4f}")


def main():
    print("Starting pipeline...")

    # 1. Load Data
    # build_dataset handles loading metadata and extracting features (cached)
    print("Loading datasets...")
    train_df = build_dataset("train", load_cached_data=True)
    val_df = build_dataset("val", load_cached_data=True)
    test_df = build_dataset("test", load_cached_data=True)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # 2. Initialize and Train Model
    predictor = EnergyPredictor()

    # Train the models
    # The train method inside EnergyPredictor handles log-transformation of targets internally
    # and prints validation metrics during training.
    predictor.train(train_df, val_df)

    # 3. Validation Assessment & Metric Calculation
    print("\nComputing Final Validation Metric...")

    # We need to manually compute the metric to ensure it matches the requirement exactness
    # and to perform failure analysis.
    # We use the internal helper to get X_val with correct columns
    X_val, y_val_log = predictor._prepare_data(val_df, is_training=True)

    # Predict in log space
    val_pred_log_form = predictor.model_formation.predict(X_val)
    val_pred_log_band = predictor.model_bandgap.predict(X_val)

    # Calculate RMSLE for each target
    # Note: y_val_log is a DataFrame with both columns.
    rmsle_form = calculate_rmsle(y_val_log[config.TARGET_COLS[0]], val_pred_log_form)
    rmsle_band = calculate_rmsle(y_val_log[config.TARGET_COLS[1]], val_pred_log_band)

    # Metric is column-wise root mean squared logarithmic error.
    # Usually this implies the mean of the RMSLEs of the columns.
    final_metric = (rmsle_form + rmsle_band) / 2.0

    print(f"Formation Energy RMSLE: {rmsle_form}")
    print(f"Bandgap Energy RMSLE: {rmsle_band}")
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    feature_cols = predictor.feature_cols
    perform_failure_analysis(
        val_df,
        y_val_log[config.TARGET_COLS[0]],
        val_pred_log_form,
        config.TARGET_COLS[0],
        feature_cols,
    )
    perform_failure_analysis(
        val_df,
        y_val_log[config.TARGET_COLS[1]],
        val_pred_log_band,
        config.TARGET_COLS[1],
        feature_cols,
    )

    # 5. Submission Generation
    threshold = 0.05194666210242679
    if final_metric < threshold:
        print(f"\nMetric {final_metric} < {threshold}. Generating submission...")
        submission_df = predictor.predict(test_df)
        save_submission(submission_df)
    else:
        print(f"\nMetric {final_metric} >= {threshold}. Submission generation skipped.")


if __name__ == "__main__":
    main()
