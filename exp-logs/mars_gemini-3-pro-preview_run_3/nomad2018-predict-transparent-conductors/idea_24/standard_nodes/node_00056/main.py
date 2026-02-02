import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# Import from the provided library files
from library.config import RANDOM_SEED, SUBMISSION_PATH
from library.data_manager import DataManager
from library.model_factory import DualModelWrapper, generate_submission


def main():
    # 1. Set Random Seeds for Reproducibility
    np.random.seed(RANDOM_SEED)

    print("Initializing Pipeline...")

    # 2. Data Loading and Feature Engineering
    # The DataManager handles caching, so we set load_cached_data=True to speed up re-runs
    dm = DataManager()
    try:
        (X_train, y_train), (X_val, y_val), (X_test, ids_test) = (
            dm.load_and_process_data(load_cached_data=True)
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. Model Training
    # Initialize the wrapper which handles two XGBoost models (one per target)
    model_wrapper = DualModelWrapper()

    # Train the models
    # Note: y_train and y_val are already log-transformed by DataManager
    model_wrapper.train(X_train, y_train, X_val, y_val)

    # 4. Validation Assessment
    print("\n--- Computing Final Validation Metric ---")

    # Predict on validation set (returns log scale predictions because model was trained on logs)
    # Note: The wrapper's predict() method returns original scale, but for metric calculation
    # which is RMSLE, it is mathematically equivalent to RMSE on the log-transformed data.
    # However, to be precise with the wrapper's interface, let's use the internal models or
    # re-log the output of predict(), or just use the internal predict method if accessible.
    # Looking at DualModelWrapper.evaluate, it predicts in log space. Let's replicate that logic here
    # to ensure we have the exact values used for the metric.

    pred_form_log = model_wrapper.model_formation.predict(X_val)
    pred_band_log = model_wrapper.model_bandgap.predict(X_val)

    # Ground truth (already log transformed)
    true_form_log = y_val["formation_energy_log"]
    true_band_log = y_val["bandgap_energy_log"]

    # Calculate RMSE on log data (which is RMSLE on original data)
    rmsle_form = np.sqrt(mean_squared_error(true_form_log, pred_form_log))
    rmsle_band = np.sqrt(mean_squared_error(true_band_log, pred_band_log))

    # Final Metric: Mean Column-wise RMSLE
    final_metric = (rmsle_form + rmsle_band) / 2

    print(f"Formation Energy RMSLE: {rmsle_form}")
    print(f"Bandgap Energy RMSLE:   {rmsle_band}")
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude per sample (average absolute error across targets in log space)
    error_form = np.abs(pred_form_log - true_form_log)
    error_band = np.abs(pred_band_log - true_band_log)
    mean_abs_error = (error_form + error_band) / 2

    # Correlate error with features
    correlations = []
    for col in X_val.columns:
        # Ensure column is numeric
        if pd.api.types.is_numeric_dtype(X_val[col]):
            # Handle potential constant columns or NaNs if any slipped through
            if X_val[col].std() > 1e-9:
                corr, _ = pearsonr(X_val[col], mean_abs_error)
                if not np.isnan(corr):
                    correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top 10 Features correlated with Prediction Error:")
    for feature, corr in correlations[:10]:
        print(f"{feature:<40}: {corr:.4f}")

    # 6. Submission Generation
    # Threshold defined in task description
    THRESHOLD = 0.055766518324569046

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} is below threshold {THRESHOLD}. Generating submission..."
        )
        generate_submission(model_wrapper, X_test, ids_test)
    else:
        print(
            f"\nMetric {final_metric} is NOT below threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
