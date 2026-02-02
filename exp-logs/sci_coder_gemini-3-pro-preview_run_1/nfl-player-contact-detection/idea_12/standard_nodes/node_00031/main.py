import os
import sys
import numpy as np
import pandas as pd
import gc
import warnings
import joblib

# Import provided library components
from library.config import get_config, WORKING_DIR, SEED, SUBMISSION_PATH, TRAIN_CONFIG
from library.data_utils import load_metadata_and_tracking
from library.feature_engineering import generate_dataset, _process_dataset
from library.models import train_with_mining_curriculum
from library.metrics import calculate_mcc, optimize_threshold

# Suppress warnings and set environment
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
np.random.seed(SEED)


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Data Loading (Gated for Training)
    # -------------------------------------------------------------------------
    # We use the default configuration provided in the library.
    # debug=False ensures we use the full training parameters.
    _, _, _, _ = get_config(debug=False)

    print("Loading Gated Training Data...")
    X_train, y_train, _ = generate_dataset("train", load_cached_data=True)

    print("Loading Gated Validation Data (for Early Stopping)...")
    X_val_gated, y_val_gated, _ = generate_dataset("val", load_cached_data=True)

    # -------------------------------------------------------------------------
    # 2. Model Training
    # -------------------------------------------------------------------------
    # Train the ensemble using the mining curriculum (Scout -> Mine -> Expert)
    # This handles internal validation on X_val_gated for early stopping.
    ensemble = train_with_mining_curriculum(
        X_train, y_train, X_val_gated, y_val_gated, load_cached_data=True
    )

    # Clean up memory
    del X_train, y_train, X_val_gated, y_val_gated
    gc.collect()

    # -------------------------------------------------------------------------
    # 3. Full Validation & Threshold Optimization
    # -------------------------------------------------------------------------
    print("\nProcessing Full Validation Set (Ungated) for Metric Calculation...")
    # Load raw validation data and process without gating to get all rows
    # This ensures the metric reflects the true performance on the full dataset distribution
    df_val_full = load_metadata_and_tracking("val", load_cached_data=True)

    # Process features (is_train=False disables gating)
    # We need to ensure we keep the same columns as the model expects
    df_val_proc, feature_cols = _process_dataset(df_val_full, is_train=False)

    # Prepare X and y
    y_val_full = df_val_proc["contact"].values

    # Select only the feature columns used during training
    X_val_full = df_val_proc[feature_cols]

    # Inference on Full Validation Set
    print("Running Inference on Full Validation Set...")
    val_probs = ensemble.predict_proba(X_val_full)

    # Optimize Threshold on Full Set
    print("Optimizing Threshold on Full Validation Set...")
    best_threshold, best_mcc = optimize_threshold(y_val_full, val_probs, step=0.01)

    print(f"Final Validation Metric: {best_mcc}")

    # -------------------------------------------------------------------------
    # 4. Failure Analysis
    # -------------------------------------------------------------------------
    print("\nPerforming Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(y_val_full - val_probs)

    # Correlate errors with features
    # We'll use a subset if the data is too large to speed up correlation calculation
    sample_size = 100000
    if len(X_val_full) > sample_size:
        indices = np.random.choice(len(X_val_full), sample_size, replace=False)
        X_analyze = X_val_full.iloc[indices]
        errors_analyze = errors[indices]
    else:
        X_analyze = X_val_full
        errors_analyze = errors

    correlations = {}
    for col in X_analyze.columns:
        # Only numeric columns
        if pd.api.types.is_numeric_dtype(X_analyze[col]):
            # Handle potential NaNs or constant columns
            if X_analyze[col].std() > 0:
                # Fill NaNs with 0 for correlation check
                corr = np.corrcoef(X_analyze[col].fillna(0), errors_analyze)[0, 1]
                if not np.isnan(corr):
                    correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for feat, corr in sorted_corr[:5]:
        print(f"  {feat}: {corr:.4f}")

    # -------------------------------------------------------------------------
    # 5. Submission
    # -------------------------------------------------------------------------
    if best_mcc > 0.6782:
        print("\nMetric exceeds threshold. Generating Submission...")

        # Clean memory
        del df_val_proc, X_val_full, y_val_full, val_probs
        gc.collect()

        # Load Test Data
        # generate_dataset('test') handles loading and processing
        X_test, _, test_ids = generate_dataset("test", load_cached_data=True)

        # Inference
        test_probs = ensemble.predict_proba(X_test)

        # Apply Threshold
        test_preds = (test_probs >= best_threshold).astype(int)

        # Create Submission DataFrame
        submission = pd.DataFrame({"contact_id": test_ids, "contact": test_preds})

        # Save
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")

    else:
        print(f"\nMetric {best_mcc} is not greater than 0.6782. Submission skipped.")


if __name__ == "__main__":
    main()
