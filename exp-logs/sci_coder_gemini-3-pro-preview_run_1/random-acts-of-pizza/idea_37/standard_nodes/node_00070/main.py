import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import roc_auc_score

# Import from the provided library files
from library.config import (
    ENSEMBLE_WEIGHTS,
    RANDOM_STATE,
    VAL_PATH,
    TEST_PATH,
    SUBMISSION_PATH,
    TARGET_COL,
    ID_COL,
)
from library.utils import set_seed
from library.model_rf import run_rf_pipeline
from library.model_mlp import run_mlp_pipeline


def perform_failure_analysis(df_val, y_true, y_pred):
    """
    Calculates correlation between prediction error and input features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate absolute error
    error = np.abs(y_true - y_pred)
    df_analysis = df_val.copy()
    df_analysis["error_magnitude"] = error

    # Select numerical columns for correlation
    # We exclude the target and the error column itself
    numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in [TARGET_COL, "error_magnitude"]]

    correlations = []
    for col in numeric_cols:
        # Handle potential NaNs by filling with median for correlation check
        series = df_analysis[col].fillna(df_analysis[col].median())

        # Skip constant columns
        if series.std() == 0:
            continue

        corr = series.corr(df_analysis["error_magnitude"])
        correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top features associated with prediction error (Pearson Correlation):")
    for name, val in correlations[:10]:
        print(f"{name:<50}: {val:.4f}")


def main():
    # 1. Setup
    set_seed(RANDOM_STATE)
    print("Starting Hybrid Ensemble Pipeline...")

    # 2. Run Random Forest Stream
    # This handles feature loading, training, and inference internally
    print("\n--- Stream A: Dispersion-Normalized Random Forest ---")
    rf_results = run_rf_pipeline(load_cached_data=True)

    # 3. Run MLP Stream
    # This handles feature loading, training (with early stopping), and inference
    print("\n--- Stream B: Centroid-Augmented Dual-Attention MLP ---")
    mlp_results = run_mlp_pipeline(load_cached_data=True)

    # 4. Ensemble and Validation
    print("\n--- Ensemble Evaluation ---")

    # Load ground truth for validation
    df_val = pd.read_csv(VAL_PATH)
    y_val = df_val[TARGET_COL].astype(int).values

    # Retrieve predictions
    rf_val_preds = rf_results["val_preds"]
    mlp_val_preds = mlp_results["val_preds"]

    # Ensure predictions are valid
    if rf_val_preds is None or mlp_val_preds is None:
        print("Error: One or more models failed to generate validation predictions.")
        sys.exit(1)

    # Weighted Ensemble
    w_rf = ENSEMBLE_WEIGHTS["rf"]
    w_mlp = ENSEMBLE_WEIGHTS["mlp"]

    val_preds_ens = (w_rf * rf_val_preds) + (w_mlp * mlp_val_preds)

    # Calculate Metric
    final_auc = roc_auc_score(y_val, val_preds_ens)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    perform_failure_analysis(df_val, y_val, val_preds_ens)

    # 6. Submission Generation
    threshold = 0.7056961514236341

    if final_auc > threshold:
        print(
            f"\nValidation metric ({final_auc}) exceeds threshold ({threshold}). Generating submission..."
        )

        # Retrieve Test Predictions
        rf_test_preds = rf_results["test_preds"]
        mlp_test_preds = mlp_results["test_preds"]

        # Weighted Ensemble for Test
        test_preds_ens = (w_rf * rf_test_preds) + (w_mlp * mlp_test_preds)

        # Load Test Metadata to get IDs
        df_test = pd.read_csv(TEST_PATH)

        # Create Submission DataFrame
        submission = pd.DataFrame({ID_COL: df_test[ID_COL], TARGET_COL: test_preds_ens})

        # Save
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({final_auc}) does not exceed threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()
