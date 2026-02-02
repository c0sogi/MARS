import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Import from the provided library files
from library.config import Config
from library.data import build_dataset, get_feature_target_split
from library.model import DualXGBoostRegressor


def calculate_rmsle(y_true, y_pred):
    """
    Calculates Root Mean Squared Logarithmic Error.
    y_true and y_pred should be in original scale.
    """
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


def perform_failure_analysis(X_val, y_val, y_pred):
    """
    Correlates prediction errors with input features to identify failure modes.
    """
    print("\n--- Failure Analysis ---")

    # Calculate absolute errors
    errors = np.abs(y_val - y_pred)

    # Combine errors into a single DataFrame for correlation analysis
    error_df = pd.DataFrame(
        {
            "error_formation": errors["formation_energy_ev_natom"],
            "error_bandgap": errors["bandgap_energy_ev"],
        },
        index=X_val.index,
    )

    # Concatenate features and errors
    analysis_df = pd.concat([X_val, error_df], axis=1)

    # Compute correlations
    # We only care about correlation of features with the error columns
    # Select numeric columns only
    numeric_df = analysis_df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()

    for target in ["error_formation", "error_bandgap"]:
        print(f"\nTop feature correlations with {target}:")
        # Get correlations with the specific error target, sort by absolute value
        target_corrs = corr_matrix[target].drop(
            ["error_formation", "error_bandgap"], errors="ignore"
        )
        sorted_corrs = target_corrs.abs().sort_values(ascending=False)

        # Print top 5
        for feature, val in sorted_corrs.head(5).items():
            # Get the sign from the original correlation
            sign = target_corrs[feature]
            print(f"  {feature}: {sign:.4f}")


def main():
    # Set random seeds for reproducibility
    np.random.seed(Config.RANDOM_SEED)

    print("Initializing CR-LEM Pipeline...")

    # 1. Load Data
    # We use load_cached_data=True to speed up if run multiple times,
    # but the first run will compute features.
    # debug=False to use full dataset for best performance to beat threshold.
    print("Loading datasets...")
    train_df = build_dataset("train", load_cached_data=True, debug=False)
    val_df = build_dataset("val", load_cached_data=True, debug=False)
    test_df = build_dataset("test", load_cached_data=True, debug=False)

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # 2. Initialize and Train Model
    print("\nInitializing Dual XGBoost Regressor...")
    # Using default params defined in Config (optimized for this task)
    model = DualXGBoostRegressor()

    print("Training models...")
    # Fit models
    model.fit(train_df, val_df, early_stopping_rounds=100)

    # 3. Validation and Metric Calculation
    print("\nRunning Validation...")
    # Predict on validation set (returns original scale)
    val_preds_df = model.predict(val_df)

    # Extract true values
    _, y_val_true = get_feature_target_split(val_df)

    # Rename columns to match submission format for consistency (Cite debug_lesson_3)
    y_val_true = y_val_true.rename(
        columns={
            "target_formation": "formation_energy_ev_natom",
            "target_bandgap": "bandgap_energy_ev",
        }
    )

    # Ensure indices align
    y_val_true = y_val_true.loc[val_preds_df.index]

    # Calculate RMSLE for each target
    rmsle_formation = calculate_rmsle(
        y_val_true["formation_energy_ev_natom"],
        val_preds_df["formation_energy_ev_natom"],
    )
    rmsle_bandgap = calculate_rmsle(
        y_val_true["bandgap_energy_ev"], val_preds_df["bandgap_energy_ev"]
    )

    # Final metric is column-wise mean
    final_metric = (rmsle_formation + rmsle_bandgap) / 2

    print(f"Validation RMSLE (Formation): {rmsle_formation}")
    print(f"Validation RMSLE (Bandgap): {rmsle_bandgap}")
    print(f"Final Validation Metric: {final_metric}")

    # 4. Failure Analysis
    X_val, _ = get_feature_target_split(val_df)
    # Align X_val with predictions just in case
    X_val = X_val.loc[val_preds_df.index]

    # Prepare prediction dataframe for analysis (align columns)
    y_val_pred_aligned = val_preds_df[
        ["formation_energy_ev_natom", "bandgap_energy_ev"]
    ]
    # Reset index to match y_val_true if needed, or just use values if indices match
    y_val_pred_aligned.index = y_val_true.index

    perform_failure_analysis(X_val, y_val_true, y_val_pred_aligned)

    # 5. Submission
    threshold = 0.056919346405286564
    if final_metric < threshold:
        print(f"\nMetric {final_metric} < {threshold}. Generating submission...")
        submission_df = model.predict(test_df)
        model.save_submission(submission_df)
    else:
        print(f"\nMetric {final_metric} >= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()
