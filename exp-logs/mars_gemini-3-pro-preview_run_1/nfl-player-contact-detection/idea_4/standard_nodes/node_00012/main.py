import pandas as pd
import numpy as np
import gc
import sys
import os

# Import from the provided library files
from library.config import Config
from library.feature_engineering import FeatureProcessor
from library.models import LGBMWrapper, XGBWrapper
from library.inference import predict_and_submit
from library.train import optimize_threshold


def main():
    # =========================================================================
    # 1. Configuration Adjustments for Fast Baseline
    # =========================================================================
    # Reduce estimators to ensure execution finishes well within 2 hours
    Config.LGBM_PARAMS["n_estimators"] = 800
    Config.XGB_PARAMS["n_estimators"] = 800

    # Ensure reproducibility
    np.random.seed(Config.SEED)

    print("Starting Runfile Execution...")

    # =========================================================================
    # 2. Data Loading and Processing
    # =========================================================================
    processor = FeatureProcessor()

    # Load Training Data
    print("Loading and processing training data...")
    df_train = processor.process_split("train", load_cached_data=True)

    # Limit training samples for speed (Fast Baseline Requirement)
    # Using 500k samples is sufficient for a robust baseline while saving time
    if len(df_train) > 500000:
        print(f"Downsampling training data from {len(df_train)} to 500,000 rows.")
        df_train = df_train.sample(n=500000, random_state=Config.SEED).reset_index(
            drop=True
        )

    # Load Validation Data
    # We use the full validation set for accurate metric reporting
    print("Loading and processing validation data...")
    df_val = processor.process_split("val", load_cached_data=True)

    # Define Feature Columns
    # Exclude metadata and ID columns
    exclude_cols = [
        "contact_id",
        "game_play",
        "step",
        "nfl_player_id_1",
        "nfl_player_id_2",
        "contact",
        "datetime",
        "p1_id",
        "p2_id",
    ]
    feature_cols = [c for c in df_train.columns if c not in exclude_cols]
    target_col = "contact"

    print(f"Training with {len(feature_cols)} features.")

    # Prepare arrays
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    X_val = df_val[feature_cols]
    y_val = df_val[target_col]

    # Clean up to free memory
    del df_train, df_val
    gc.collect()

    # =========================================================================
    # 3. Model Training
    # =========================================================================

    # --- LightGBM ---
    print("\nTraining LightGBM...")
    lgbm = LGBMWrapper()
    lgbm.fit(X_train, y_train, X_val, y_val)
    lgbm.save("lgbm_model.joblib")
    print("Generating LightGBM validation probabilities...")
    lgbm_probs = lgbm.predict_proba(X_val)

    # --- XGBoost ---
    print("\nTraining XGBoost...")
    xgb_mod = XGBWrapper()
    xgb_mod.fit(X_train, y_train, X_val, y_val)
    xgb_mod.save("xgb_model.joblib")
    print("Generating XGBoost validation probabilities...")
    xgb_probs = xgb_mod.predict_proba(X_val)

    # =========================================================================
    # 4. Ensemble and Evaluation
    # =========================================================================
    print("\nEvaluating Ensemble...")
    # Unweighted Average Ensemble
    ensemble_probs = (lgbm_probs + xgb_probs) / 2.0

    # Optimize Threshold
    best_thresh, best_mcc = optimize_threshold(y_val, ensemble_probs)

    # Print Final Metric (Required Format)
    print(f"Final Validation Metric: {best_mcc}")

    # =========================================================================
    # 5. Failure Analysis
    # =========================================================================
    print("\nPerforming Failure Analysis...")

    # Calculate error magnitude
    errors = np.abs(y_val - ensemble_probs)

    # Create a temporary dataframe for correlation analysis
    # We calculate correlation between features and the error magnitude
    analysis_df = X_val.copy()
    analysis_df["error_mag"] = errors

    # Compute correlations
    correlations = analysis_df.corr()["error_mag"].drop("error_mag")

    # Sort by absolute correlation to find most impactful features
    top_correlations = correlations.abs().sort_values(ascending=False).head(10)

    print("Top 10 Features Correlated with Error Magnitude:")
    for feat in top_correlations.index:
        corr_val = correlations[feat]
        print(f"{feat}: {corr_val}")

    # Clean up
    del analysis_df, X_train, X_val, y_train, y_val
    gc.collect()

    # =========================================================================
    # 6. Submission
    # =========================================================================
    TARGET_SCORE = 0.5979349867102601

    if best_mcc > TARGET_SCORE:
        print(
            f"\nValidation score ({best_mcc}) exceeds target ({TARGET_SCORE}). Generating submission..."
        )
        predict_and_submit(threshold=best_thresh, load_cached_data=True)
    else:
        print(
            f"\nValidation score ({best_mcc}) did not exceed target ({TARGET_SCORE}). Submission skipped."
        )


if __name__ == "__main__":
    main()
