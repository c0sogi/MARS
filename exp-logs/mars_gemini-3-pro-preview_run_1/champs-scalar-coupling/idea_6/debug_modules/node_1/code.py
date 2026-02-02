import os
import sys
import numpy as np
import pandas as pd
import warnings
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library import config, data_loader, feature_engineering, model, utils


def run_pipeline_demo():
    print("Starting Pipeline Demo...")

    # =========================================================================
    # 1. CONFIGURATION OVERRIDES
    # =========================================================================
    # Redirect working directory to a demo folder to avoid conflicts
    DEMO_DIR = os.path.join("working", "demo_execution")
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Update config paths
    config.WORKING_DIR = DEMO_DIR
    config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Optimize XGBoost parameters for speed (Demo Mode)
    print("Configuring model parameters for speed...")
    config.BASE_XGB_PARAMS["n_estimators"] = 10
    config.BASE_XGB_PARAMS["max_depth"] = 4
    config.BASE_XGB_PARAMS["learning_rate"] = 0.1
    config.BASE_XGB_PARAMS["verbosity"] = 0
    config.EARLY_STOPPING_ROUNDS = 5
    config.VERBOSE_EVAL = False  # Suppress XGBoost logs

    # Apply updates to type-specific params
    for key in config.TYPE_SPECIFIC_PARAMS:
        config.TYPE_SPECIFIC_PARAMS[key].update(config.BASE_XGB_PARAMS)

    # Set random seeds for reproducibility
    np.random.seed(config.RANDOM_STATE)

    # =========================================================================
    # 2. DATA LOADING & SAMPLING
    # =========================================================================
    print("Loading and sampling metadata...")

    # Load metadata
    df_train_full = data_loader.load_metadata("train")
    df_val_full = data_loader.load_metadata("val")
    df_test_full = data_loader.load_metadata("test")

    # Sample data (2000 rows each) to ensure the script completes quickly
    SAMPLE_SIZE = 2000
    df_train = df_train_full.head(SAMPLE_SIZE).copy()
    df_val = df_val_full.head(SAMPLE_SIZE).copy()
    df_test = df_test_full.head(SAMPLE_SIZE).copy()

    print(
        f"Sampled Data Shapes - Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}"
    )

    # =========================================================================
    # 3. FEATURE ENGINEERING
    # =========================================================================
    print("Executing Feature Engineering Pipeline...")

    # Note: The first call will build the global molecular graph and cache it.
    # Subsequent calls will load the graph from cache (within the demo dir).

    print("  -> Processing Training Set...")
    feat_train = feature_engineering.generate_hierarchical_features(
        df_train, "train_demo", load_cached_data=True
    )

    print("  -> Processing Validation Set...")
    feat_val = feature_engineering.generate_hierarchical_features(
        df_val, "val_demo", load_cached_data=True
    )

    print("  -> Processing Test Set...")
    feat_test = feature_engineering.generate_hierarchical_features(
        df_test, "test_demo", load_cached_data=True
    )

    # Verification of features
    expected_cols = ["dist", "dist_inv", "atom_0_L1_count_C"]
    for col in expected_cols:
        if col not in feat_train.columns:
            raise AssertionError(
                f"Expected feature '{col}' missing from generated features."
            )

    if feat_train.shape[0] != SAMPLE_SIZE:
        raise AssertionError(
            f"Feature generation resulted in {feat_train.shape[0]} rows, expected {SAMPLE_SIZE}."
        )

    # =========================================================================
    # 4. MODEL TRAINING
    # =========================================================================
    print("Training Stratified Ensemble...")

    ensemble = model.StratifiedEnsemble()
    ensemble.fit(feat_train, feat_val)

    # =========================================================================
    # 5. INFERENCE & EVALUATION
    # =========================================================================
    print("Running Inference on Validation Set...")
    val_preds = ensemble.predict(feat_val)

    # Calculate Metric
    # We filter for non-NaN predictions (in case sampling missed some coupling types entirely)
    valid_idx = ~val_preds.isna()
    if valid_idx.sum() > 0:
        metric_score, type_metrics = utils.calculate_log_mae(
            feat_val[valid_idx], val_preds[valid_idx]
        )
        print(f"Validation Log MAE: {metric_score:.5f}")
        print("Type-specific metrics:", type_metrics)
    else:
        print(
            "Warning: No valid predictions generated for validation set (likely due to aggressive sampling)."
        )

    print("Running Inference on Test Set...")
    test_preds = ensemble.predict(feat_test)

    # =========================================================================
    # 6. SUBMISSION
    # =========================================================================
    print("Generating Submission...")

    # Fill NaNs with 0.0 (safe fallback for types not present in training sample)
    test_preds_filled = test_preds.fillna(0.0)

    utils.format_submission(feat_test["id"], test_preds_filled)

    # =========================================================================
    # 7. FINAL VERIFICATION
    # =========================================================================
    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not created.")

    submission_df = pd.read_csv(config.SUBMISSION_PATH)

    if submission_df.shape != (SAMPLE_SIZE, 2):
        raise AssertionError(
            f"Submission shape mismatch. Expected ({SAMPLE_SIZE}, 2), got {submission_df.shape}"
        )

    if submission_df.isnull().any().any():
        raise AssertionError("Submission contains NaN values.")

    print(f"Demo completed successfully. Output stored in {config.WORKING_DIR}")


if __name__ == "__main__":
    run_pipeline_demo()
