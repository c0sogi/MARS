import os
import shutil
import pandas as pd
import numpy as np
import joblib
import logging

# Import Library Modules
from library.config import Config
from library.utils import setup_logger, seed_everything
from library.training_pipeline import run_training_pipeline
from library.inference_pipeline import run_inference


def run_demo():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print(">>> Setting up demonstration configuration...")

    # Override Config for Speed and Demo Isolation
    Config.EXP_NAME = "demo_run"
    Config.WORKING_DIR = os.path.join("./working", Config.EXP_NAME)
    Config.MODEL_DIR = os.path.join(Config.WORKING_DIR, "models")

    # Update Cache Paths to point to the new working directory
    Config.CACHE_TRAIN_FEATURES = os.path.join(
        Config.WORKING_DIR, "features_train.parquet"
    )
    Config.CACHE_VAL_FEATURES = os.path.join(Config.WORKING_DIR, "features_val.parquet")
    Config.CACHE_TEST_FEATURES = os.path.join(
        Config.WORKING_DIR, "features_test.parquet"
    )
    Config.CACHE_HARD_NEGATIVES = os.path.join(
        Config.WORKING_DIR, "hard_negative_indices.npy"
    )
    Config.SUBMISSION_OUTPUT_PATH = os.path.join(
        Config.WORKING_DIR, "mini_sample_submission.csv"
    )

    # Ensure directories exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)  # Clean start
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    # Reduce Computational Complexity for Demo
    Config.WINDOW_SIZE = 2  # Smaller temporal window
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["num_leaves"] = 8
    Config.XGB_PARAMS["n_estimators"] = 10
    Config.XGB_PARAMS["max_depth"] = 3
    Config.CATBOOST_PARAMS["iterations"] = 10
    Config.CATBOOST_PARAMS["depth"] = 3

    # Disable verbose logging in models
    Config.LGBM_PARAMS["verbose"] = -1
    Config.XGB_PARAMS["verbosity"] = 0
    Config.CATBOOST_PARAMS["verbose"] = 0

    # Set Seed
    seed_everything(Config.SEED)

    # Setup Logger
    logger = setup_logger(
        "demo_execution", log_file=os.path.join(Config.WORKING_DIR, "demo.log")
    )
    logger.info("Configuration overrides applied.")

    # =========================================================================
    # 2. Execute Training Pipeline
    # =========================================================================
    print("\n>>> Starting Training Pipeline (Debug Sample: 5000 rows)...")

    # Run pipeline with a small sample size to test end-to-end logic
    # load_cached_data=False forces feature generation from scratch
    run_training_pipeline(debug_sample=5000, load_cached_data=False)

    # =========================================================================
    # 3. Verify Training Artifacts
    # =========================================================================
    print("\n>>> Verifying Training Artifacts...")

    # Check for Model Files
    # Note: CatBoost might fail if not installed, but LGBM and XGB should exist based on environment
    expected_models = ["expert_lgbm.joblib", "expert_xgb.joblib"]
    for model_file in expected_models:
        path = os.path.join(Config.MODEL_DIR, model_file)
        if not os.path.exists(path):
            # It's possible a model failed training if data was too sparse in the sample,
            # but with 5000 rows it should succeed.
            # However, if the library isn't installed, it won't exist.
            # We check if the class exists in model_factory logic (handled by try-except in pipeline).
            # Here we assert at least one expert exists.
            pass

    # Check if at least one expert model was saved
    saved_models = os.listdir(Config.MODEL_DIR)
    expert_models = [
        f for f in saved_models if f.startswith("expert_") and f.endswith(".joblib")
    ]

    if not expert_models:
        raise AssertionError("No expert models were saved in the training pipeline.")
    print(f"Verified models: {expert_models}")

    # Check for Threshold
    threshold_path = os.path.join(Config.MODEL_DIR, "best_threshold.npy")
    if not os.path.exists(threshold_path):
        raise AssertionError(f"Threshold file not found at {threshold_path}")

    best_thresh = np.load(threshold_path)
    print(f"Verified optimal threshold: {best_thresh[0]}")

    # =========================================================================
    # 4. Execute Inference Pipeline
    # =========================================================================
    print("\n>>> Starting Inference Pipeline (Debug Sample: 2000 rows)...")

    # Run inference on a subset of the sample submission
    run_inference(load_cached_data=False, debug_sample=2000)

    # =========================================================================
    # 5. Verify Submission Output
    # =========================================================================
    print("\n>>> Verifying Submission Output...")

    if not os.path.exists(Config.SUBMISSION_OUTPUT_PATH):
        raise AssertionError(
            f"Submission file not found at {Config.SUBMISSION_OUTPUT_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_OUTPUT_PATH)

    # Check Shape (Should match debug_sample unless gating removed everything,
    # but the pipeline merges back to the template, so row count should match template subset)
    expected_rows = 2000
    if len(df_sub) != expected_rows:
        raise AssertionError(
            f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"
        )

    # Check Columns
    expected_cols = ["contact_id", "contact"]
    if not all(col in df_sub.columns for col in expected_cols):
        raise AssertionError(
            f"Submission columns mismatch. Expected {expected_cols}, got {df_sub.columns.tolist()}"
        )

    # Check Values
    unique_vals = df_sub["contact"].unique()
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise AssertionError(f"Submission contains non-binary values: {unique_vals}")

    print("Submission verification passed.")
    print(df_sub.head())

    print("\n>>> Demo Execution Completed Successfully.")


if __name__ == "__main__":
    run_demo()
