import os
import sys
import pandas as pd
import numpy as np
import shutil
from library.config import Config
from library.utils import set_seed, setup_logger
from library.model_engine import DualStreamModel

# =============================================================================
# Configuration & Setup
# =============================================================================


def setup_demo_environment():
    """
    Prepares the environment for a fast demonstration run.
    - Sets random seeds.
    - Redirects working directories to avoid conflicts.
    - Reduces model complexity (n_estimators) for speed.
    """
    # 1. Set Seed
    set_seed(42)

    # 2. Configure Logger
    logger = setup_logger("DemoRun")
    logger.info("Setting up demo environment...")

    # 3. Create a specific working directory for this demo
    demo_working_dir = "./working/demo_run"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Update Config to use this directory
    Config.WORKING_DIR = demo_working_dir

    # 4. Override XGBoost Parameters for Speed
    # We use a very small number of estimators to ensure training finishes in seconds.
    # We also switch to 'hist' to avoid GPU overhead for tiny datasets if necessary,
    # though 'gpu_hist' is generally fine.

    demo_xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",  # Use CPU hist for tiny demo data to avoid overhead
        "learning_rate": 0.05,
        "max_depth": 3,
        "n_estimators": 10,  # drastically reduced from 5000
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": 4,
        "verbosity": 0,
    }

    Config.XGB_PARAMS_STREAM_A = demo_xgb_params.copy()
    Config.XGB_PARAMS_STREAM_B = demo_xgb_params.copy()

    # Update Early Stopping to match
    Config.EARLY_STOPPING_ROUNDS = 5

    return logger


def create_mini_datasets(logger):
    """
    Subsamples the metadata CSVs to create tiny training/validation/test sets.
    Updates Config paths to point to these new files.
    """
    logger.info("Creating mini datasets for rapid demonstration...")

    # Define paths for mini datasets
    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test.csv")

    # Helper to sample and save
    def sample_and_save(src_path, dest_path, n_samples=500):
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source file not found: {src_path}")

        df = pd.read_csv(src_path)

        # Ensure we have enough samples, otherwise take all
        n = min(len(df), n_samples)

        # We want to ensure we have both positive and negative classes if possible
        if "contact" in df.columns:
            # Stratified sample if possible, or just random
            df_sample = df.sample(n=n, random_state=42)
        else:
            df_sample = df.sample(n=n, random_state=42)

        df_sample.to_csv(dest_path, index=False)
        logger.info(f"Created {dest_path} with {len(df_sample)} rows.")
        return dest_path

    # Create mini datasets
    # We use slightly larger samples to ensure Stream A (Player-Player) and Stream B (Player-Ground)
    # both get data after filtering in FeatureBuilder.
    Config.TRAIN_META_PATH = sample_and_save(
        Config.TRAIN_META_PATH, mini_train_path, n_samples=2000
    )
    Config.VAL_META_PATH = sample_and_save(
        Config.VAL_META_PATH, mini_val_path, n_samples=1000
    )

    # For test, we also subsample. Note: The final submission generation code fills missing
    # predictions with 0, so this is safe for demonstration purposes.
    Config.TEST_META_PATH = sample_and_save(
        Config.TEST_META_PATH, mini_test_path, n_samples=1000
    )


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # 1. Setup
    logger = setup_demo_environment()

    # 2. Prepare Data
    create_mini_datasets(logger)

    # 3. Instantiate Model Engine
    logger.info("Initializing DualStreamModel...")
    model = DualStreamModel()

    # 4. Train Models
    # This will trigger FeatureBuilder -> DataLoader -> Caching -> XGBoost Training
    logger.info("Starting Training Pipeline...")
    model.train()

    # Verification: Check if models are trained
    if model.model_a is None:
        raise AssertionError("Stream A model failed to train (model_a is None).")
    if model.model_b is None:
        raise AssertionError("Stream B model failed to train (model_b is None).")
    logger.info("Models trained successfully.")

    # 5. Optimize Thresholds
    logger.info("Optimizing Thresholds...")
    model.optimize_thresholds()

    # Verification: Check thresholds
    if not (0.0 < model.best_threshold_a < 1.0):
        raise AssertionError(f"Invalid threshold A: {model.best_threshold_a}")
    if not (0.0 < model.best_threshold_b < 1.0):
        raise AssertionError(f"Invalid threshold B: {model.best_threshold_b}")
    logger.info(
        f"Thresholds optimized: A={model.best_threshold_a:.2f}, B={model.best_threshold_b:.2f}"
    )

    # 6. Generate Submission
    logger.info("Generating Submission...")
    model.generate_submission()

    # 7. Validate Output
    submission_path = Config.OUTPUT_SUBMISSION_PATH
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file was not created at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    df_sample = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Check row count
    if len(df_sub) != len(df_sample):
        raise AssertionError(
            f"Submission row count mismatch. Expected {len(df_sample)}, got {len(df_sub)}"
        )

    # Check columns
    expected_cols = ["contact_id", "contact"]
    if not all(col in df_sub.columns for col in expected_cols):
        raise AssertionError(
            f"Submission columns mismatch. Expected {expected_cols}, got {df_sub.columns.tolist()}"
        )

    # Check values are binary
    unique_vals = df_sub["contact"].unique()
    if not all(v in [0, 1] for v in unique_vals):
        raise AssertionError(f"Submission contains non-binary values: {unique_vals}")

    logger.info("Demo run completed successfully. All validations passed.")
