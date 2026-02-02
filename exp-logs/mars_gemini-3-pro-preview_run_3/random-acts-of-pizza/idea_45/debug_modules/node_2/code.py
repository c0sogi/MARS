import os
import shutil
import pandas as pd
import numpy as np
import warnings
import torch

# Import from the provided library
from library.config import Config
from library.trainer import Trainer
from library.utils import set_seed

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def configure_demo_settings():
    """
    Overrides default Config parameters to ensure the demo runs quickly
    and uses a dedicated working directory.
    """
    print("Configuring demo settings...")

    # 1. Set a unique cache directory for this demo run
    Config.CACHE_DIR = "./working/demo_execution/cache"
    Config.SUBMISSION_DIR = "./working/demo_execution/submission"

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Reduce Cross-Validation Folds
    Config.N_FOLDS = 2

    # 3. Reduce Model Complexity / Iterations for Speed
    # Random Forests
    Config.LEXICAL_RF_PARAMS["n_estimators"] = 10
    Config.COMMUNITY_RF_PARAMS["n_estimators"] = 10
    Config.SEMANTIC_RF_PARAMS["n_estimators"] = 10

    # Gradient Boosting (XGBoost)
    Config.SEMANTIC_XGB_PARAMS["n_estimators"] = 10
    Config.SEMANTIC_XGB_PARAMS["n_jobs"] = (
        1  # Reduce parallelism overhead for small data
    )

    # Gradient Boosting (LightGBM)
    Config.SEMANTIC_LGBM_PARAMS["n_estimators"] = 10
    Config.TEMPORAL_LGBM_PARAMS["n_estimators"] = 10

    # Logistic Regression
    Config.METADATA_LR_PARAMS["max_iter"] = 100

    print("Configuration updated for rapid execution.")


def validate_submission(limit):
    """
    Validates the generated submission file.
    """
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df = pd.read_csv(submission_path)

    # Check dimensions
    if len(df) != limit:
        raise AssertionError(f"Expected {limit} rows in submission, found {len(df)}")

    # Check columns
    expected_cols = [Config.ID_COL, Config.TARGET_COL]
    if not all(col in df.columns for col in expected_cols):
        raise AssertionError(
            f"Submission missing required columns. Found: {df.columns}"
        )

    # Check probability range
    probs = df[Config.TARGET_COL]
    if probs.min() < 0 or probs.max() > 1:
        raise AssertionError("Predicted probabilities are out of range [0, 1]")

    print(f"Submission validation passed: {len(df)} rows, valid probabilities.")


def main():
    # Set global seed for reproducibility
    set_seed(42)

    # Apply demo-specific configuration
    configure_demo_settings()

    # Define a small limit for the dataset to ensure speed
    DEMO_LIMIT = 50

    print(f"\n{'='*40}")
    print(f"Starting End-to-End Pipeline Demo")
    print(f"Sample Limit: {DEMO_LIMIT}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*40}\n")

    # Initialize the Trainer
    trainer = Trainer()

    # Execute the pipeline
    # load_cached_data=False forces the pipeline to process raw data and generate features
    # instead of looking for pre-existing cache files which might not match our limit.
    try:
        trainer.run(load_cached_data=False, limit=DEMO_LIMIT)
    except Exception as e:
        print(f"\nPipeline execution failed with error: {e}")
        raise e

    print(f"\n{'='*40}")
    print("Validating Results")
    print(f"{'='*40}\n")

    # Validate the output
    validate_submission(DEMO_LIMIT)

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    main()
