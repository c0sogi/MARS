import os
import sys
import shutil
import numpy as np
import pandas as pd
import warnings
import torch

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.feature_extraction import run_feature_extraction
from library.level0_experts import train_level0_experts
from library.level1_meta import train_meta_learner


def configure_demo_settings():
    """
    Overrides the global Config to run a fast demonstration.
    """
    print("Configuring settings for fast demonstration...")

    # 1. Paths
    # Use a specific working directory for this demo to avoid conflicts
    Config.CACHE_DIR = "./working/demo_execution/cache"
    Config.SUBMISSION_DIR = "./working/demo_execution/submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure clean slate
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    if os.path.exists(Config.SUBMISSION_DIR):
        shutil.rmtree(Config.SUBMISSION_DIR)

    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # 2. Compute / Resources
    # Reduce batch size for speed/memory safety during demo
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 2

    # 3. Model Hyperparameters (Optimize for Speed)
    # Reduce folds
    Config.N_FOLDS = 3

    # Reduce complexity of Level-0 Experts
    Config.PCA_COMPONENTS = 16  # Smaller PCA for small debug dataset

    # Ridge: Fewer alphas
    Config.RIDGE_ALPHAS = [0.1, 1.0, 10.0]

    # SVR: Relaxed cache
    Config.SVR_PARAMS["cache_size"] = 500

    # ExtraTrees: Fewer estimators
    Config.ET_PARAMS["n_estimators"] = 20

    # LightGBM: Fewer estimators, no verbosity
    Config.LGBM_PARAMS["n_estimators"] = 20
    Config.LGBM_PARAMS["num_leaves"] = 15
    Config.LGBM_ES_ROUNDS = 5

    # Level-1 Meta Learner: Fewer iterations
    Config.META_MODEL_PARAMS["max_iter"] = 50

    # 4. Backbones
    # We will use all backbones defined in Config to verify the full stacking logic,
    # but the 'debug=True' flag in the pipeline functions will limit the number of images processed.
    print(f"Active Backbones: {list(Config.BACKBONES.keys())}")


def validate_submission():
    """
    Validates the generated submission file.
    """
    print("\nValidating submission file...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    expected_cols = [Config.ID_COL, Config.TARGET_COL]
    if list(df.columns) != expected_cols:
        raise AssertionError(
            f"Submission columns mismatch. Expected {expected_cols}, got {list(df.columns)}"
        )

    # Check values
    preds = df[Config.TARGET_COL]
    if preds.isnull().any():
        raise AssertionError("Submission contains NaN values.")

    min_val, max_val = preds.min(), preds.max()
    print(f"Prediction Range: [{min_val:.4f}, {max_val:.4f}]")

    if min_val < 0 or max_val > 100:
        raise AssertionError(
            f"Predictions out of expected range [0, 100]. Found [{min_val}, {max_val}]"
        )

    # Check length
    # In debug mode, Level0Experts limits test set to 20 samples.
    # Level1MetaLearner uses the IDs passed from Level0.
    # Therefore, we expect 20 rows.
    expected_len = 20
    if len(df) != expected_len:
        raise AssertionError(
            f"Submission length mismatch. Expected {expected_len} (debug limit), got {len(df)}"
        )

    print("Submission validation passed successfully.")


def main():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set global seed
    set_seed(42)

    # 1. Configure Demo Settings
    configure_demo_settings()

    print("\n" + "=" * 50)
    print("STEP 1: Feature Extraction")
    print("=" * 50)
    # Run feature extraction in debug mode.
    # debug=True limits the dataset to 100 samples per split.
    # load_cached_data=False forces the extraction to run (verifying the model loading and inference).
    run_feature_extraction(debug=True, load_cached_data=False)

    print("\n" + "=" * 50)
    print("STEP 2: Level-0 Expert Training")
    print("=" * 50)
    # Train base models (Ridge, SVR, ET, LGBM) on the extracted features.
    # debug=True limits the training set size further for speed.
    train_level0_experts(debug=True, load_cached_data=False)

    print("\n" + "=" * 50)
    print("STEP 3: Level-1 Meta-Learner Training")
    print("=" * 50)
    # Train the meta-learner on OOF predictions and generate submission.
    train_meta_learner(debug=True, load_cached_data=False)

    print("\n" + "=" * 50)
    print("STEP 4: Validation")
    print("=" * 50)
    validate_submission()

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
