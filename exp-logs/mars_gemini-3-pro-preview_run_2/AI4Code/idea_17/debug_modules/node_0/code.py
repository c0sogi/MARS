import os
import shutil
import pandas as pd
import numpy as np
import warnings
import random

# Import provided library components
from library.config import Config
from library.model_trainer import ModelTrainer
from library.utils import kendall_tau


def run_demo():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    print("=== Setting up Demo Environment ===")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Define a temporary working directory for this run
    DEMO_WORKDIR = "./working/demo_run"
    if os.path.exists(DEMO_WORKDIR):
        shutil.rmtree(DEMO_WORKDIR)
    os.makedirs(DEMO_WORKDIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # 2. Create Mini Metadata (Subsampling for Speed)
    # --------------------------------------------------------------------------
    print("Creating mini-datasets from existing metadata...")

    # Load original metadata
    # We assume these files exist based on the problem description
    orig_train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    orig_val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    orig_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Sample a small subset (e.g., 30 train, 10 val, 10 test)
    # This ensures the pipeline runs in seconds/minutes instead of hours
    mini_train = orig_train_meta.head(30).copy()
    mini_val = orig_val_meta.head(10).copy()
    mini_test = orig_test_meta.head(10).copy()

    # Save mini metadata
    mini_train_path = os.path.join(DEMO_WORKDIR, "mini_train.csv")
    mini_val_path = os.path.join(DEMO_WORKDIR, "mini_val.csv")
    mini_test_path = os.path.join(DEMO_WORKDIR, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    print(f"Mini metadata saved to {DEMO_WORKDIR}")

    # --------------------------------------------------------------------------
    # 3. Override Config for Demo
    # --------------------------------------------------------------------------
    print("Overriding Config parameters for fast execution...")

    # Update paths to point to mini metadata and demo working dir
    Config.WORKING_DIR = DEMO_WORKDIR
    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path
    Config.TEST_METADATA_PATH = mini_test_path

    # Update output paths
    Config.SUBMISSION_DIR = os.path.join(DEMO_WORKDIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce Hyperparameters for speed
    # Text Vectorization
    Config.MD_TFIDF_PARAMS["max_features"] = 50
    Config.CODE_TFIDF_PARAMS["max_features"] = 50

    # Dimensionality Reduction & Clustering
    Config.CODE_SVD_COMPONENTS = 5
    Config.NUM_CODE_CLUSTERS = 2  # Reduced from 5

    # Model Training
    Config.NUM_FOLDS = 2  # Minimum for CV
    Config.LGBM_PARAMS["n_estimators"] = 10  # Very few trees
    Config.LGBM_PARAMS["num_leaves"] = 8
    Config.LGBM_EARLY_STOPPING_ROUNDS = 5
    Config.LGBM_VERBOSE_EVAL = -1  # Silent

    # Re-run setup to ensure directories exist
    Config.setup()

    # --------------------------------------------------------------------------
    # 4. Execute Training Pipeline
    # --------------------------------------------------------------------------
    print("\n=== Executing Training Pipeline ===")

    # Instantiate the Trainer
    trainer = ModelTrainer()

    # Run Training
    # load_cached_data=False forces the pipeline to process the new mini dataset
    ridge_model, lgbm_model = trainer.train(load_cached_data=False)

    print("Training complete.")

    # --------------------------------------------------------------------------
    # 5. Generate Submission (Inference)
    # --------------------------------------------------------------------------
    print("\n=== Generating Submission ===")

    submission_df = trainer.generate_submission(load_cached_data=False)

    print("Submission generation complete.")
    print(submission_df.head())

    # --------------------------------------------------------------------------
    # 6. Validation & Assertions
    # --------------------------------------------------------------------------
    print("\n=== Validating Results ===")

    # Check 1: Submission File Existence
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # Check 2: Submission Dimensions
    expected_rows = len(mini_test)
    if len(submission_df) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} rows in submission, got {len(submission_df)}"
        )

    # Check 3: Submission Columns
    required_cols = {"id", "cell_order"}
    if not required_cols.issubset(submission_df.columns):
        raise AssertionError(
            f"Submission missing required columns: {required_cols - set(submission_df.columns)}"
        )

    print("Pipeline output validation passed.")

    # Check 4: Metric Logic Verification (Unit Test for utils.kendall_tau)
    print("Verifying Kendall Tau metric logic...")

    # Case A: Perfect Match
    gt_perfect = [["a", "b", "c"]]
    pred_perfect = [["a", "b", "c"]]
    score_perfect = kendall_tau(gt_perfect, pred_perfect)
    assert np.isclose(
        score_perfect, 1.0
    ), f"Perfect match should be 1.0, got {score_perfect}"

    # Case B: Complete Inversion
    # n=3, pairs = 3*(2)/2 = 3. Inversion swaps = 3.
    # K = 1 - 4 * (3 / 6) = 1 - 2 = -1.0
    gt_inv = [["a", "b", "c"]]
    pred_inv = [["c", "b", "a"]]
    score_inv = kendall_tau(gt_inv, pred_inv)
    assert np.isclose(
        score_inv, -1.0
    ), f"Inverted match should be -1.0, got {score_inv}"

    print("Metric logic validation passed.")
    print("\nSUCCESS: Demo script executed without errors.")


if __name__ == "__main__":
    run_demo()
