import os
import pandas as pd
import numpy as np
import warnings
import sys

# Import from the provided library
from library import config
from library import training_pipeline
from library import utils


def run_demo():
    # =========================================================================
    # 1. Configuration Override for Speed
    # =========================================================================
    print("[Demo] Patching configuration for rapid execution...")

    # Reduce Cross-Validation folds
    config.N_FOLDS = 2

    # Reduce complexity of Base Learners
    # Lexical Bagger (RF)
    config.MODEL_LEXICAL_PARAMS["n_estimators"] = 10

    # Community Bagger (RF)
    config.MODEL_COMMUNITY_PARAMS["n_estimators"] = 10

    # Semantic Booster (XGBoost)
    config.MODEL_SEMANTIC_XGB_PARAMS["n_estimators"] = 10
    config.MODEL_SEMANTIC_XGB_PARAMS["early_stopping_rounds"] = 2

    # Semantic Bagger (RF)
    config.MODEL_SEMANTIC_RF_PARAMS["n_estimators"] = 10

    # Reduce feature dimensionality for speed
    config.TEXT_TFIDF_PARAMS["max_features"] = 500
    config.COMMUNITY_TFIDF_PARAMS["max_features"] = 100

    # =========================================================================
    # 2. Pipeline Execution
    # =========================================================================
    print("[Demo] Initializing EnsembleTrainer...")
    trainer = training_pipeline.EnsembleTrainer()

    print("[Demo] Running pipeline (forcing fresh data processing)...")
    # We set load_cached_data=False to verify the FeatureExtractor logic works
    # and to ensure we don't accidentally load heavy cached files from a previous run.
    trainer.run(load_cached_data=False)

    # =========================================================================
    # 3. Verification
    # =========================================================================
    print("[Demo] Verifying submission output...")

    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not generated at {config.SUBMISSION_PATH}"
        )

    df_submission = pd.read_csv(config.SUBMISSION_PATH)

    # Check 1: Dimensions
    # Test set has 1162 rows
    expected_rows = 1162
    if len(df_submission) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} rows in submission, found {len(df_submission)}"
        )

    # Check 2: Columns
    expected_cols = ["request_id", "requester_received_pizza"]
    if list(df_submission.columns) != expected_cols:
        raise AssertionError(
            f"Expected columns {expected_cols}, found {list(df_submission.columns)}"
        )

    # Check 3: Value Ranges
    probs = df_submission["requester_received_pizza"]
    if probs.min() < 0 or probs.max() > 1:
        raise AssertionError("Probabilities out of range [0, 1]")

    # Check 4: ID consistency
    # Load original test metadata to compare IDs
    df_test_meta = pd.read_parquet(config.TEST_PATH)
    test_ids = set(df_test_meta["request_id"])
    sub_ids = set(df_submission["request_id"])

    if test_ids != sub_ids:
        diff = test_ids.symmetric_difference(sub_ids)
        raise AssertionError(f"Mismatch in request_ids. Difference: {diff}")

    print("[Demo] All checks passed successfully!")
    print(f"[Demo] Sample predictions:\n{df_submission.head()}")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Ensure fully deterministic behavior
    utils.set_seed(42)

    try:
        run_demo()
    except Exception as e:
        print(f"\n[Demo] FAILED: {e}")
        sys.exit(1)
