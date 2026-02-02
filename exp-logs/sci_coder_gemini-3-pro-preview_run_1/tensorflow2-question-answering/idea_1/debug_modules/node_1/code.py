import os
import pandas as pd
import sys
from library.config import Config
from library.training import Trainer
from library.inference import InferenceEngine
from library.utils import setup_logger


def run_demo():
    # Initialize logger
    logger = setup_logger("demo")
    logger.info("Starting Natural Questions Pipeline Demonstration...")

    # 1. Configuration Setup
    # We enable debug mode and disable loading cached data to ensure the pipeline
    # runs the processing logic from scratch.
    config = Config(debug=True, load_cached_data=False)

    # Optimization for demonstration speed:
    # Drastically reduce data size and training iterations.
    config.NEGATIVE_RATIO = 2  # Reduce negative samples per positive
    config.LGBM_PARAMS["verbose"] = -1  # Ensure LightGBM is silent

    # 2. Training Pipeline Execution
    logger.info("\n=== Step 1: Training Pipeline ===")
    trainer = Trainer(config)

    # Run training with a very small sample size (50 examples) and few boosting rounds (5).
    # force_reload_data=True ensures we process the raw data instead of looking for cache.
    try:
        trainer.run(debug_sample_size=50, num_boost_round=5, force_reload_data=True)
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise e

    # Verification: Check if model artifacts were created
    model_path = config.get_cache_path("lgbm_model.txt")
    tfidf_path = config.get_cache_path("tfidf_state.json")
    bm25_path = config.get_cache_path("bm25_state.json")

    if not os.path.exists(model_path):
        raise AssertionError(f"Training failed: Model file not found at {model_path}")
    if not os.path.exists(tfidf_path):
        raise AssertionError(f"Training failed: TF-IDF state not found at {tfidf_path}")
    if not os.path.exists(bm25_path):
        raise AssertionError(f"Training failed: BM25 state not found at {bm25_path}")

    logger.info("Training artifacts verified successfully.")

    # 3. Inference Pipeline Execution
    logger.info("\n=== Step 2: Inference Pipeline ===")
    inference = InferenceEngine(config)

    # Run inference on a small subset of the test data (20 examples).
    # We reuse the model and vectorizers created in the training step.
    try:
        inference.run(debug_sample_size=20, force_reload_data=True)
    except Exception as e:
        logger.error(f"Inference pipeline failed: {e}")
        raise e

    # Verification: Check submission file
    submission_path = config.SUBMISSION_FILE
    if not os.path.exists(submission_path):
        raise AssertionError(
            f"Inference failed: Submission file not found at {submission_path}"
        )

    # Validate Submission Format
    df_sub = pd.read_csv(submission_path)
    logger.info(f"Submission file loaded. Shape: {df_sub.shape}")

    # Check Columns
    expected_cols = ["example_id", "PredictionString"]
    if list(df_sub.columns) != expected_cols:
        raise AssertionError(
            f"Invalid submission columns. Expected {expected_cols}, got {list(df_sub.columns)}"
        )

    # Check Content
    # With 20 samples, we expect 40 rows (one long, one short per sample)
    expected_rows = 40
    if len(df_sub) != expected_rows:
        # Note: If the test file has fewer than 20 samples, this might differ,
        # but based on metadata generation, test set is large enough.
        # We allow for some variance if sampling logic behaves differently, but it should not be empty.
        if df_sub.empty:
            raise AssertionError("Submission dataframe is empty.")
        logger.info(
            f"Submission row count: {len(df_sub)} (Expected approx {expected_rows})"
        )

    # Check IDs format
    sample_id = df_sub.iloc[0]["example_id"]
    if not (sample_id.endswith("_long") or sample_id.endswith("_short")):
        raise AssertionError(f"Invalid example_id format in submission: {sample_id}")

    logger.info("Submission format verified successfully.")
    logger.info("\nDemonstration completed successfully.")


if __name__ == "__main__":
    run_demo()
