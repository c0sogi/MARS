import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings

# Import from library
from library.config import Config
from library.utils import seed_everything, compute_kendall_tau
from library.data_factory import load_data_factory
from library.sparse_engine import SparseRanker
from library.dense_engine import DenseEngine
from library.trainer import train_sparse_model, train_dense_model
from library.inference_engine import generate_submission, sort_notebooks

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Initializing Demonstration...")

    # ----------------------------------------------------------------
    # 1. Runtime Configuration Override for Speed and Demo purposes
    # ----------------------------------------------------------------
    # We modify the Config class attributes directly to ensure the code
    # runs within the time limit using a very small subset of data.
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Process only 20 notebooks for speed
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.VOCAB_SIZE = 500  # Smaller vocab for faster TF-IDF

    # Redirect output to a specific demo directory
    Config.WORKING_DIR = "./working/demo_output"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Update artifact paths based on new working dir
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.TFIDF_VECTORIZER_PATH = os.path.join(
        Config.WORKING_DIR, "tfidf_vectorizer.joblib"
    )
    Config.RIDGE_MODEL_PATH = os.path.join(Config.WORKING_DIR, "ridge_model.joblib")

    # Setup directories
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print(f"Configuration set. Working directory: {Config.WORKING_DIR}")

    # ----------------------------------------------------------------
    # 2. Data Loading and Processing Verification
    # ----------------------------------------------------------------
    print("\n--- Testing Data Factory ---")
    # Force load from scratch (load_cached_data=False) to verify processing logic
    # This will read from ./input and save parquet files to ./working/demo_output
    df_train, df_val, df_test, test_anchors = load_data_factory(load_cached_data=False)

    print(f"Train rows: {len(df_train)}")
    print(f"Val rows: {len(df_val)}")
    print(f"Test rows: {len(df_test)}")
    print(f"Test Anchors (Notebooks): {len(test_anchors)}")

    # Assertions to verify data integrity
    assert len(df_train) > 0, "Training data should not be empty."
    assert "rank" in df_train.columns, "Training data must have 'rank' column."
    assert "source" in df_train.columns, "Training data must have 'source' column."
    assert len(test_anchors) > 0, "Should have identified test anchors."

    # ----------------------------------------------------------------
    # 3. Sparse Engine Verification
    # ----------------------------------------------------------------
    print("\n--- Testing Sparse Engine ---")
    # Train model using the trainer utility
    sparse_ranker = train_sparse_model(df_train, df_val)

    # Verify artifacts exist
    assert os.path.exists(Config.TFIDF_VECTORIZER_PATH), "TFIDF Vectorizer not saved."
    assert os.path.exists(Config.RIDGE_MODEL_PATH), "Ridge model not saved."

    # Test Prediction on validation set
    preds_sparse = sparse_ranker.predict(df_val)
    assert len(preds_sparse) == len(df_val), "Sparse predictions length mismatch."
    assert isinstance(preds_sparse, np.ndarray), "Predictions should be numpy array."
    print("Sparse Engine test passed.")

    # ----------------------------------------------------------------
    # 4. Dense Engine Verification
    # ----------------------------------------------------------------
    print("\n--- Testing Dense Engine ---")
    # Train model using the trainer utility (1 epoch, small batch)
    dense_engine = train_dense_model(df_train, df_val)

    # Verify artifacts exist
    assert os.path.exists(Config.BEST_MODEL_PATH), "Dense model weights not saved."

    # Test Prediction on validation set
    preds_dense = dense_engine.predict(df_val)
    assert len(preds_dense) == len(df_val), "Dense predictions length mismatch."
    print("Dense Engine test passed.")

    # ----------------------------------------------------------------
    # 5. Inference and Submission Verification
    # ----------------------------------------------------------------
    print("\n--- Testing Inference and Submission ---")
    # generate_submission internally calls predict_hybrid and sort_notebooks.
    # It will load the models saved in the previous steps.
    generate_submission(load_cached_data=True)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission rows: {len(df_sub)}")

    # Validate submission format
    assert (
        "id" in df_sub.columns and "cell_order" in df_sub.columns
    ), "Submission columns missing."
    assert len(df_sub) > 0, "Submission dataframe is empty."

    # Check format of cell_order (space delimited strings)
    sample_order = df_sub.iloc[0]["cell_order"]
    assert isinstance(sample_order, str), "cell_order must be a string."
    assert len(sample_order.split()) > 0, "cell_order must contain cell IDs."
    print("Inference test passed.")

    # ----------------------------------------------------------------
    # 6. Metric Utility Verification
    # ----------------------------------------------------------------
    print("\n--- Testing Metric Utility ---")
    # Create dummy ground truth and prediction to verify Kendall Tau calculation

    # Case 1: Perfect match
    dummy_ids = ["nb_1", "nb_2"]
    dummy_orders = ["a b c", "x y z"]

    df_gt = pd.DataFrame({"id": dummy_ids, "cell_order": dummy_orders})
    df_pred = pd.DataFrame({"id": dummy_ids, "cell_order": dummy_orders})

    score = compute_kendall_tau(df_pred, df_gt)
    print(f"Perfect Match Score: {score}")
    assert np.isclose(score, 1.0), "Perfect match should have score 1.0"

    # Case 2: Complete inversion (worst case)
    # For n=3 (a b c), swaps needed to reverse (c b a) is 3. Total pairs 3*2 = 6.
    # Score = 1 - 4 * (3 / 6) = 1 - 2 = -1.0
    dummy_orders_rev = ["c b a", "z y x"]
    df_pred_rev = pd.DataFrame({"id": dummy_ids, "cell_order": dummy_orders_rev})

    score_rev = compute_kendall_tau(df_pred_rev, df_gt)
    print(f"Reverse Match Score: {score_rev}")
    assert np.isclose(score_rev, -1.0), "Reverse match should have score -1.0"

    print("Metric utility test passed.")

    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
