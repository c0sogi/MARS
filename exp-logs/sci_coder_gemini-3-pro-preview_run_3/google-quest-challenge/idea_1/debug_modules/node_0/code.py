import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import provided library modules
from library.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    TARGET_COLS,
    SEED,
    SUBMISSION_PATH,
    WORKING_DIR,
)
from library.data_loader import load_dataset, prepare_text_pairs, get_targets, get_ids
from library.feature_extractor import EmbeddingPipeline
from library.model import LinearHead
from library.utils import compute_spearman_score, save_submission


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    print("Starting demonstration script...")
    set_seed(SEED)

    # -------------------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------------------
    print("\n1. Loading datasets...")
    train_df = load_dataset(TRAIN_DATA_PATH)
    val_df = load_dataset(VAL_DATA_PATH)
    test_df = load_dataset(TEST_DATA_PATH)

    # OPTIMIZATION: Subset data to ensure the demo runs quickly
    # We use a small sample size (e.g., 50) to demonstrate functionality without long waits.
    SAMPLE_SIZE = 50
    print(f"   Subsetting data to {SAMPLE_SIZE} samples for demonstration speed.")

    train_subset = train_df.head(SAMPLE_SIZE).copy()
    val_subset = val_df.head(SAMPLE_SIZE).copy()
    test_subset = test_df.head(SAMPLE_SIZE).copy()

    # -------------------------------------------------------------------------
    # 2. Prepare Text Data
    # -------------------------------------------------------------------------
    print("\n2. Preparing text pairs...")
    # We use unique split names (e.g., 'train_demo') to avoid cache conflicts with full runs
    train_q, train_a = prepare_text_pairs(
        train_subset, split_name="train_demo", load_cached_data=False
    )
    val_q, val_a = prepare_text_pairs(
        val_subset, split_name="val_demo", load_cached_data=False
    )
    test_q, test_a = prepare_text_pairs(
        test_subset, split_name="test_demo", load_cached_data=False
    )

    # Validation: Check lengths
    assert len(train_q) == SAMPLE_SIZE
    assert len(train_a) == SAMPLE_SIZE
    print(f"   Successfully prepared {len(train_q)} training text pairs.")

    # -------------------------------------------------------------------------
    # 3. Feature Extraction
    # -------------------------------------------------------------------------
    print("\n3. Extracting features (Embeddings)...")
    pipeline = EmbeddingPipeline()

    # Generate features
    # Note: load_cached_data=False forces computation for this demo
    X_train = pipeline.get_features(
        train_q, train_a, split_name="train_demo", load_cached_data=False
    )
    X_val = pipeline.get_features(
        val_q, val_a, split_name="val_demo", load_cached_data=False
    )
    X_test = pipeline.get_features(
        test_q, test_a, split_name="test_demo", load_cached_data=False
    )

    # Validation: Check feature shapes
    # Expected dim: 384 (MiniLM) * 4 (u, v, |u-v|, u*v) = 1536
    expected_dim = 384 * 4
    print(f"   Feature matrix shape: {X_train.shape}")

    if X_train.shape != (SAMPLE_SIZE, expected_dim):
        raise AssertionError(
            f"Expected shape ({SAMPLE_SIZE}, {expected_dim}), got {X_train.shape}"
        )

    # -------------------------------------------------------------------------
    # 4. Prepare Targets
    # -------------------------------------------------------------------------
    print("\n4. Preparing targets...")
    y_train = get_targets(train_subset)
    y_val = get_targets(val_subset)

    # Validation: Check target shapes
    if y_train.shape != (SAMPLE_SIZE, 30):
        raise AssertionError(
            f"Expected target shape ({SAMPLE_SIZE}, 30), got {y_train.shape}"
        )

    # -------------------------------------------------------------------------
    # 5. Model Training
    # -------------------------------------------------------------------------
    print("\n5. Training LinearHead model...")
    model = LinearHead()

    # Fit the model and evaluate on validation subset
    model.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        target_cols=TARGET_COLS,
    )

    # Save and Load check
    model.save("demo_model.joblib")
    model.load("demo_model.joblib")

    # -------------------------------------------------------------------------
    # 6. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n6. Generating predictions and submission...")

    # Predict on test set
    test_preds = model.predict(X_test)

    # Validation: Check prediction range and shape
    if test_preds.shape != (SAMPLE_SIZE, 30):
        raise AssertionError(f"Prediction shape mismatch: {test_preds.shape}")

    if test_preds.min() < 0.0 or test_preds.max() > 1.0:
        raise AssertionError("Predictions contain values outside [0, 1] range.")

    # Get Test IDs
    test_ids = get_ids(test_subset)

    # Save submission
    # We use a temporary path for the demo to avoid overwriting main submission if needed,
    # or just use the config path. Here we use the config path.
    save_submission(test_preds, test_ids, TARGET_COLS, SUBMISSION_PATH)
    print(f"   Submission saved to {SUBMISSION_PATH}")

    # Final Verification of the saved file
    print("\n7. Verifying submission file...")
    sub_df = pd.read_csv(SUBMISSION_PATH)

    # Check rows (header + data)
    if len(sub_df) != SAMPLE_SIZE:
        raise AssertionError(
            f"Submission has {len(sub_df)} rows, expected {SAMPLE_SIZE}."
        )

    # Check columns (qa_id + 30 targets)
    expected_cols = ["qa_id"] + TARGET_COLS
    if list(sub_df.columns) != expected_cols:
        raise AssertionError("Submission columns do not match requirements.")

    print("   Verification successful. Demo completed.")
