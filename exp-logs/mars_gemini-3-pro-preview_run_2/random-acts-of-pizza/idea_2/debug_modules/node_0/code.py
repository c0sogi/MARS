import os
import sys
import numpy as np
import pandas as pd

# Import provided library components
from library.config import SUBMISSION_FILE
from library.data_loader import load_and_preprocess_data
from library.feature_extractor import HybridFeaturePipeline
from library.trainer import Trainer
from library.utils import set_seed


def main():
    # 1. Setup and Reproducibility
    print("Setting random seed...")
    set_seed(42)

    # 2. Data Loading
    # We set load_cached_data=False to demonstrate loading from raw JSON/CSV sources
    print("Loading and preprocessing data...")
    df_train, df_val, df_test = load_and_preprocess_data(load_cached_data=False)

    # Verify data loading
    assert not df_train.empty, "Training DataFrame is empty"
    assert not df_val.empty, "Validation DataFrame is empty"
    assert not df_test.empty, "Test DataFrame is empty"
    print(
        f"Data loaded. Train: {df_train.shape}, Val: {df_val.shape}, Test: {df_test.shape}"
    )

    # 3. Optimization for Speed
    # To ensure the demonstration runs quickly, we slice the training and validation sets.
    # We MUST keep the test set intact because the submission generation relies on
    # matching the exact number of rows in the test metadata file.
    SUBSET_SIZE = 50
    print(f"Slicing training and validation data to {SUBSET_SIZE} samples for speed...")

    df_train_small = df_train.head(SUBSET_SIZE).copy()
    df_val_small = df_val.head(SUBSET_SIZE).copy()

    # 4. Feature Extraction
    print("Initializing Feature Pipeline...")
    pipeline = HybridFeaturePipeline()

    print("Extracting features (Text Embeddings + Numerical Scaling)...")
    # We pass load_cached_data=False to ensure the pipeline processes our sliced DataFrames
    # instead of loading previously cached full-size arrays.
    X_train, y_train, X_val, y_val, X_test = pipeline.fit_transform(
        df_train_small, df_val_small, df_test, load_cached_data=False
    )

    # Verify Feature Shapes
    print("Verifying feature shapes...")
    # Train/Val should match subset size
    assert (
        len(X_train) == SUBSET_SIZE
    ), f"X_train size mismatch. Expected {SUBSET_SIZE}, got {len(X_train)}"
    assert (
        len(X_val) == SUBSET_SIZE
    ), f"X_val size mismatch. Expected {SUBSET_SIZE}, got {len(X_val)}"
    # Test should match full size
    assert len(X_test) == len(
        df_test
    ), f"X_test size mismatch. Expected {len(df_test)}, got {len(X_test)}"
    # Feature dimensions should match
    assert (
        X_train.shape[1] == X_test.shape[1]
    ), "Feature dimension mismatch between train and test"
    # Labels should match
    assert len(y_train) == SUBSET_SIZE, "y_train size mismatch"

    print(f"Feature extraction successful. Feature dimension: {X_train.shape[1]}")

    # 5. Model Training
    print("Initializing Trainer...")
    trainer = Trainer()

    # Run Cross-Validation
    # We use k_folds=3 because our subset size is small (50 samples)
    print("Running Cross-Validation to select best C...")
    best_c = trainer.run_cross_validation(X_train, y_train, k_folds=3)

    # Verify CV results
    assert best_c is not None, "Trainer failed to select a best C value"
    assert trainer.best_auc != -1.0, "Trainer failed to update best_auc"
    print(f"Best C selected: {best_c}")

    # Train Final Model
    print("Training final model on combined Train + Val subset...")
    trainer.train_final_model(X_train, y_train, X_val, y_val)

    # Verify model exists
    assert trainer.model is not None, "Final model is None after training"

    # 6. Submission Generation
    print("Generating submission for full test set...")
    trainer.generate_submission(X_test)

    # 7. Verification of Submission
    print("Verifying submission file...")
    if not os.path.exists(SUBMISSION_FILE):
        raise FileNotFoundError(f"Submission file not found at {SUBMISSION_FILE}")

    df_submission = pd.read_csv(SUBMISSION_FILE)

    # Check dimensions
    assert len(df_submission) == len(
        df_test
    ), f"Submission rows ({len(df_submission)}) do not match Test set rows ({len(df_test)})"

    # Check columns
    expected_cols = ["request_id", "requester_received_pizza"]
    assert (
        list(df_submission.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_submission.columns)}"

    # Check value types
    assert pd.api.types.is_numeric_dtype(
        df_submission["requester_received_pizza"]
    ), "Prediction column is not numeric"

    print("Process completed successfully!")


if __name__ == "__main__":
    main()
