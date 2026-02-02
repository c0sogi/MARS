import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score

# Import from the provided library files
from library.config import SEED, WORKING_DIR
from library.utils import set_seed, get_logger
from library.data import load_dataset, SanitizedPreprocessor
from library.model import OASDiscriminant

# Initialize Logger
logger = get_logger("demo_script")


def run_demo():
    logger.info("Starting End-to-End Demo Script...")

    # 1. Setup and Configuration
    set_seed(SEED)
    submission_path = os.path.join(WORKING_DIR, "demo_submission.csv")

    # 2. Data Loading & Feature Generation
    # We load the full datasets. The dataset is small (~1000 images total),
    # so feature extraction (including image processing) is fast (< 1 min).
    logger.info("Loading datasets...")

    # Load Training Data
    X_train, y_train, ids_train = load_dataset("train", load_cached_data=True)
    logger.info(f"Training Data Loaded: X={X_train.shape}, y={y_train.shape}")

    # Load Validation Data
    X_val, y_val, ids_val = load_dataset("val", load_cached_data=True)
    logger.info(f"Validation Data Loaded: X={X_val.shape}, y={y_val.shape}")

    # Load Test Data (No targets)
    X_test, _, ids_test = load_dataset("test", load_cached_data=True)
    logger.info(f"Test Data Loaded: X={X_test.shape}")

    # Validation: Check feature count
    # Expected: 192 tabular features (margin/shape/texture) + 6 geometric features = 198
    expected_cols = 198
    assert (
        X_train.shape[1] == expected_cols
    ), f"Expected {expected_cols} features, got {X_train.shape[1]}"

    # 3. Preprocessing
    # The SanitizedPreprocessor handles float64 casting, constant removal,
    # Yeo-Johnson transformation, and Standardization.
    logger.info("Fitting Preprocessor...")
    preprocessor = SanitizedPreprocessor()

    # Fit on Train, Transform Train
    X_train_proc = preprocessor.fit_transform(X_train)

    # Transform Val and Test
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    logger.info(f"Preprocessing complete. Features retained: {X_train_proc.shape[1]}")

    # Validation: Check statistics of processed data (approx 0 mean, 1 std)
    train_mean = np.mean(X_train_proc, axis=0)
    train_std = np.std(X_train_proc, axis=0)
    assert np.allclose(train_mean, 0, atol=1e-5), "Processed training mean should be ~0"
    assert np.allclose(train_std, 1, atol=1e-5), "Processed training std should be ~1"

    # 4. Model Training
    # OASDiscriminant is a custom Linear Discriminant Analysis using OAS covariance estimation.
    logger.info("Initializing and Fitting OASDiscriminant Model...")
    model = OASDiscriminant()
    model.fit(X_train_proc, y_train)

    logger.info(f"Model fitted on {len(model.classes_)} classes.")

    # 5. Evaluation on Validation Set
    logger.info("Evaluating on Validation Set...")

    # Get probabilities
    val_probs_df = model.predict_proba(X_val_proc)

    # Get hard predictions
    val_preds = model.predict(X_val_proc)

    # Validation: Check probability properties
    # Sum of rows should be 1.0
    row_sums = val_probs_df.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9), "Probabilities must sum to 1"

    # Calculate Metrics
    # Note: We need to ensure the columns of val_probs_df match the class order expected by log_loss
    # OASDiscriminant.predict_proba returns a DataFrame with class names as columns.
    # We align y_val to these classes.
    acc = accuracy_score(y_val, val_preds)

    # Log Loss requires the full probability matrix aligned with classes
    # y_val contains string labels.
    loss = log_loss(y_val, val_probs_df[model.classes_].values, labels=model.classes_)

    logger.info(f"Validation Accuracy: {acc:.4f}")
    logger.info(f"Validation Log Loss: {loss:.4f}")

    # 6. Generate Submission
    logger.info("Generating predictions for Test Set...")
    test_probs_df = model.predict_proba(X_test_proc)

    # Prepare Submission DataFrame
    # Format: id, Class1, Class2, ...
    submission = pd.DataFrame()
    submission["id"] = ids_test

    # Concatenate probabilities
    # Ensure columns are sorted alphabetically or match sample_submission requirements if strict
    # The sample submission usually has sorted columns.
    sorted_classes = sorted(model.classes_)
    submission = pd.concat([submission, test_probs_df[sorted_classes]], axis=1)

    # Save Submission
    submission.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to {submission_path}")

    # Final Validation of Submission File
    saved_df = pd.read_csv(submission_path)
    assert saved_df.shape[0] == len(X_test), "Submission row count mismatch"
    assert (
        saved_df.shape[1] == len(sorted_classes) + 1
    ), "Submission column count mismatch"
    assert "id" in saved_df.columns, "id column missing from submission"

    logger.info("Demo completed successfully.")


if __name__ == "__main__":
    run_demo()
