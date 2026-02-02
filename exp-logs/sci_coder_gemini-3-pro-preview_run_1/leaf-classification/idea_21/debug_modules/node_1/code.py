import os
import sys
import numpy as np
import pandas as pd
import shutil

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import (
    INPUT_DIR,
    METADATA_DIR,
    WORKING_DIR,
    SUBMISSION_DIR,
    SEED,
    NUMERIC_TYPE,
    FEATURE_COLS,
)
from library.utils import set_seed, save_submission
from library.data_loader import load_dataset
from library.preprocessing import RobustPreprocessor, get_preprocessed_data
from library.model import LinearizedOASClassifier


def main():
    print("=== Starting Library Usage Demonstration ===\n")

    # 1. Configuration & Utils Verification
    print("--- 1. Verifying Configuration & Utils ---")
    print(f"Input Directory: {INPUT_DIR}")
    print(f"Metadata Directory: {METADATA_DIR}")
    print(f"Working Directory: {WORKING_DIR}")

    # Set global seed
    set_seed(SEED)
    print(f"Random seed set to {SEED}.")

    # Test save_submission with dummy data
    dummy_ids = [1001, 1002]
    dummy_classes = ["ClassA", "ClassB"]
    dummy_probs = np.array([[0.1, 0.9], [0.8, 0.2]])
    dummy_sub_path = os.path.join(WORKING_DIR, "dummy_submission.csv")

    save_submission(dummy_ids, dummy_probs, dummy_classes, dummy_sub_path)

    # Validate the saved file
    assert os.path.exists(dummy_sub_path), "Submission file was not created."
    df_dummy = pd.read_csv(dummy_sub_path)
    assert df_dummy.shape == (2, 3), f"Expected shape (2, 3), got {df_dummy.shape}"
    assert list(df_dummy.columns) == [
        "id",
        "ClassA",
        "ClassB",
    ], "Incorrect columns in submission file."
    print("Utils verification successful.\n")

    # 2. Data Loading Verification
    print("--- 2. Verifying Data Loader ---")
    # Force reload from metadata to test raw loading logic
    X_train_raw, y_train_raw, ids_train_raw = load_dataset(
        "train", load_cached_data=False
    )
    X_val_raw, y_val_raw, ids_val_raw = load_dataset("val", load_cached_data=False)
    X_test_raw, _, ids_test_raw = load_dataset("test", load_cached_data=False)

    print(f"Train Raw Shape: {X_train_raw.shape}")
    print(f"Val Raw Shape: {X_val_raw.shape}")
    print(f"Test Raw Shape: {X_test_raw.shape}")

    # Assertions
    assert isinstance(X_train_raw, pd.DataFrame), "X_train should be a DataFrame."
    assert len(X_train_raw) == len(
        y_train_raw
    ), "Mismatch between X and y lengths in train."
    assert (
        X_train_raw.shape[1] == 192
    ), f"Expected 192 features, got {X_train_raw.shape[1]}."
    assert X_train_raw.dtypes.iloc[0] == NUMERIC_TYPE, "Data type mismatch."

    # Check Feature Columns consistency
    assert (
        list(X_train_raw.columns) == FEATURE_COLS
    ), "Feature columns do not match config."
    print("Data Loader verification successful.\n")

    # 3. Preprocessing Verification
    print("--- 3. Verifying Preprocessing ---")

    # Test individual class usage
    preprocessor = RobustPreprocessor()
    print("Fitting RobustPreprocessor on raw training data...")
    preprocessor.fit(X_train_raw)

    print("Transforming validation data...")
    X_val_trans_manual = preprocessor.transform(X_val_raw)

    assert isinstance(
        X_val_trans_manual, np.ndarray
    ), "Transformed data should be numpy array."
    assert (
        X_val_trans_manual.dtype == NUMERIC_TYPE
    ), "Transformed data should be float64."
    assert not np.isnan(X_val_trans_manual).any(), "Transformed data contains NaNs."

    # Test the high-level orchestration function
    # This will cache the results in WORKING_DIR
    print("Running get_preprocessed_data()...")
    (train_data, val_data, test_data) = get_preprocessed_data(load_cached_data=False)

    X_train, y_train, ids_train = train_data
    X_val, y_val, ids_val = val_data
    X_test, ids_test = test_data

    assert X_train.shape == X_train_raw.shape, "Transformed shape mismatch."
    print("Preprocessing verification successful.\n")

    # 4. Model Verification
    print("--- 4. Verifying LinearizedOASClassifier ---")
    clf = LinearizedOASClassifier()

    print("Fitting model...")
    clf.fit(X_train, y_train)

    print("Predicting probabilities on validation set...")
    val_probs = clf.predict_proba(X_val)

    # Check output properties
    assert val_probs.shape == (
        len(X_val),
        99,
    ), f"Expected (n_samples, 99), got {val_probs.shape}"

    # Check probability constraints
    row_sums = np.sum(val_probs, axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1."
    assert (
        val_probs.min() >= 0.0 and val_probs.max() <= 1.0
    ), "Probabilities out of range [0, 1]."

    # Check prediction labels
    val_preds = clf.predict(X_val)
    assert len(val_preds) == len(X_val), "Prediction length mismatch."

    # Calculate a simple accuracy metric
    acc = np.mean(val_preds == y_val)
    print(f"Validation Accuracy: {acc:.4f}")
    print("Model verification successful.\n")

    # 5. Full Pipeline & Submission Generation
    print("--- 5. Generating Final Submission ---")

    # Predict on Test Set
    print("Predicting on test set...")
    test_probs = clf.predict_proba(X_test)

    # Get class names from the fitted model
    classes = list(clf.classes_)
    assert len(classes) == 99, "Model should have 99 classes."

    # Define output path
    submission_path = os.path.join(WORKING_DIR, "final_submission_demo.csv")

    # Save submission
    save_submission(ids_test, test_probs, classes, submission_path)

    # Final check
    if os.path.exists(submission_path):
        print(f"Successfully generated submission at: {submission_path}")
    else:
        raise FileNotFoundError("Failed to generate submission file.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
