import os
import shutil
import numpy as np
import pandas as pd
import warnings

# Import provided library modules
from library import config
from library import data_loader
from library import preprocessing
from library import model
from library import metrics


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("1. Setting up configuration for demonstration...")

    # Set fixed random seed for reproducibility
    np.random.seed(42)

    # Override configuration for speed and isolation
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 50  # Use a tiny subset for quick demonstration
    config.WORKING_DIR = "./working/demo_execution"  # Isolated working dir
    config.SUBMISSION_PATH = os.path.join(config.WORKING_DIR, "submission.csv")

    # Ensure clean working directory
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    print(f"   Debug Mode: {config.DEBUG}")
    print(f"   Sample Size: {config.DEBUG_SAMPLE_SIZE}")
    print(f"   Working Directory: {config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n2. Demonstrating Data Loader...")

    # Load datasets (force processing from metadata, ignore existing cache)
    (X_train, y_train, train_ids), (X_val, y_val, val_ids), (X_test, test_ids) = (
        data_loader.load_datasets(load_cached_data=False)
    )

    # Validation
    print("   Validating loaded data shapes and types...")
    assert (
        len(X_train) == config.DEBUG_SAMPLE_SIZE
    ), f"Expected {config.DEBUG_SAMPLE_SIZE} training samples, got {len(X_train)}"
    assert (
        len(X_val) == config.DEBUG_SAMPLE_SIZE
    ), f"Expected {config.DEBUG_SAMPLE_SIZE} validation samples, got {len(X_val)}"

    # Check feature consistency
    expected_cols = config.FEATURES
    assert (
        list(X_train.columns) == expected_cols
    ), "Feature columns do not match config."
    assert X_train.dtypes.iloc[0] == np.float64, "Features must be float64."

    print("   Data Loader verification passed.")

    # -------------------------------------------------------------------------
    # 3. Preprocessing Demonstration
    # -------------------------------------------------------------------------
    print("\n3. Demonstrating Preprocessing Pipeline...")

    # Initialize and run preprocessing
    # This caches files to disk and returns numpy arrays
    X_train_trans, X_val_trans, X_test_trans = preprocessing.preprocess_datasets(
        X_train, X_val, X_test, load_cached_data=False
    )

    # Validation
    print("   Validating transformed data...")
    assert isinstance(
        X_train_trans, np.ndarray
    ), "Transformed data must be numpy array."
    assert X_train_trans.dtype == np.float64, "Transformed data must be float64."
    assert X_train_trans.shape == X_train.shape, "Shape mismatch after transformation."

    # Check if standardization worked (mean ~ 0, std ~ 1)
    # Note: On very small debug samples, this might fluctuate, but we check rough bounds
    mean_val = np.mean(X_train_trans)
    std_val = np.std(X_train_trans)
    print(f"   Global Mean: {mean_val:.4f}, Global Std: {std_val:.4f}")

    assert np.abs(mean_val) < 0.1, "Global mean should be close to 0."
    assert np.abs(std_val - 1.0) < 0.2, "Global std should be close to 1."

    print("   Preprocessing verification passed.")

    # -------------------------------------------------------------------------
    # 4. Model Training & Prediction Demonstration
    # -------------------------------------------------------------------------
    print("\n4. Demonstrating CholeskyOASLinearDiscriminant Model...")

    clf = model.CholeskyOASLinearDiscriminant()

    # Fit model
    print("   Fitting model...")
    clf.fit(X_train_trans, y_train)

    # Validate internal state
    assert clf.classes_ is not None, "Model classes not initialized."
    assert clf.W_ is not None, "Model weights not initialized."
    assert clf.b_ is not None, "Model bias not initialized."

    # Predict on validation set
    print("   Predicting probabilities...")
    val_probs = clf.predict_proba(X_val_trans)

    # Validation of probabilities
    assert val_probs.shape == (
        len(X_val),
        len(clf.classes_),
    ), "Probability matrix shape mismatch."

    # Check row sums are 1.0
    row_sums = val_probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1."

    # Check range [0, 1]
    assert (val_probs >= 0).all() and (
        val_probs <= 1
    ).all(), "Probabilities out of range [0, 1]."

    print("   Model verification passed.")

    # -------------------------------------------------------------------------
    # 5. Metrics Demonstration
    # -------------------------------------------------------------------------
    print("\n5. Demonstrating Metric Calculation...")

    loss = metrics.calculate_log_loss(y_val, val_probs, labels=clf.classes_)
    print(f"   Calculated Log Loss: {loss:.6f}")

    assert isinstance(loss, float), "Log loss must be a float."
    assert loss >= 0, "Log loss must be non-negative."

    print("   Metric verification passed.")

    # -------------------------------------------------------------------------
    # 6. End-to-End Pipeline Demonstration
    # -------------------------------------------------------------------------
    print("\n6. Demonstrating Full Pipeline (train_and_predict)...")

    # The train_and_predict function encapsulates the whole flow and writes to SUBMISSION_PATH
    # We expect it to use the cache we just generated since load_cached_data=True is default inside
    model.train_and_predict()

    # Validation of submission file
    print(f"   Verifying submission file at {config.SUBMISSION_PATH}...")
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(config.SUBMISSION_PATH)

    # Check shape: (n_test_samples, n_classes + 1 for id)
    # Note: In debug mode, test size is config.DEBUG_SAMPLE_SIZE
    expected_rows = config.DEBUG_SAMPLE_SIZE
    # Number of columns = number of classes + 1 ('id')
    # We can estimate classes from y_train unique values in this debug subset
    n_classes_debug = len(np.unique(y_train))

    # Note: The model might learn all classes present in y_train.
    # In a tiny debug subset, not all 99 classes might be present.
    # However, the submission columns are derived from clf.classes_.
    # Let's verify basic structure.

    assert (
        len(df_sub) == expected_rows
    ), f"Submission has {len(df_sub)} rows, expected {expected_rows}."
    assert "id" in df_sub.columns, "Submission missing 'id' column."

    # Check that probabilities in submission are valid
    prob_cols = [c for c in df_sub.columns if c != "id"]
    sub_probs = df_sub[prob_cols].values

    # Row sums check
    sub_row_sums = sub_probs.sum(axis=1)
    # Allow small floating point tolerance
    assert np.allclose(
        sub_row_sums, 1.0, atol=1e-5
    ), "Submission probabilities do not sum to 1."

    print("   Full pipeline verification passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
