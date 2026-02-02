import os
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import SEED, WORKING_DIR, SUBMISSION_FILE_PATH, FLOAT_PRECISION
from library.data_loader import load_datasets
from library.preprocessing import get_preprocessed_data
from library.model import CholeskyOASClassifier
from library.utils import compute_log_loss, save_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed):
    np.random.seed(seed)


def main():
    print("=== Starting Library Usage Demonstration ===\n")

    # 1. Setup
    set_seed(SEED)
    print(f"Random seed set to {SEED}")
    print(f"Working directory: {WORKING_DIR}")

    # 2. Data Loading
    print("\n--- 1. Data Loading ---")
    # We set load_cached_data=False to demonstrate loading from source CSVs
    # and to ensure the cache logic inside the function is triggered for writing.
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_datasets(
        load_cached_data=False
    )

    print(f"Train shape: {X_train.shape}, Label shape: {y_train.shape}")
    print(f"Val shape:   {X_val.shape}, Label shape: {y_val.shape}")
    print(f"Test shape:  {X_test.shape}")
    print(f"Number of classes: {len(classes)}")

    # Assertions for Data Loading
    assert X_train.dtype == FLOAT_PRECISION, "X_train should be float64"
    assert len(X_train) == len(y_train), "Mismatch in training samples and labels"
    assert len(classes) == 99, f"Expected 99 classes, got {len(classes)}"
    print("Assertion Passed: Data loaded correctly with expected shapes and types.")

    # 3. Preprocessing
    print("\n--- 2. Preprocessing ---")
    # This applies Yeo-Johnson and/or Standard Scaling based on config
    X_train_trans, X_val_trans, X_test_trans = get_preprocessed_data(
        X_train, X_val, X_test, load_cached_data=False
    )

    # Assertions for Preprocessing
    assert (
        X_train_trans.shape == X_train.shape
    ), "Preprocessing changed feature dimensions unexpectedly"
    assert (
        X_train_trans.dtype == FLOAT_PRECISION
    ), "Preprocessed data lost float64 precision"
    assert not np.isnan(X_train_trans).any(), "Preprocessed data contains NaNs"

    # Check if standardization worked (mean roughly 0, std roughly 1)
    # Note: If Yeo-Johnson is used without standardization in the config, this might vary,
    # but the provided config has USE_STANDARD_SCALER = True.
    mean_val = np.mean(X_train_trans)
    std_val = np.std(X_train_trans)
    print(f"Transformed Data Stats -> Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    assert abs(mean_val) < 0.1, "Data does not appear centered (Mean > 0.1)"
    print("Assertion Passed: Preprocessing pipeline output valid.")

    # 4. Model Training
    print("\n--- 3. Model Training (CholeskyOASClassifier) ---")
    model = CholeskyOASClassifier()

    # Fit the model
    model.fit(X_train_trans, y_train)

    # Assertions for Model State
    assert model.W_ is not None, "Model weights (W_) not initialized after fit"
    assert model.b_ is not None, "Model bias (b_) not initialized after fit"
    assert model.W_.shape == (
        len(classes),
        X_train.shape[1],
    ), f"Weight matrix shape mismatch. Expected {(len(classes), X_train.shape[1])}, got {model.W_.shape}"
    print("Assertion Passed: Model fitted and internal parameters populated.")

    # 5. Prediction and Evaluation
    print("\n--- 4. Evaluation ---")
    # Predict probabilities on validation set
    val_probs = model.predict_proba(X_val_trans)

    # Assertions for Prediction
    assert val_probs.shape == (len(X_val), len(classes)), "Prediction shape mismatch"
    # Check that probabilities sum to 1 (within float precision error)
    row_sums = val_probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"

    # Compute Log Loss
    val_loss = compute_log_loss(y_val, val_probs, classes)
    print(f"Validation Multi-class Log Loss: {val_loss:.5f}")
    assert val_loss >= 0, "Log loss cannot be negative"
    print("Assertion Passed: Metrics calculated successfully.")

    # 6. Submission Generation
    print("\n--- 5. Submission ---")
    # Predict on test set
    test_probs = model.predict_proba(X_test_trans)

    # Save submission
    save_submission(test_probs, test_ids, classes)

    # Assertions for Submission
    assert os.path.exists(SUBMISSION_FILE_PATH), "Submission file was not created"

    # Verify file content format
    df_sub = pd.read_csv(SUBMISSION_FILE_PATH)
    assert df_sub.shape == (
        len(X_test),
        len(classes) + 1,
    ), f"Submission shape mismatch. Expected {(len(X_test), len(classes) + 1)}, got {df_sub.shape}"
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert df_sub["id"].iloc[0] == test_ids[0], "ID mismatch in submission file"

    print(f"Submission verified. Shape: {df_sub.shape}")
    print("Assertion Passed: Submission file generated and verified.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
