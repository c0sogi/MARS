import os
import sys
import numpy as np
import pandas as pd
import warnings

# Ensure we can import from the library
sys.path.append(".")

# Import provided library components
from library.config import SEED, WORKING_DIR, ALL_FEATURES, FLOAT_PRECISION, ID_COL
from library.data_loader import load_and_process_data
from library.preprocessing import InductivePreprocessor, get_preprocessed_data
from library.model import RatioProjectedOAS
from library.evaluation import compute_log_loss, generate_submission_file

# Set seeds for reproducibility
np.random.seed(SEED)


def run_demo():
    print("=== Starting Demonstration of Leaf Classification Pipeline ===")

    # -------------------------------------------------------------------------
    # 1. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n[1] Testing Data Loader...")

    # We use load_cached_data=True to utilize any existing cache for speed,
    # but the function will automatically process from scratch if cache is missing.
    (train_data, val_data, test_data) = load_and_process_data(load_cached_data=True)

    X_train_raw, y_train, train_ids = train_data
    X_val_raw, y_val, val_ids = val_data
    X_test_raw, test_ids = test_data

    # Validation Checks
    print(f"    Train shape: {X_train_raw.shape}, Labels: {y_train.shape}")
    print(f"    Val shape:   {X_val_raw.shape}, Labels: {y_val.shape}")
    print(f"    Test shape:  {X_test_raw.shape}")

    # Check if custom geometric features were added
    expected_cols = set(ALL_FEATURES)
    actual_cols = set(X_train_raw.columns)
    missing_cols = expected_cols - actual_cols

    assert len(missing_cols) == 0, f"Missing expected features: {missing_cols}"
    assert (
        "Form_Factor" in X_train_raw.columns
    ), "Ratio feature 'Form_Factor' not found."
    assert "Area" in X_train_raw.columns, "Geometric primitive 'Area' not found."

    print("    Data Loader verification successful: All features present.")

    # -------------------------------------------------------------------------
    # 2. Preprocessing Demonstration
    # -------------------------------------------------------------------------
    print("\n[2] Testing Preprocessing Logic...")

    # A. Unit Test on Synthetic Data
    print("    Running unit test on InductivePreprocessor with synthetic data...")
    synth_X = np.random.rand(100, 5).astype(FLOAT_PRECISION)
    # Add some skew to test Yeo-Johnson
    synth_X[:, 0] = np.exp(synth_X[:, 0])

    proc = InductivePreprocessor()
    proc.fit(synth_X)
    synth_trans = proc.transform(synth_X)

    # Check standardization (Mean ~ 0, Std ~ 1)
    means = np.mean(synth_trans, axis=0)
    stds = np.std(synth_trans, axis=0)

    assert np.allclose(means, 0, atol=1e-6), f"Means not centered: {means}"
    assert np.allclose(stds, 1, atol=1e-6), f"Stds not scaled: {stds}"
    print("    Unit test passed.")

    # B. Full Pipeline Execution
    print("    Running full preprocessing pipeline on real data...")
    X_train, X_val, X_test = get_preprocessed_data(
        X_train_raw, X_val_raw, X_test_raw, load_cached_data=True
    )

    assert X_train.dtype == FLOAT_PRECISION, "X_train not float64"
    assert not np.isnan(X_train).any(), "NaN values found in preprocessed training data"
    print("    Preprocessing pipeline execution successful.")

    # -------------------------------------------------------------------------
    # 3. Model Training Demonstration
    # -------------------------------------------------------------------------
    print("\n[3] Testing Model (RatioProjectedOAS)...")

    model = RatioProjectedOAS()
    model.fit(X_train, y_train)

    # Check internal attributes
    n_classes = len(np.unique(y_train))
    n_features = X_train.shape[1]

    assert model.means_.shape == (n_classes, n_features), "Model means shape mismatch"
    assert model.W_.shape == (n_classes, n_features), "Model weights shape mismatch"
    assert model.b_.shape == (n_classes,), "Model bias shape mismatch"

    print(f"    Model fitted. Classes: {n_classes}, Features: {n_features}")

    # -------------------------------------------------------------------------
    # 4. Prediction and Evaluation Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Testing Prediction and Evaluation...")

    # Predict Probabilities
    val_probs = model.predict_proba(X_val)

    # Check probability properties
    row_sums = np.sum(val_probs, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), "Probabilities do not sum to 1"
    assert (
        val_probs.min() >= 0 and val_probs.max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    # Compute Log Loss
    loss = compute_log_loss(y_val, val_probs, model.classes_)
    print(f"    Validation Log Loss: {loss:.5f}")

    assert isinstance(loss, float), "Log loss is not a float"
    assert loss >= 0, "Log loss is negative"

    # Predict Classes
    val_preds = model.predict(X_val)
    assert len(val_preds) == len(y_val), "Prediction length mismatch"
    print("    Prediction and evaluation verification successful.")

    # -------------------------------------------------------------------------
    # 5. Submission Generation Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Testing Submission Generation...")

    # Generate predictions for test set
    test_probs = model.predict_proba(X_test)

    # Define output path
    demo_submission_path = os.path.join(WORKING_DIR, "demo_submission.csv")

    # Generate file
    generate_submission_file(
        ids=test_ids,
        probs=test_probs,
        classes=model.classes_,
        output_path=demo_submission_path,
    )

    # Verify file content
    assert os.path.exists(demo_submission_path), "Submission file not created"

    df_sub = pd.read_csv(demo_submission_path)

    # Check dimensions: rows = n_test, cols = id + n_classes
    expected_rows = len(test_ids)
    expected_cols = 1 + len(model.classes_)

    assert df_sub.shape == (
        expected_rows,
        expected_cols,
    ), f"Submission shape mismatch. Expected {(expected_rows, expected_cols)}, got {df_sub.shape}"

    assert ID_COL in df_sub.columns, f"'{ID_COL}' column missing in submission"

    # Check if ID column matches
    # Note: IDs in file might be int, ensure comparison is valid
    file_ids = df_sub[ID_COL].values
    assert np.array_equal(file_ids, test_ids), "IDs in submission do not match test IDs"

    print("    Submission generation verification successful.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
