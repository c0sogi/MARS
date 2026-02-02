import os
import numpy as np
import pandas as pd
import sys

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import SEED, WORKING_DIR, SUBMISSION_DIR, N_CLASSES
from library.utils import seed_everything, log_loss_metric, format_submission
from library.data_manager import get_data
from library.preprocessor import get_preprocessed_data, PrecisionPipeline
from library.oas_discriminant import TransductiveOAS


def run_demonstration():
    print("=== Starting Task Demonstration ===\n")

    # 1. Setup
    print("[1] Setting random seeds...")
    seed_everything(SEED)

    # 2. Data Loading
    print("\n[2] Loading Raw Data...")
    # Force reload to demonstrate the loading logic
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, test_ids, class_names = (
        get_data(load_cached_data=False)
    )

    print(f"    Train shape: {X_train_raw.shape}, dtype: {X_train_raw.dtype}")
    print(f"    Val shape:   {X_val_raw.shape}, dtype: {X_val_raw.dtype}")
    print(f"    Test shape:  {X_test_raw.shape}, dtype: {X_test_raw.dtype}")

    # Validation: Raw data should be float64
    assert X_train_raw.dtype == np.float64, "Raw training data should be float64."
    assert (
        len(class_names) == N_CLASSES
    ), f"Expected {N_CLASSES} classes, found {len(class_names)}."
    print("    -> Raw data validation passed.")

    # 3. Preprocessing
    print("\n[3] Running Precision Preprocessing Pipeline...")
    # This uses PrecisionPipeline internally: Yeo-Johnson -> StandardScaler -> float32 cast
    X_train, y_train, X_val, y_val, X_test, test_ids, class_names = (
        get_preprocessed_data(load_cached_data=False)
    )

    print(f"    Transformed Train shape: {X_train.shape}, dtype: {X_train.dtype}")

    # Validation: Transformed data should be float32
    assert (
        X_train.dtype == np.float32
    ), "Transformed data must be float32 for precision consistency."
    assert X_val.dtype == np.float32, "Transformed validation data must be float32."

    # Validation: Check standardization (Mean ~ 0, Std ~ 1)
    train_mean = np.mean(X_train)
    train_std = np.std(X_train)
    print(f"    Global Mean: {train_mean:.4f}, Global Std: {train_std:.4f}")
    assert np.abs(train_mean) < 1e-2, "Standardization failed: Mean is not approx 0."
    assert (
        np.abs(train_std - 1.0) < 1e-2
    ), "Standardization failed: Std is not approx 1."
    print("    -> Preprocessing validation passed.")

    # 4. Model Training (Transductive OAS)
    print("\n[4] Training Transductive OAS Discriminant...")
    # We use a high confidence threshold to be safe, or lower it to ensure some samples are selected for demo
    model = TransductiveOAS(confidence_threshold=0.90)

    # Fit with training data AND unlabeled test data for transduction
    model.fit(X_train, y_train, X_test_unlabeled=X_test)

    # Validation: Check parameter quantization
    print("    Checking model parameter precision...")
    assert (
        model.means_.dtype == np.float32
    ), "Model means should be quantized to float32."
    assert (
        model.precision_.dtype == np.float32
    ), "Model precision matrix should be quantized to float32."
    print("    -> Model parameter validation passed.")

    # 5. Evaluation
    print("\n[5] Evaluating on Validation Set...")
    val_probs = model.predict_proba(X_val)

    # Validation: Probabilities shape and sum
    assert val_probs.shape == (len(X_val), N_CLASSES), "Probability shape mismatch."
    # Sum of rows should be close to 1 (softmax output)
    row_sums = np.sum(val_probs, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1."

    # Calculate Metric
    score = log_loss_metric(y_val, val_probs)
    print(f"    Validation Multi-class Log Loss: {score:.5f}")
    assert score >= 0, "Log loss cannot be negative."
    print("    -> Evaluation validation passed.")

    # 6. Submission
    print("\n[6] Generating Submission...")
    test_probs = model.predict_proba(X_test)

    submission_path = os.path.join(SUBMISSION_DIR, "demo_submission.csv")
    format_submission(test_ids, test_probs, class_names, submission_path)

    # Validation: Check file creation
    assert os.path.exists(submission_path), "Submission file was not created."
    df_sub = pd.read_csv(submission_path)
    assert df_sub.shape == (
        len(X_test),
        N_CLASSES + 1,
    ), "Submission file has incorrect shape (rows or cols)."
    assert "id" in df_sub.columns, "Submission file missing 'id' column."
    print(f"    -> Submission generated successfully at {submission_path}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
