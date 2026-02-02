import os
import sys
import numpy as np
import pandas as pd
import random

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library import config
from library.data_processor import LeafDataProcessor, PreprocessingPipeline
from library.model import CholeskyOASClassifier, run_task
from library.evaluation import compute_log_loss, save_submission


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def test_preprocessing_pipeline():
    print("\n--- Testing PreprocessingPipeline ---")
    # Create dummy data with some variance to avoid singular matrices in PowerTransformer
    X_dummy = np.random.rand(20, 5) * 10

    pipeline = PreprocessingPipeline()
    pipeline.fit(X_dummy)
    X_transformed = pipeline.transform(X_dummy)

    # Check shape preservation
    assert (
        X_transformed.shape == X_dummy.shape
    ), f"Shape mismatch: expected {X_dummy.shape}, got {X_transformed.shape}"

    # Check dtype enforcement
    assert (
        X_transformed.dtype == np.float64
    ), f"Dtype mismatch: expected float64, got {X_transformed.dtype}"

    # Check standardization (mean approx 0, std approx 1)
    # Note: With small N=20, this won't be perfect, but should be bounded
    means = np.mean(X_transformed, axis=0)
    stds = np.std(X_transformed, axis=0)

    assert np.all(np.abs(means) < 1e-7), f"Means not centered: {means}"
    assert np.all(np.abs(stds - 1) < 1e-7), f"Stds not scaled: {stds}"

    print("PreprocessingPipeline logic verified.")


def test_data_processor():
    print("\n--- Testing LeafDataProcessor ---")
    processor = LeafDataProcessor()

    # Force processing from scratch to verify the pipeline integration
    print("Loading data (force process)...")
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = processor.load_data(
        load_cached_data=False
    )

    # Validate dimensions
    n_features = 192  # 64 margin + 64 shape + 64 texture
    n_classes = 99

    assert (
        X_train.shape[1] == n_features
    ), f"X_train features: expected {n_features}, got {X_train.shape[1]}"
    assert (
        X_val.shape[1] == n_features
    ), f"X_val features: expected {n_features}, got {X_val.shape[1]}"
    assert (
        len(classes) == n_classes
    ), f"Classes: expected {n_classes}, got {len(classes)}"

    print(f"Data Loaded: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
    print("LeafDataProcessor logic verified.")

    return X_train, y_train, X_val, y_val, X_test, test_ids, classes


def test_model(X_train, y_train, X_val, classes):
    print("\n--- Testing CholeskyOASClassifier ---")
    clf = CholeskyOASClassifier()

    # Fit model
    print("Fitting model...")
    clf.fit(X_train, y_train)

    # Check attributes
    assert hasattr(clf, "covariance_"), "Model missing covariance_ attribute after fit"
    assert hasattr(clf, "coef_"), "Model missing coef_ attribute after fit"

    # Predict
    print("Predicting probabilities...")
    probs = clf.predict_proba(X_val)

    # Validate Output
    assert probs.shape == (
        X_val.shape[0],
        len(classes),
    ), f"Prediction shape mismatch: expected {(X_val.shape[0], len(classes))}, got {probs.shape}"

    # Check probability sum constraint
    row_sums = np.sum(probs, axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1.0"

    print("CholeskyOASClassifier logic verified.")
    return probs


def test_evaluation(y_val, probs, test_ids, X_test, classes, clf):
    print("\n--- Testing Evaluation & Submission ---")

    # 1. Compute Log Loss
    loss = compute_log_loss(y_val, probs)
    assert isinstance(loss, float), "Log loss is not a float"
    assert loss > 0, "Log loss should be positive"
    print(f"Verified Log Loss computation: {loss:.4f}")

    # 2. Save Submission
    test_probs = clf.predict_proba(X_test)
    dummy_output = os.path.join(config.WORKING_DIR, "demo_submission_test.csv")

    save_submission(test_ids, test_probs, classes, output_path=dummy_output)

    assert os.path.exists(dummy_output), "Submission file was not created"

    # Verify file content
    df_sub = pd.read_csv(dummy_output)
    expected_cols = 1 + len(classes)  # id + 99 classes
    assert (
        df_sub.shape[1] == expected_cols
    ), f"Submission columns: expected {expected_cols}, got {df_sub.shape[1]}"
    assert config.ID_COL in df_sub.columns, f"Submission missing {config.ID_COL} column"

    print("Evaluation and Submission logic verified.")


def test_full_pipeline_execution():
    print("\n--- Testing Full Pipeline (run_task) ---")
    # Run in debug mode to ensure it completes quickly
    run_task(load_cached_data=True, debug=True)

    # Check if the final submission file from run_task exists
    assert os.path.exists(
        config.SUBMISSION_FILE
    ), "Final submission file missing after run_task"
    print("Full pipeline execution verified.")


if __name__ == "__main__":
    set_seed(42)

    # 1. Test Preprocessing Pipeline Unit
    test_preprocessing_pipeline()

    # 2. Test Data Loading & Processing
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = test_data_processor()

    # 3. Test Model Logic
    val_probs = test_model(X_train, y_train, X_val, classes)

    # 4. Test Evaluation and Submission Utils
    # We need the classifier instance to generate test probs for the submission test
    clf = CholeskyOASClassifier()
    clf.fit(X_train, y_train)
    test_evaluation(y_val, val_probs, test_ids, X_test, classes, clf)

    # 5. Test End-to-End Task
    test_full_pipeline_execution()

    print("\nAll demonstrations and verifications completed successfully.")
