import os
import sys
import numpy as np
import pandas as pd
import shutil

# Import provided library modules
from library import config, utils, data_loader, teacher, student


def run_demo():
    print("--- Starting Library Demo ---")

    # 1. Setup and Configuration Overrides for Speed
    # We override the working directory for this demo to avoid conflicts
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override config paths to point to this demo directory
    config.WORKING_DIR = demo_dir
    config.SUBMISSION_DIR = demo_dir
    config.SUBMISSION_FILE_PATH = os.path.join(demo_dir, "submission.csv")

    # Set a small number of synthetic samples for the demo to run instantly
    # Default is 10,000; we use 50 per class for demonstration.
    DEMO_N_SYNTHETIC = 50

    print(f"Working Directory: {config.WORKING_DIR}")

    # 2. Demonstrate Data Loading (data_loader.py)
    print("\n[1/5] Testing Data Loader...")

    # Force processing from scratch to verify pipeline logic
    (X_train, y_train, ids_train, X_val, y_val, ids_val, X_test, ids_test, classes) = (
        data_loader.get_processed_data(load_cached_data=False)
    )

    # Verifications
    n_features = 192  # 64 margin + 64 shape + 64 texture
    assert (
        X_train.shape[1] == n_features
    ), f"Expected {n_features} features, got {X_train.shape[1]}"
    assert len(X_train) == len(y_train), "Mismatch in training samples and labels"
    assert X_train.dtype == config.FLOAT_PRECISION, "X_train dtype mismatch"
    assert len(classes) == 99, f"Expected 99 classes, got {len(classes)}"

    print(f"  Data Loaded Successfully.")
    print(f"  Train shape: {X_train.shape}")
    print(f"  Val shape:   {X_val.shape}")
    print(f"  Test shape:  {X_test.shape}")

    # 3. Demonstrate Teacher Model (teacher.py)
    print("\n[2/5] Testing Teacher Model (OAS-LDA)...")

    # Instantiate and fit teacher
    teacher_model = teacher.OASTeacher()
    teacher_model.fit(X_train, y_train)

    # Verify internal state
    assert teacher_model.means_.shape == (
        99,
        n_features,
    ), "Teacher means shape incorrect"
    assert teacher_model.covariance_.shape == (
        n_features,
        n_features,
    ), "Teacher covariance shape incorrect"

    # Test Analytic Weights Calculation
    coef, intercept = teacher_model.get_analytic_weights()
    assert coef.shape == (99, n_features), "Analytic coefficients shape incorrect"
    assert intercept.shape == (99,), "Analytic intercept shape incorrect"

    print("  Teacher fitted and analytic weights computed.")

    # 4. Demonstrate Synthetic Data Generation
    print("\n[3/5] Testing Synthetic Data Generation...")

    X_syn, y_syn = teacher_model.generate_synthetic_data(
        n_samples_per_class=DEMO_N_SYNTHETIC
    )

    expected_syn_samples = 99 * DEMO_N_SYNTHETIC
    assert X_syn.shape == (
        expected_syn_samples,
        n_features,
    ), "Synthetic data shape incorrect"
    assert len(y_syn) == expected_syn_samples, "Synthetic labels count incorrect"
    assert X_syn.dtype == config.FLOAT_PRECISION, "Synthetic data dtype mismatch"

    print(f"  Generated {len(X_syn)} synthetic samples.")

    # 5. Demonstrate Student Model (student.py)
    print("\n[4/5] Testing Student Model (Logistic Regression)...")

    # Instantiate student
    student_model = student.SyntheticStudent()

    # Fit student using synthetic data and initializing with teacher
    # We pass X_val/y_val to trigger the internal evaluation print
    student_model.fit(X_syn, y_syn, X_val=X_val, y_val=y_val, teacher=teacher_model)

    # Predict on validation set
    probs_val = student_model.predict_proba(X_val)

    # Verifications
    assert probs_val.shape == (len(X_val), 99), "Prediction probability shape incorrect"
    # Check if probabilities sum to ~1 (within float precision tolerance)
    row_sums = probs_val.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"

    # Calculate metric explicitly
    loss = student_model.evaluate(X_val, y_val)
    assert loss > 0, "Log loss should be positive"

    print(f"  Student trained and evaluated. Validation Log Loss: {loss:.4f}")

    # 6. Demonstrate Utils / Submission (utils.py)
    print("\n[5/5] Testing Submission Formatting...")

    # Predict on test set
    probs_test = student_model.predict_proba(X_test)

    # Generate submission file
    sub_df = utils.format_submission(
        test_ids=ids_test,
        y_pred_probs=probs_test,
        class_labels=classes,
        output_path=config.SUBMISSION_FILE_PATH,
    )

    # Verifications
    assert os.path.exists(
        config.SUBMISSION_FILE_PATH
    ), "Submission file was not created"
    assert sub_df.shape == (
        len(X_test),
        100,
    ), "Submission DF shape incorrect (99 classes + 1 id)"
    assert config.ID_COL in sub_df.columns, "ID column missing in submission"

    print(f"  Submission saved to {config.SUBMISSION_FILE_PATH}")
    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    # Ensure reproducibility
    np.random.seed(42)
    run_demo()
