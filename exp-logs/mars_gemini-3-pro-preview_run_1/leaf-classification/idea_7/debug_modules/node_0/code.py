import os
import sys
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import library modules
from library.utils import set_seed, calculate_log_loss, save_submission
from library.data_loader import load_datasets
from library.preprocessor import GlobalPreprocessor
from library.feature_bagging_lda import FeatureBaggingLDAEnsemble
from library.config import SEED, WORKING_DIR


def run_demonstration():
    print("=" * 40)
    print("Running Machine Learning Pipeline Demo")
    print("=" * 40)

    # 1. Setup
    print("\n[Step 1] Setting random seeds for reproducibility...")
    set_seed(SEED)

    # 2. Data Loading
    print("\n[Step 2] Loading datasets...")
    # We disable cache loading to demonstrate the raw data loading process
    # and ensure assertions run on fresh data.
    train_data, val_data, test_data, classes = load_datasets(load_cached_data=False)

    X_train, y_train, train_ids = train_data["X"], train_data["y"], train_data["ids"]
    X_val, y_val, val_ids = val_data["X"], val_data["y"], val_data["ids"]
    X_test, test_ids = test_data["X"], test_data["ids"]

    print(f"  Train Data Shape: {X_train.shape}")
    print(f"  Val Data Shape:   {X_val.shape}")
    print(f"  Test Data Shape:  {X_test.shape}")
    print(f"  Number of Classes: {len(classes)}")

    # Verification
    assert X_train.shape[1] == 192, "Feature dimension mismatch (expected 192)."
    assert len(y_train) == X_train.shape[0], "Train labels count mismatch."
    assert len(classes) > 0, "No classes found."

    # 3. Preprocessing
    print("\n[Step 3] Running Global Preprocessing (PowerTransform + Scaling)...")
    preprocessor = GlobalPreprocessor()

    # Process data (caching disabled to force computation)
    X_train_proc, X_val_proc, X_test_proc = preprocessor.process_and_cache(
        X_train, X_val, X_test, load_cached_data=False
    )

    # Verification
    assert (
        X_train_proc.shape == X_train.shape
    ), "Preprocessing altered feature dimensions."
    assert not np.isnan(X_train_proc).any(), "Preprocessing introduced NaNs."
    # Check standardization (approximate)
    mean_val = np.mean(X_train_proc)
    std_val = np.std(X_train_proc)
    print(f"  Processed Train Mean: {mean_val:.4f} (Expected ~0)")
    print(f"  Processed Train Std:  {std_val:.4f} (Expected ~1)")
    assert abs(mean_val) < 0.1, "Data not centered."
    assert abs(std_val - 1.0) < 0.1, "Data not scaled."

    # 4. Model Training
    print("\n[Step 4] Training Feature Bagging LDA Ensemble...")
    # Using a small number of estimators for the demo to ensure speed
    n_estimators_demo = 5
    print(f"  Training with n_estimators={n_estimators_demo} for demonstration...")

    model = FeatureBaggingLDAEnsemble(
        n_estimators=n_estimators_demo, subsample_rate=0.8, random_state=SEED
    )
    model.fit(X_train_proc, y_train)

    # Verification
    assert (
        len(model.estimators_) == n_estimators_demo
    ), "Ensemble not populated correctly."
    print("  Model training complete.")

    # 5. Evaluation
    print("\n[Step 5] Evaluating on Validation Set...")
    val_proba = model.predict_proba(X_val_proc)
    val_pred = model.predict(X_val_proc)

    # Calculate Log Loss
    # Note: calculate_log_loss handles clipping and normalization internally
    loss = calculate_log_loss(y_val, val_proba, labels=classes)

    # Calculate Accuracy
    accuracy = np.mean(val_pred == y_val)

    print(f"  Validation Log Loss: {loss:.5f}")
    print(f"  Validation Accuracy: {accuracy:.5f}")

    # Verification
    assert val_proba.shape == (
        len(y_val),
        len(classes),
    ), "Probability output shape mismatch."
    assert 0 <= accuracy <= 1, "Accuracy out of bounds."

    # 6. Submission Generation
    print("\n[Step 6] Generating Submission File...")
    test_proba = model.predict_proba(X_test_proc)

    # Define output path
    submission_path = os.path.join(WORKING_DIR, "demo_submission.csv")

    save_submission(test_ids, test_proba, classes, submission_path)

    # Verification
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"  Submission saved to: {submission_path}")
        print(f"  Submission shape: {df_sub.shape}")
        assert df_sub.shape[0] == len(test_ids), "Submission row count mismatch."
        assert df_sub.shape[1] == len(classes) + 1, "Submission column count mismatch."
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    run_demonstration()
