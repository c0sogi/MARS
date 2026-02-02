import os
import numpy as np
import pandas as pd
import warnings
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import ConvergenceWarning

# Import library components
from library.data_loader import load_and_preprocess
from library.models import get_linear_branch, get_generative_branch, get_kernel_branch
from library.ensemble import soft_vote, run_ensemble

# Configuration
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Suppress warnings for cleaner output during demo
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def validate_data_loader():
    """
    Demonstrates and validates the data loading and preprocessing module.
    """
    print("\n=== 1. Validating Data Loader ===")

    # Load data (forcing reprocessing to verify logic, though cache might be used if present)
    # We use load_cached_data=False to ensure we test the raw reading logic.
    X_train, y_train, X_test, test_ids, le = load_and_preprocess(load_cached_data=False)

    # Basic Shape Assertions
    print(f"Loaded Train Shape: {X_train.shape}")
    print(f"Loaded Test Shape: {X_test.shape}")
    print(f"Number of Classes: {len(le.classes_)}")

    assert X_train.ndim == 2, "X_train must be 2D"
    assert X_test.ndim == 2, "X_test must be 2D"
    assert len(y_train) == X_train.shape[0], "y_train length mismatch"
    assert len(test_ids) == X_test.shape[0], "test_ids length mismatch"
    assert X_train.shape[1] == 192, "Feature count should be 192 (64*3)"

    # Check Scaling (StandardScaler should result in mean ~0 and std ~1)
    mean_val = np.mean(X_train)
    std_val = np.std(X_train)
    print(f"Global Mean: {mean_val:.4f}, Global Std: {std_val:.4f}")

    assert (
        np.abs(mean_val) < 0.1
    ), "Data does not appear to be centered (StandardScaler)"
    assert (
        np.abs(std_val - 1.0) < 0.1
    ), "Data does not appear to be scaled (StandardScaler)"

    return X_train, y_train, X_test, le


def validate_models(X, y):
    """
    Demonstrates instantiation and training of individual model branches.
    Uses a small subset and reduced iterations for speed.
    """
    print("\n=== 2. Validating Model Branches (Fast Mode) ===")

    # Create a small subset for rapid testing
    # Cite debug_lesson_2: Verify Class Counts Exceed CV Folds When Subsampling
    target_classes = np.unique(y)[:5]
    mask = np.isin(y, target_classes)
    X_sub = X[mask]
    y_sub = y[mask]
    subset_size = len(X_sub)
    print(
        f"Subset created with {len(target_classes)} classes and {subset_size} samples."
    )

    # ---------------------------------------------------------
    # A. Linear Branch (Logistic Regression)
    # ---------------------------------------------------------
    print("Testing Linear Branch...")
    # Reduce CV folds and max_iter for speed demonstration
    linear_model = get_linear_branch(cv=2, max_iter=10, random_state=RANDOM_SEED)
    linear_model.fit(X_sub, y_sub)

    check_is_fitted(linear_model)
    probs = linear_model.predict_proba(X_sub)
    assert probs.shape == (
        subset_size,
        len(target_classes),
    ), f"Probability shape mismatch: {probs.shape}"
    assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities do not sum to 1"
    print("-> Linear Branch OK")

    # ---------------------------------------------------------
    # B. Generative Branch (LDA)
    # ---------------------------------------------------------
    print("Testing Generative Branch...")
    gen_model = get_generative_branch()
    gen_model.fit(X_sub, y_sub)

    check_is_fitted(gen_model)
    probs_gen = gen_model.predict_proba(X_sub)
    assert probs_gen.shape == (subset_size, len(target_classes))
    print("-> Generative Branch OK")

    # ---------------------------------------------------------
    # C. Kernel Branch (Nystroem + LR)
    # ---------------------------------------------------------
    print("Testing Kernel Branch...")
    # Reduce components and iterations for speed
    kernel_model = get_kernel_branch(
        cv=2, max_iter=10, nystroem_components=50, random_state=RANDOM_SEED
    )
    kernel_model.fit(X_sub, y_sub)

    # Pipeline check
    assert "nystroem" in kernel_model.named_steps
    assert "classifier" in kernel_model.named_steps
    probs_kern = kernel_model.predict_proba(X_sub)
    assert probs_kern.shape == (subset_size, len(target_classes))
    print("-> Kernel Branch OK")


def validate_ensemble_logic():
    """
    Validates the soft voting mechanism.
    """
    print("\n=== 3. Validating Ensemble Logic ===")

    # Create dummy probabilities: 2 samples, 3 classes
    p1 = np.array([[0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
    p2 = np.array([[0.3, 0.4, 0.3], [0.8, 0.1, 0.1]])

    # Expected average
    expected = (p1 + p2) / 2.0

    result = soft_vote([p1, p2])

    assert np.allclose(result, expected), "Soft vote averaging is incorrect"
    print("-> Soft Vote Logic OK")


def validate_full_pipeline():
    """
    Runs the full ensemble pipeline provided by the library.
    This ensures integration works as expected.
    """
    print("\n=== 4. Validating Full Pipeline (run_ensemble) ===")
    print("Note: This runs the actual training sequence defined in library/ensemble.py")

    # We use load_cached_data=True to speed up if data was cached by step 1
    # random_state ensures reproducibility
    df_submission = run_ensemble(load_cached_data=True, random_state=RANDOM_SEED)

    # Verify Output
    assert isinstance(df_submission, pd.DataFrame), "Output must be a DataFrame"
    assert "id" in df_submission.columns, "Submission must contain 'id' column"

    # Check dimensions (Test set has 99 samples, +1 column for ID, 99 classes)
    # Total columns = 1 (id) + 99 (species) = 100
    expected_rows = 99
    expected_cols = 100

    print(f"Submission Shape: {df_submission.shape}")
    assert df_submission.shape == (
        expected_rows,
        expected_cols,
    ), f"Expected ({expected_rows}, {expected_cols}), got {df_submission.shape}"

    # Check file existence
    submission_path = "./submission/submission.csv"
    assert os.path.exists(submission_path), "Submission file was not saved"
    print(f"-> Pipeline execution successful. File saved at {submission_path}")


if __name__ == "__main__":
    print("Starting Library Demonstration...")

    # 1. Validate Data Loading
    X_train, y_train, X_test, le = validate_data_loader()

    # 2. Validate Individual Models (Fast)
    validate_models(X_train, y_train)

    # 3. Validate Helper Logic
    validate_ensemble_logic()

    # 4. Validate Full Ensemble Execution
    validate_full_pipeline()

    print("\nAll demonstrations completed successfully.")
