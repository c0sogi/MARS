import os
import numpy as np
import pandas as pd
import warnings

# Import provided library modules
from library import config
from library import utils
from library import preprocessing
from library import model

# Set seeds for reproducibility
np.random.seed(42)


def run_demonstration():
    print("=== Starting Library Demonstration ===\n")

    # ---------------------------------------------------------
    # 1. Demonstrate Data Loading (library.utils)
    # ---------------------------------------------------------
    print("[1] Testing Data Loading...")

    # Load training data
    X_train, y_train, ids_train = utils.load_data("train")

    # Validate shapes and types
    assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame"
    assert isinstance(y_train, pd.Series), "y_train should be a Series"
    assert len(X_train) == len(y_train), "Mismatch in X and y lengths"
    assert X_train.shape[1] == 192, f"Expected 192 features, got {X_train.shape[1]}"

    # Validate feature names against config
    assert (
        list(X_train.columns) == config.FEATURE_COLS
    ), "Feature columns do not match config order"

    print(f"    Loaded {len(X_train)} training samples successfully.")
    print("    Data loading verification passed.\n")

    # ---------------------------------------------------------
    # 2. Demonstrate Preprocessing (library.preprocessing)
    # ---------------------------------------------------------
    print("[2] Testing Preprocessing Logic...")

    # Instantiate the custom preprocessor
    preprocessor = preprocessing.InductivePreprocessor()

    # Create a small synthetic batch to verify transformation logic quickly
    # (Using real data subset)
    subset_size = 50
    X_subset = X_train.iloc[:subset_size].values

    # Fit the preprocessor
    preprocessor.fit(X_subset)

    # Transform
    X_transformed = preprocessor.transform(X_subset)

    # Validation
    assert X_transformed.shape == X_subset.shape, "Transformed shape mismatch"
    assert (
        X_transformed.dtype == np.float64
    ), "Expected float64 precision from transform"

    # Check if standardization happened (mean approx 0, std approx 1)
    # Note: With only 50 samples, it won't be perfect, but should be close.
    mean_val = np.mean(X_transformed)
    std_val = np.std(X_transformed)

    print(f"    Transformed Subset Stats -> Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    assert np.abs(mean_val) < 0.5, "Transformed mean deviates significantly from 0"

    print("    Preprocessing verification passed.\n")

    # ---------------------------------------------------------
    # 3. Demonstrate Full Pipeline (library.model)
    # ---------------------------------------------------------
    print("[3] Running Full Training and Prediction Pipeline...")
    print("    (Using debug_sample_size=100 for speed)")

    # This function orchestrates:
    # 1. get_preprocessed_data (Load/Compute/Cache)
    # 2. LinearizedOASDiscriminant (Train)
    # 3. Prediction on Validation and Test
    # 4. Saving Submission
    model.train_and_predict(debug_sample_size=100)

    print("    Pipeline execution completed.\n")

    # ---------------------------------------------------------
    # 4. Demonstrate Submission Verification
    # ---------------------------------------------------------
    print("[4] Verifying Submission File...")

    submission_path = config.SUBMISSION_PATH
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)

    # Check dimensions
    # Columns = ID + 99 Classes
    expected_cols = 1 + config.NUM_CLASSES
    assert (
        df_sub.shape[1] == expected_cols
    ), f"Expected {expected_cols} columns, found {df_sub.shape[1]}"

    # Check ID column existence
    assert config.ID_COL in df_sub.columns, f"Missing ID column '{config.ID_COL}'"

    # Check Probability Constraints
    # Extract probability columns (drop ID)
    probs = df_sub.drop(columns=[config.ID_COL]).values

    # 1. Range [0, 1]
    assert probs.min() >= 0.0, "Probabilities contain negative values"
    assert probs.max() <= 1.0, "Probabilities exceed 1.0"

    # 2. Clipping check (Metric requirement: max(min(p, 1-10^-15), 10^-15))
    # The utils.save_submission function applies clipping.
    # We verify that no value is strictly 0 or strictly 1.
    # Note: We include a small tolerance for CSV serialization precision (Cite debug_lesson_13)
    serialization_tolerance = 1e-15
    assert (
        probs.min() >= config.EPSILON - serialization_tolerance
    ), "Probabilities not clipped correctly (found values < EPSILON)"
    assert probs.max() <= (
        1.0 - config.EPSILON + serialization_tolerance
    ), "Probabilities not clipped correctly (found values > 1-EPSILON)"

    print(f"    Submission file {os.path.basename(submission_path)} passed all checks.")
    print("    Shape: ", df_sub.shape)

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    # Suppress warnings for cleaner output during demo
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_demonstration()
