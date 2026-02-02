import os
import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import log_loss

# Add the current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

# Import provided library modules
from library.config import SEED, FLOAT_PRECISION, CACHE_DIR
from library.utils import stable_softmax, clip_probabilities, alphanumeric_sort
from library.data_loader import load_data, get_features_and_targets
from library.preprocessing import Float64Preprocessor, get_preprocessed_data
from library.model import MetricOptimizedCholeskyLDA

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def set_seed(seed=SEED):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def verify_utils():
    """
    Verifies the correctness of utility functions in library/utils.py.
    """
    print("\n--- Verifying Utility Functions ---")

    # 1. Test Alphanumeric Sort
    cols = ["margin_2", "margin_1", "margin_10"]
    sorted_cols = alphanumeric_sort(cols)
    expected = ["margin_1", "margin_10", "margin_2"]
    assert sorted_cols == expected, f"Sort failed: {sorted_cols} != {expected}"
    print("Alphanumeric sort: OK")

    # 2. Test Stable Softmax
    logits = np.array([[0, 0, 0], [1000, 1000, 1000]], dtype=FLOAT_PRECISION)
    probs = stable_softmax(logits)
    # Expect uniform probabilities
    expected_probs = np.array(
        [[1 / 3, 1 / 3, 1 / 3], [1 / 3, 1 / 3, 1 / 3]], dtype=FLOAT_PRECISION
    )
    assert np.allclose(probs, expected_probs), "Softmax failed on large logits"
    print("Stable Softmax: OK")

    # 3. Test Clip Probabilities
    raw_probs = np.array([[-0.1, 0.5, 1.2], [0.0, 1.0, 0.0]], dtype=FLOAT_PRECISION)
    clipped = clip_probabilities(raw_probs)
    epsilon = 1e-15
    assert clipped.min() >= epsilon, "Clipping lower bound failed"
    assert clipped.max() <= (1.0 - epsilon), "Clipping upper bound failed"
    print("Clip Probabilities: OK")


def demonstrate_data_loading():
    """
    Demonstrates data loading using library/data_loader.py.
    """
    print("\n--- Demonstrating Data Loading ---")

    # Force reload from CSVs to demonstrate logic (bypass cache if exists)
    # In a real run, we might leave load_cached_data=True
    df_train, df_val, df_test = load_data(load_cached_data=False)

    print(f"Train shape: {df_train.shape}")
    print(f"Val shape:   {df_val.shape}")
    print(f"Test shape:  {df_test.shape}")

    # Verify types
    feature_cols = [
        c for c in df_train.columns if c not in ["id", "species", "file_path"]
    ]
    assert (
        df_train[feature_cols[0]].dtype == FLOAT_PRECISION
    ), "Features are not float64"

    # Extract X and y
    X_train, y_train = get_features_and_targets(df_train, is_test=False)
    X_test, ids_test = get_features_and_targets(df_test, is_test=True)

    assert X_train.dtype == FLOAT_PRECISION, "Extracted X_train is not float64"
    assert len(X_train) == len(y_train), "Mismatch in X and y lengths"

    print("Data loading and extraction: OK")
    return X_train, y_train, df_val, df_test


def demonstrate_preprocessing(X_train, X_val_raw):
    """
    Demonstrates preprocessing using library/preprocessing.py.
    """
    print("\n--- Demonstrating Preprocessing ---")

    preprocessor = Float64Preprocessor()

    # Fit on training data
    print("Fitting preprocessor on training data...")
    preprocessor.fit(X_train)

    # Transform
    print("Transforming data...")
    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val_raw)

    # Validation: Check statistics (StandardScaler should make mean~0, std~1)
    # Note: Yeo-Johnson is applied before scaling, so final output is strictly scaled
    mean_val = np.mean(X_train_trans, axis=0)
    std_val = np.std(X_train_trans, axis=0)

    assert np.allclose(mean_val, 0, atol=1e-7), "Transformed mean is not zero"
    assert np.allclose(std_val, 1, atol=1e-7), "Transformed std is not one"
    assert X_train_trans.dtype == FLOAT_PRECISION, "Transformed data lost precision"

    print("Preprocessing pipeline: OK")
    return X_train_trans, X_val_trans


def demonstrate_modeling(X_train, y_train, X_val, y_val):
    """
    Demonstrates model training and inference using library/model.py.
    """
    print("\n--- Demonstrating Model Training (MetricOptimizedCholeskyLDA) ---")

    model = MetricOptimizedCholeskyLDA()

    # Fit model (includes internal Grid Search for shrinkage alpha)
    print("Fitting model (this runs grid search)...")
    model.fit(X_train, y_train)

    print(f"Best Alpha Selected: {model.best_alpha_}")
    assert model.best_alpha_ is not None, "Model failed to select alpha"
    assert model.W_ is not None, "Model weights not computed"

    # Inference on Validation set
    print("Predicting on validation set...")
    probs_val = model.predict_proba(X_val)

    # Validate Probabilities
    row_sums = np.sum(probs_val, axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"

    # Calculate Metric
    # Clip first as per competition metric
    probs_val_clipped = clip_probabilities(probs_val)

    # Encode y_val to match model classes for log_loss calculation
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    le.fit(y_train)  # Ensure we use training classes
    y_val_enc = le.transform(y_val)

    score = log_loss(
        y_val_enc, probs_val_clipped, labels=np.arange(len(model.classes_))
    )
    print(f"Validation Log Loss: {score:.4f}")

    return model


def generate_submission(model, X_test, ids_test, output_path):
    """
    Generates a submission file.
    """
    print("\n--- Generating Submission ---")

    # Predict
    probs_test = model.predict_proba(X_test)
    probs_test = clip_probabilities(probs_test)

    # Create DataFrame
    submission_df = pd.DataFrame(probs_test, columns=model.classes_)
    submission_df.insert(0, "id", ids_test)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

    # Validate file format
    saved_df = pd.read_csv(output_path)
    assert "id" in saved_df.columns, "Submission missing 'id' column"
    assert len(saved_df) == len(ids_test), "Submission row count mismatch"
    assert (
        saved_df.shape[1] == len(model.classes_) + 1
    ), "Submission column count mismatch"


def main():
    set_seed(SEED)

    # 1. Verify Utilities
    verify_utils()

    # 2. Load Data (Raw)
    X_train_raw, y_train, df_val, df_test = demonstrate_data_loading()

    # Extract raw validation/test features for manual flow
    X_val_raw, y_val = get_features_and_targets(df_val, is_test=False)
    X_test_raw, ids_test = get_features_and_targets(df_test, is_test=True)

    # 3. Preprocess Data
    # We can use the manual class usage shown in demonstrate_preprocessing
    # OR use the high-level get_preprocessed_data function which handles caching.
    # Let's verify the high-level function as it's the intended primary entry point.
    print("\n--- Using High-Level Data Pipeline ---")
    # This function internally loads data, fits preprocessor, transforms, and caches
    X_train, y_train_p, X_val, y_val_p, X_test, ids_test_p = get_preprocessed_data(
        load_cached_data=False
    )

    # Verify consistency
    assert np.array_equal(
        y_train, y_train_p
    ), "Target mismatch between raw and processed loader"

    # 4. Train Model
    model = demonstrate_modeling(X_train, y_train, X_val, y_val)

    # 5. Generate Submission
    output_file = "./working/demo_submission.csv"
    generate_submission(model, X_test, ids_test, output_file)

    print("\nSUCCESS: All demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
