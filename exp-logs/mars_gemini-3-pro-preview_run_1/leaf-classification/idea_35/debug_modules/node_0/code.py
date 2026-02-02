import os
import numpy as np
import pandas as pd
import shutil
from library.config import (
    set_seed,
    WORKING_DIR,
    SUBMISSION_PATH,
    get_alphanumeric_feature_order,
)
from library.data_loader import load_datasets
from library.preprocessing import Preprocessor, get_preprocessed_data
from library.model import CholeskyOASDiscriminant, train_and_predict


def run_demo():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Setup
    set_seed(42)
    print(f"Working Directory: {WORKING_DIR}")

    # Clean working directory to ensure fresh execution for demo purposes
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 2. Demonstrate Data Loading
    print("\n--- Testing Data Loader ---")
    # Load from metadata CSVs (ignoring cache to verify raw loading logic)
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_datasets(
        load_cached_data=False
    )

    # Assertions for Data Loader
    print("Verifying data shapes and types...")
    assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame"
    assert isinstance(y_train, np.ndarray), "y_train should be a numpy array"
    assert len(X_train) == len(y_train), "Mismatch in training samples and labels"
    assert X_train.shape[1] == 192, f"Expected 192 features, got {X_train.shape[1]}"

    # Verify strict alphanumeric ordering
    expected_cols = get_alphanumeric_feature_order()
    assert (
        list(X_train.columns) == expected_cols
    ), "Feature columns are not alphanumerically sorted"

    print(f"Train shape: {X_train.shape}")
    print(f"Val shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Number of classes: {len(classes)}")
    print("Data Loader verification passed.")

    # 3. Demonstrate Preprocessing
    print("\n--- Testing Preprocessor ---")
    preprocessor = Preprocessor()

    # Fit on training data
    print("Fitting preprocessor...")
    preprocessor.fit(X_train)

    # Transform validation data
    print("Transforming validation data...")
    X_val_trans = preprocessor.transform(X_val)

    # Assertions for Preprocessor
    assert isinstance(
        X_val_trans, np.ndarray
    ), "Transformed data should be a numpy array"
    assert X_val_trans.dtype == np.float64, "Transformed data must be float64"
    assert X_val_trans.shape == X_val.shape, "Transformed shape mismatch"

    # Check statistics (StandardScaler should make mean~0 and std~1)
    # Note: Checking on Train data for strict mean=0, Val data will be close
    X_train_trans = preprocessor.transform(X_train)
    mean_val = np.mean(X_train_trans)
    std_val = np.std(X_train_trans)
    print(f"Transformed Train Mean (should be ~0): {mean_val:.4f}")
    print(f"Transformed Train Std (should be ~1): {std_val:.4f}")

    assert np.abs(mean_val) < 1e-1, "Preprocessing mean is not centered enough"
    assert np.abs(std_val - 1.0) < 1e-1, "Preprocessing std is not scaled enough"
    print("Preprocessor verification passed.")

    # Test the high-level caching function
    print("Testing get_preprocessed_data (Caching logic)...")
    # First call creates cache
    _ = get_preprocessed_data(load_cached_data=True)
    # Second call should load from cache (observable via file timestamps or logs,
    # but here we just ensure it returns consistent data)
    X_train_c, y_train_c, _, _, _, _, _ = get_preprocessed_data(load_cached_data=True)
    assert np.array_equal(
        X_train_trans, X_train_c
    ), "Cached data does not match computed data"
    print("Caching logic verification passed.")

    # 4. Demonstrate Model
    print("\n--- Testing CholeskyOASDiscriminant Model ---")
    model = CholeskyOASDiscriminant()

    # Fit model
    print("Fitting model...")
    model.fit(X_train_trans, y_train)

    # Predict probabilities
    print("Predicting probabilities on validation set...")
    probs = model.predict_proba(X_val_trans)

    # Assertions for Model
    assert probs.shape == (len(X_val), len(classes)), "Probability shape mismatch"

    # Check if probabilities sum to 1
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"

    # Check if predictions are valid
    preds = model.predict(X_val_trans)
    assert len(preds) == len(X_val), "Prediction length mismatch"
    assert set(preds).issubset(set(classes)), "Predictions contain unknown classes"

    print("Model verification passed.")

    # 5. Demonstrate Full Pipeline
    print("\n--- Testing Full Pipeline (train_and_predict) ---")
    # This function handles loading, training, predicting, and saving submission
    # We force a reload/reprocess to test the full flow
    score = train_and_predict(load_cached_data=True)

    print(f"Pipeline executed. Validation Log Loss: {score:.5f}")

    # Verify Submission File
    if os.path.exists(SUBMISSION_PATH):
        print(f"Submission file found at {SUBMISSION_PATH}")
        df_sub = pd.read_csv(SUBMISSION_PATH)

        # Check format
        assert "id" in df_sub.columns, "Submission missing 'id' column"
        assert len(df_sub) == len(
            X_test
        ), f"Submission row count mismatch. Expected {len(X_test)}, got {len(df_sub)}"

        # Check if all class columns are present
        missing_cols = set(classes) - set(df_sub.columns)
        assert not missing_cols, f"Submission missing class columns: {missing_cols}"

        print("Submission file format verification passed.")
    else:
        raise FileNotFoundError(f"Submission file was not created at {SUBMISSION_PATH}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
