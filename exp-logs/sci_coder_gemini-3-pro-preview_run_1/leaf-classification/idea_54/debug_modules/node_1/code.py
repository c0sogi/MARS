import os
import numpy as np
import pandas as pd
import sys

# Import from the provided library files
from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    GEOMETRIC_FEATURES,
    SUBMISSION_PATH,
    RANDOM_SEED,
    FLOAT_PRECISION,
)
from library.feature_engineering import extract_geometric_features_single
from library.data_processing import get_processed_data
from library.model import HighPrecisionOASDiscriminant, train_and_predict


def set_seed(seed):
    np.random.seed(seed)


def test_geometric_feature_extraction():
    print("\n--- Testing Geometric Feature Extraction ---")

    # Load metadata to find a valid image path
    if not os.path.exists(TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found at {TRAIN_METADATA_PATH}")

    df_train = pd.read_csv(TRAIN_METADATA_PATH)
    if df_train.empty:
        raise ValueError("Training metadata is empty.")

    # Get the first image path
    sample_image_rel_path = df_train.iloc[0]["file_path"]
    print(f"Testing extraction on image: {sample_image_rel_path}")

    # Extract features
    features = extract_geometric_features_single(sample_image_rel_path)

    # Validation
    print(f"Extracted features: {features}")

    # Check if all expected keys are present
    missing_keys = [key for key in GEOMETRIC_FEATURES if key not in features]
    if missing_keys:
        raise AssertionError(f"Missing geometric features: {missing_keys}")

    # Check if values are floats (or convertible to float)
    for key, value in features.items():
        if not isinstance(value, (float, np.floating, int, np.integer)):
            raise AssertionError(f"Feature {key} is not numeric: {type(value)}")

    print("Geometric feature extraction test passed.")


def test_data_processing_pipeline():
    print("\n--- Testing Data Processing Pipeline ---")

    # Force reload to test the processing logic (bypass cache)
    print("Loading and processing data (load_cached_data=False)...")
    X_train, y_train, X_val, y_val, X_test, test_ids = get_processed_data(
        load_cached_data=False
    )

    # Validation
    print(f"X_train shape: {X_train.shape}")
    print(f"X_train dtype: {X_train.dtype}")

    # Check Precision
    if X_train.dtype != FLOAT_PRECISION:
        raise AssertionError(f"Expected dtype {FLOAT_PRECISION}, got {X_train.dtype}")

    # Check Scaling (StandardScaler should result in mean ~0 and std ~1 for train set)
    # We check a random feature column
    col_idx = 0
    mean_val = np.mean(X_train[:, col_idx])
    std_val = np.std(X_train[:, col_idx])

    print(f"Feature {col_idx} - Mean: {mean_val:.4f}, Std: {std_val:.4f}")

    if not np.isclose(mean_val, 0, atol=1e-5):
        raise AssertionError("Feature standardization failed: Mean is not approx 0")
    if not np.isclose(std_val, 1, atol=1e-5):
        raise AssertionError("Feature standardization failed: Std is not approx 1")

    print("Data processing pipeline test passed.")
    return X_train, y_train, X_val, y_val


def test_model_logic(X_train, y_train, X_val, y_val):
    print("\n--- Testing HighPrecisionOASDiscriminant Model ---")

    # Encode labels for the model
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    # Fit on all known labels to be safe
    all_labels = np.concatenate([y_train, y_val])
    le.fit(all_labels)

    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)

    # Instantiate
    model = HighPrecisionOASDiscriminant()

    # Fit
    print("Fitting model...")
    model.fit(X_train, y_train_enc)

    # Check if attributes are set
    if model.means_ is None or model.precision_ is None:
        raise AssertionError("Model attributes (means_, precision_) not set after fit.")

    # Predict
    print("Predicting on validation set...")
    probs = model.predict_proba(X_val)

    # Validation
    print(f"Probabilities shape: {probs.shape}")

    if probs.shape[0] != X_val.shape[0]:
        raise AssertionError("Prediction row count mismatch.")
    if probs.shape[1] != len(le.classes_):
        raise AssertionError("Prediction column count mismatch (classes).")

    # Check probability constraints
    # Sum of probs should be approx 1
    row_sums = np.sum(probs, axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-9):
        raise AssertionError("Probabilities do not sum to 1.")

    # Check clipping (no values should be exactly 0 or 1 if clipping logic works for log loss)
    if np.any(probs == 0) or np.any(probs == 1):
        print("Note: Probabilities contain exact 0 or 1. Checking clipping logic...")
        # The model code clips to [1e-15, 1-1e-15], so strict 0 or 1 should not exist
        if np.min(probs) < 1e-15:
            raise AssertionError("Probabilities contain values smaller than epsilon.")

    print("Model logic test passed.")


def test_full_pipeline():
    print("\n--- Testing Full Pipeline (Train & Predict) ---")

    # This function in library.model runs the whole flow and saves submission
    val_loss = train_and_predict(load_cached_data=True)

    print(f"Pipeline finished with Validation Log Loss: {val_loss:.4f}")

    # Verify submission file
    if not os.path.exists(SUBMISSION_PATH):
        raise AssertionError(f"Submission file not found at {SUBMISSION_PATH}")

    df_sub = pd.read_csv(SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")

    # Check columns
    if "id" not in df_sub.columns:
        raise AssertionError("Submission missing 'id' column.")

    # Check if we have probabilities for classes
    # We expect 99 classes + 1 id column = 100 columns
    if df_sub.shape[1] != 100:
        raise AssertionError(
            f"Expected 100 columns in submission, found {df_sub.shape[1]}"
        )

    print("Full pipeline test passed.")


if __name__ == "__main__":
    set_seed(RANDOM_SEED)

    try:
        # 1. Test Feature Engineering
        test_geometric_feature_extraction()

        # 2. Test Data Processing
        X_train, y_train, X_val, y_val = test_data_processing_pipeline()

        # 3. Test Model
        test_model_logic(X_train, y_train, X_val, y_val)

        # 4. Test Full Pipeline
        test_full_pipeline()

        print("\nAll tests completed successfully!")

    except Exception as e:
        print(f"\nFAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
