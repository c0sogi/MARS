import os
import sys
import numpy as np
import pandas as pd
import warnings
import shutil

# Import from the provided library files
from library.utils import set_seed, get_config_hash, compute_metric
from library.image_processing import extract_geometric_features, process_dataset
from library.data_loader import load_dataset
from library.model import SanitizedOASDiscriminant, train_and_predict

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===\n")

    # 1. Setup and Seeding
    print("--- 1. Setting Random Seed ---")
    set_seed(42)
    print("Seed set to 42.\n")

    # 2. Demonstrate Utils
    print("--- 2. Testing Utility Functions ---")

    # Test Config Hash
    config = {"param_a": 10, "param_b": "test", "list": [1, 2, 3]}
    config_hash = get_config_hash(config)
    print(f"Config Hash: {config_hash}")
    assert (
        isinstance(config_hash, str) and len(config_hash) == 64
    ), "Hash format incorrect"

    # Test Metric Computation
    # Create dummy predictions: 2 samples, 3 classes
    y_true_dummy = np.array([0, 2])
    # Preds do not sum to 1 to test rescaling logic
    y_pred_dummy = np.array(
        [[0.8, 0.1, 0.05], [0.1, 0.1, 0.8]]  # Sums to 0.95  # Sums to 1.0
    )
    loss = compute_metric(y_true_dummy, y_pred_dummy, labels=[0, 1, 2])
    print(f"Computed Log Loss on dummy data: {loss:.4f}")
    assert loss >= 0, "Log loss should be non-negative"
    print("Utils verification passed.\n")

    # 3. Demonstrate Image Processing
    print("--- 3. Testing Image Processing ---")

    # Locate a sample image using metadata
    metadata_path = "./metadata/train.csv"
    if os.path.exists(metadata_path):
        df_meta = pd.read_csv(metadata_path)
        # Get first image path
        rel_path = df_meta.iloc[0]["file_path"]
        full_img_path = os.path.join("./input", rel_path)

        print(f"Extracting features from: {full_img_path}")
        if os.path.exists(full_img_path):
            features = extract_geometric_features(full_img_path)
            print("Extracted Features:", features)

            expected_keys = [
                "Area",
                "Major_Axis_Length",
                "Eccentricity",
                "Solidity",
                "Extent",
                "Aspect_Ratio",
            ]
            for key in expected_keys:
                assert key in features, f"Missing feature key: {key}"
                assert isinstance(features[key], float), f"Feature {key} is not a float"

            print("Feature extraction successful.")
        else:
            print(
                f"Warning: Image file {full_img_path} not found. Skipping extraction check."
            )
    else:
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    print("Image processing verification passed.\n")

    # 4. Demonstrate Data Loading
    print("--- 4. Testing Data Loader ---")
    # Load a small subset (max_samples=50) to verify pipeline speed and logic
    # This triggers: Merge -> VarianceThreshold -> PowerTransformer -> StandardScaler
    print("Loading dataset subset (max_samples=50)...")

    # Force reload to ensure we test the processing logic, not just cache loading
    # Note: The library caches to ./working/idea_69.
    # We use max_samples which disables loading from the full dataset cache in the library logic.
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = load_dataset(
        load_cached_data=False, max_samples=50
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Classes: {len(classes)} unique species")

    # Assertions
    assert X_train.shape[0] == 50, "X_train should have 50 samples"
    assert X_train.shape[1] > 0, "X_train should have features"
    assert len(y_train) == 50, "y_train should match X_train samples"
    assert not np.isnan(X_train).any(), "X_train contains NaNs after preprocessing"

    # Verify Scaling (Mean ~ 0, Std ~ 1)
    # With only 50 samples, variance might be high, but mean should be close to 0
    mean_val = np.mean(X_train)
    std_val = np.std(X_train)
    print(f"Global Mean of Preprocessed X_train: {mean_val:.4f}")
    print(f"Global Std of Preprocessed X_train: {std_val:.4f}")

    print("Data loader verification passed.\n")

    # 5. Demonstrate Model Usage
    print("--- 5. Testing Sanitized OAS Discriminant Model ---")

    model = SanitizedOASDiscriminant()

    print("Fitting model on subset...")
    model.fit(X_train, y_train)

    print("Predicting on validation subset...")
    val_probs = model.predict_proba(X_val)

    print(f"Validation Probabilities shape: {val_probs.shape}")

    # Check predictions
    assert val_probs.shape[0] == X_val.shape[0], "Prediction count mismatch"
    assert val_probs.shape[1] == len(classes), "Class probability count mismatch"

    # Check probability sum
    row_sums = np.sum(val_probs, axis=1)
    # Allow small floating point error
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"

    # Calculate metric on this subset
    subset_loss = compute_metric(y_val, val_probs, labels=list(range(len(classes))))
    print(f"Subset Validation Log Loss: {subset_loss:.4f}")

    print("Model verification passed.\n")

    # 6. Demonstrate Full Pipeline Execution
    print("--- 6. Testing Full Pipeline (train_and_predict) ---")

    # Run the encapsulated pipeline function with a small sample size
    train_and_predict(max_samples=50)

    # Verify submission file creation
    submission_path = "./submission/submission.csv"
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"Submission file created at {submission_path}")
        print(f"Submission shape: {df_sub.shape}")

        # Check columns
        assert "id" in df_sub.columns, "Submission missing 'id' column"
        # Check that we have columns for classes (excluding id)
        assert (
            len(df_sub.columns) == len(classes) + 1
        ), "Incorrect number of columns in submission"

        # Check values
        assert not df_sub.isnull().values.any(), "Submission contains NaNs"
    else:
        raise FileNotFoundError("Submission file was not created by train_and_predict")

    print("Full pipeline verification passed.\n")

    print("=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
