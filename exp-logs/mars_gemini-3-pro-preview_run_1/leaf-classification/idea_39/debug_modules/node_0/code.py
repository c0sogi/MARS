import os
import sys
import numpy as np
import pandas as pd
import shutil

# Ensure the current directory is in the path for module imports
sys.path.append(os.getcwd())

# Import library components
from library.utils import set_seed, compute_log_loss, generate_config_hash
from library.config import GEOMETRIC_FEATURES, INPUT_DIR, METADATA_DIR, SUBMISSION_FILE
from library.features import extract_geometric_properties, batch_extract_features
from library.data import DataManager
from library.model import OASDiscriminant, train_and_evaluate


def demo_utils():
    """
    Demonstrates and validates utility functions.
    """
    print("\n=== Demo: Utilities ===")

    # 1. Test Config Hash
    config_a = {"param1": 10, "param2": "test"}
    config_b = {"param2": "test", "param1": 10}  # Different order
    hash_a = generate_config_hash(config_a)
    hash_b = generate_config_hash(config_b)

    print(f"Hash A: {hash_a}")
    assert (
        hash_a == hash_b
    ), "Hash should be deterministic regardless of dictionary key order"
    print("Config hash verification passed.")

    # 2. Test Log Loss Calculation
    # Scenario: 2 samples, 2 classes.
    # Row 1: True class 0. Preds [0.9, 0.1] -> Good
    # Row 2: True class 1. Preds [0.2, 0.8] -> Good
    y_true = np.array([0, 1])
    y_pred = np.array([[0.9, 0.1], [0.2, 0.8]])

    loss = compute_log_loss(y_true, y_pred)
    print(f"Computed Log Loss: {loss:.4f}")

    # Manual check: - (log(0.9) + log(0.8)) / 2
    expected_loss = -(np.log(0.9) + np.log(0.8)) / 2
    assert np.isclose(
        loss, expected_loss
    ), f"Log loss mismatch. Got {loss}, expected {expected_loss}"
    print("Log loss calculation verification passed.")


def demo_features():
    """
    Demonstrates geometric feature extraction.
    """
    print("\n=== Demo: Feature Extraction ===")

    # Load metadata to get a valid image path
    train_meta_path = os.path.join(METADATA_DIR, "train.csv")
    if not os.path.exists(train_meta_path):
        print("Metadata not found, skipping feature demo.")
        return

    df_train = pd.read_csv(train_meta_path)
    if df_train.empty:
        print("Metadata empty, skipping feature demo.")
        return

    # 1. Single Image Extraction
    sample_row = df_train.iloc[0]
    image_rel_path = sample_row["file_path"]
    full_image_path = os.path.join(INPUT_DIR, image_rel_path)

    print(f"Extracting features for: {full_image_path}")
    features = extract_geometric_properties(full_image_path)

    # Validation
    assert isinstance(features, dict), "Output should be a dictionary"
    assert all(
        k in features for k in GEOMETRIC_FEATURES
    ), "Missing geometric features in output"
    print("Single image feature extraction successful.")
    print(
        f"Sample features: Area={features['Area']}, Solidity={features['Solidity']:.4f}"
    )

    # 2. Batch Extraction
    print("Testing batch extraction on top 5 rows...")
    subset_df = df_train.head(5).copy()

    # We use load_cached_data=False to force execution of the extraction logic for demonstration
    df_batch = batch_extract_features(subset_df, INPUT_DIR, load_cached_data=False)

    assert len(df_batch) == 5, "Batch output length mismatch"
    assert "id" in df_batch.columns, "ID column missing in batch output"
    assert (
        df_batch.shape[1] == len(GEOMETRIC_FEATURES) + 1
    ), "Incorrect column count in batch output"
    print("Batch extraction verification passed.")


def demo_data_pipeline():
    """
    Demonstrates DataManager and Preprocessing.
    """
    print("\n=== Demo: Data Pipeline ===")

    dm = DataManager()

    # Load processed data (this triggers merging, cleaning, Yeo-Johnson, Scaling)
    # Using cached data if available for speed, logic handles recompute if needed
    print("Loading processed data via DataManager...")
    data = dm.get_processed_data(load_cached_data=True)

    X_train, y_train, ids_train, X_val, y_val, ids_val, X_test, ids_test, classes = data

    print(f"Train Shape: {X_train.shape}")
    print(f"Val Shape:   {X_val.shape}")
    print(f"Test Shape:  {X_test.shape}")
    print(f"Classes: {len(classes)}")

    # Validations
    assert not np.isnan(X_train).any(), "NaNs found in training data"
    assert not np.isinf(X_train).any(), "Infs found in training data"
    assert len(X_train) == len(y_train), "Mismatch between X and y lengths"
    assert X_train.dtype == np.float64, "Data should be float64 precision"

    print("Data pipeline verification passed.")
    return X_train, y_train, X_val, y_val, classes


def demo_model(X_train, y_train, X_val, y_val):
    """
    Demonstrates Model Training and Inference.
    """
    print("\n=== Demo: Model Training & Inference ===")

    model = OASDiscriminant()

    print("Fitting OAS Discriminant...")
    model.fit(X_train, y_train)

    print("Predicting on Validation set...")
    probs = model.predict_proba(X_val)

    # Validations
    assert probs.shape == (
        len(X_val),
        len(np.unique(y_train)),
    ), "Probability shape mismatch"

    # Check probability constraints
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1"
    assert (probs >= 0).all() and (
        probs <= 1
    ).all(), "Probabilities out of range [0, 1]"

    # Calculate Score
    score = compute_log_loss(y_val, probs)
    print(f"Validation Log Loss: {score:.5f}")
    print("Model verification passed.")


def demo_full_execution():
    """
    Runs the high-level train_and_evaluate function.
    """
    print("\n=== Demo: Full Execution Pipeline ===")

    # Run the main routine provided in library.model
    train_and_evaluate(load_cached_data=True)

    # Verify submission
    if os.path.exists(SUBMISSION_FILE):
        df_sub = pd.read_csv(SUBMISSION_FILE)
        print(f"Submission file generated at {SUBMISSION_FILE}")
        print(f"Submission shape: {df_sub.shape}")

        # Check format
        assert "id" in df_sub.columns, "Submission missing 'id' column"
        assert (
            df_sub.shape[0] == 99
        ), "Submission should have 99 rows (based on test set size)"
        print("Full execution pipeline verification passed.")
    else:
        raise FileNotFoundError("Submission file was not generated.")


if __name__ == "__main__":
    # 1. Set Seed
    set_seed(42)

    # 2. Run Demos
    try:
        demo_utils()
        demo_features()

        # Pass data from pipeline to model demo to avoid reloading
        X_train, y_train, X_val, y_val, classes = demo_data_pipeline()
        demo_model(X_train, y_train, X_val, y_val)

        demo_full_execution()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\nDEMO FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAN ERROR OCCURRED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
