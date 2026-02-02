import os
import sys
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, ensure_float64, save_to_cache, load_from_cache
from library.morphology import extract_morphological_features
from library.feature_manager import FeatureManager
from library.preprocessor import RobustPipeline
from library.data_loader import DataLoader
from library.model import OASDiscriminant


def demo_utils():
    print("\n--- 1. Demonstrating Utilities ---")

    # Test Precision Enforcement
    data_int = np.array([1, 2, 3], dtype=int)
    data_float = ensure_float64(data_int)
    assert data_float.dtype == np.float64, "ensure_float64 failed to convert to float64"
    print("Precision enforcement verified: int array converted to float64.")

    # Test Caching Mechanism
    # We use a temporary filename to avoid conflicts
    dummy_data = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    cache_name = "demo_cache_test"

    # Save
    save_to_cache(cache_name, dummy_data)
    expected_path = os.path.join(Config.CACHE_DIR, cache_name + ".parquet")
    assert os.path.exists(expected_path), f"Cache file not created at {expected_path}"

    # Load
    loaded_data = load_from_cache(cache_name, expected_type="dataframe")
    assert loaded_data is not None, "Failed to load data from cache"
    assert loaded_data.equals(dummy_data), "Loaded data does not match saved data"
    print(
        f"Caching mechanism verified: Data saved to and loaded from {Config.CACHE_DIR}"
    )


def demo_morphology_extraction():
    print("\n--- 2. Demonstrating Morphology Extraction (Subset) ---")

    # Load metadata to get file paths
    if not os.path.exists(Config.TRAIN_DATA_PATH):
        raise FileNotFoundError(f"Train metadata not found at {Config.TRAIN_DATA_PATH}")

    df_meta = pd.read_csv(Config.TRAIN_DATA_PATH)

    # Use a small subset for demonstration speed
    subset_size = 5
    df_subset = df_meta.head(subset_size).copy()

    print(f"Extracting features for {subset_size} images...")

    # Extract features using a unique dataset name to avoid overwriting main cache
    # This calls the OpenCV logic in library/morphology.py
    features_df = extract_morphological_features(
        df_subset,
        dataset_name="demo_subset",
        load_cached_data=False,  # Force re-computation for demo
    )

    # Validation
    assert len(features_df) == subset_size, "Feature DF length mismatch"
    assert all(
        col in features_df.columns for col in Config.MORPHOLOGICAL_FEATURES
    ), "Missing morphological columns"
    assert features_df["Area"].dtype == np.float64, "Features are not float64"

    print("Morphology extraction verified.")
    print(f"Sample features:\n{features_df.iloc[0][:3]}")


def demo_full_pipeline_and_model():
    print("\n--- 3. Demonstrating Full Data Pipeline & Modeling ---")

    # 1. Data Loading
    # DataLoader orchestrates FeatureManager (merge) and RobustPipeline (transform)
    loader = DataLoader()

    print("Loading and preprocessing full dataset (Train/Val/Test)...")
    X_train, y_train, X_val, y_val, X_test, test_ids = loader.load_data(
        load_cached_data=True
    )

    # Verify Data Shapes and Types
    print(f"Training Data: {X_train.shape}, Type: {X_train.dtype}")
    print(f"Validation Data: {X_val.shape}, Type: {X_val.dtype}")
    print(f"Test Data: {X_test.shape}, Type: {X_test.dtype}")

    assert X_train.dtype == np.float64
    assert not np.isnan(X_train).any(), "NaNs found in training data"

    # 2. Model Initialization
    print("\nInitializing OAS Discriminant Model...")
    model = OASDiscriminant(assume_centered=True)

    # 3. Training
    print("Fitting model...")
    model.fit(X_train, y_train)

    assert model.W_ is not None, "Model weights not initialized"
    assert model.b_ is not None, "Model bias not initialized"
    print(f"Model fitted. Classes: {len(model.classes_)}")

    # 4. Prediction (Validation Set)
    print("Predicting on validation set...")
    val_probs = model.predict_proba(X_val)
    val_preds = model.predict(X_val)

    # Verify Probabilities
    # Rows should sum to 1.0
    row_sums = np.sum(val_probs, axis=1)
    assert np.allclose(row_sums, 1.0), "Probabilities do not sum to 1.0"

    # Calculate Accuracy
    acc = np.mean(val_preds == y_val)
    print(f"Validation Accuracy: {acc:.4f}")

    # 5. Prediction (Test Set)
    print("Predicting on test set...")
    test_probs = model.predict_proba(X_test)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(test_probs, columns=model.classes_)
    submission_df.insert(0, "id", test_ids)

    print(f"Submission DataFrame created with shape: {submission_df.shape}")
    print("Head of submission:")
    print(submission_df.head())


def main():
    # Set global seed for reproducibility
    set_seed(Config.SEED)

    print(f"Starting Demo. Working Directory: {os.getcwd()}")
    print(f"Input Directory: {Config.INPUT_DIR}")

    try:
        demo_utils()
        demo_morphology_extraction()
        demo_full_pipeline_and_model()
        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\nDEMO FAILED: Assertion Error - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nDEMO FAILED: Exception - {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
