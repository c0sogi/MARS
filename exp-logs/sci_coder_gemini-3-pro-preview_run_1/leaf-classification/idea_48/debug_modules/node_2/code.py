import os
import sys
import numpy as np
import pandas as pd
import warnings
import shutil

# Import from the provided library files
from library.config import Config
from library.feature_extraction import extract_single_image_features, process_dataset
from library.preprocessing import HighPrecisionPipeline, get_preprocessed_data
from library.data_loader import load_and_process_data, set_seed
from library.oas_lda import OASDiscriminant, run_oas_strategy

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration of Leaf Identification Library ===\n")

    # 1. Setup and Configuration
    print("--- 1. Verifying Configuration ---")
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Setup directories
    Config.setup()

    # Verify paths
    assert os.path.exists(Config.INPUT_DIR), "Input directory missing"
    assert os.path.exists(Config.METADATA_DIR), "Metadata directory missing"
    print(f"Configured Input Dir: {Config.INPUT_DIR}")
    print(f"Configured Working Dir: {Config.WORKING_DIR}")
    print("Configuration verified.\n")

    # 2. Feature Extraction Demonstration
    print("--- 2. Demonstrating Feature Extraction ---")

    # Load train metadata to get a valid image path
    df_train_meta = pd.read_csv(Config.TRAIN_DATA_PATH)
    sample_row = df_train_meta.iloc[0]
    sample_image_path = os.path.join(Config.INPUT_DIR, sample_row["file_path"])

    print(f"Extracting features for image: {sample_row['file_path']}")

    # Test single image extraction
    features = extract_single_image_features(sample_image_path)

    # Validation
    assert isinstance(features, dict), "Output must be a dictionary"
    assert "area" in features, "Feature 'area' missing"
    assert "eccentricity" in features, "Feature 'eccentricity' missing"
    assert isinstance(features["area"], float), "Feature values must be floats"

    print(f"Extracted {len(features)} geometric features.")
    print(
        f"Sample features: Area={features['area']:.2f}, Eccentricity={features['eccentricity']:.4f}"
    )

    # Test dataset processing (Debug mode for speed)
    print("Processing a small subset of the training dataset (Geometric Features)...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20

    # We force load_cached_data=False to demonstrate computation
    df_geo_debug = process_dataset(
        Config.TRAIN_DATA_PATH, "train_debug", load_cached_data=False
    )

    assert (
        len(df_geo_debug) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} rows, got {len(df_geo_debug)}"
    assert "id" in df_geo_debug.columns, "ID column missing in geometric dataframe"
    print("Batch feature extraction successful.\n")

    # 3. Preprocessing Pipeline Demonstration
    print("--- 3. Demonstrating High-Precision Preprocessing ---")

    # Create dummy data
    X_dummy = np.random.rand(100, 10).astype(np.float64)
    # Introduce some scale differences to test standardization
    X_dummy[:, 0] = X_dummy[:, 0] * 1000

    pipeline = HighPrecisionPipeline()

    # Fit and Transform
    X_transformed = pipeline.fit_transform(X_dummy)

    # Validation
    assert X_transformed.shape == X_dummy.shape, "Shape mismatch after transformation"
    assert X_transformed.dtype == np.float64, "Pipeline must preserve float64 precision"

    # Check standardization (mean approx 0, std approx 1)
    mean_check = np.mean(X_transformed, axis=0)
    std_check = np.std(X_transformed, axis=0)

    assert np.allclose(mean_check, 0, atol=1e-1), "Transformed data not centered"
    assert np.allclose(std_check, 1, atol=1e-1), "Transformed data not scaled"

    print("HighPrecisionPipeline verified (Yeo-Johnson + StandardScaler).\n")

    # 4. Data Loader Integration
    print("--- 4. Demonstrating Data Loader ---")

    # Load data in debug mode
    # This integrates metadata loading, geometric extraction, merging, and preprocessing
    print("Loading and processing data (Debug Mode)...")
    data = load_and_process_data(
        load_cached_data=False, debug=True, debug_sample_size=50
    )
    X_train, y_train, X_val, y_val, X_test, test_ids, classes = data

    # Validation
    assert X_train.dtype == np.float64, "X_train must be float64"
    assert len(X_train) == 50, "X_train size mismatch for debug mode"
    assert len(y_train) == 50, "y_train size mismatch"
    assert len(X_train) == len(y_train), "X and y dimension mismatch"
    assert len(classes) > 0, "Classes array is empty"

    print(f"Data Loaded: X_train shape {X_train.shape}, n_classes={len(classes)}")
    print("Data Loader integration successful.\n")

    # 5. Model Training (OAS Discriminant)
    print("--- 5. Demonstrating OAS Discriminant Model ---")

    model = OASDiscriminant()

    print("Fitting OAS model...")
    model.fit(X_train, y_train)

    print("Predicting on validation set...")
    probs = model.predict_proba(X_val)

    # Validation
    assert probs.shape == (
        len(X_val),
        len(classes),
    ), "Probability matrix shape mismatch"
    # Check if probabilities sum to 1
    row_sums = np.sum(probs, axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9), "Probabilities do not sum to 1"

    print("Model training and prediction successful.")
    print(f"Prediction shape: {probs.shape}\n")

    # 6. End-to-End Strategy Execution
    print("--- 6. Running Full Strategy (End-to-End) ---")

    # Run the high-level function provided in oas_lda.py
    # We use a slightly larger sample size to ensure we hit the test set logic correctly
    run_oas_strategy(load_cached_data=False, debug=True, debug_sample_size=60)

    # Verify Submission
    submission_path = Config.SUBMISSION_FILE
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file loaded. Shape: {df_sub.shape}")

    # Check format
    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert (
        len(df_sub.columns) == len(classes) + 1
    ), "Incorrect number of columns in submission"

    # Check values
    # Probabilities should be between 0 and 1
    feature_cols = [c for c in df_sub.columns if c != "id"]
    vals = df_sub[feature_cols].values
    assert np.all(vals >= 0) and np.all(vals <= 1), "Probabilities out of bounds [0, 1]"

    print("End-to-End execution successful. Submission generated.\n")

    print("=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
