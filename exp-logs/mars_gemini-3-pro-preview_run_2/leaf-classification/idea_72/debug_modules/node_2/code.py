import os
import numpy as np
import pandas as pd
import cv2
import shutil
import sys
import importlib

# Import library modules
import library.config
import library.dataset
import library.image_features
import library.modeling
import library.execution

# Reload modules to ensure config changes are picked up in persistent environments
importlib.reload(library.config)
importlib.reload(library.image_features)
importlib.reload(library.dataset)
importlib.reload(library.modeling)
importlib.reload(library.execution)

# Set fixed seeds for reproducibility
np.random.seed(42)
library.config.RANDOM_SEED = 42


def setup_demo_environment():
    """
    Sets up a temporary directory for the demo and patches the library
    configuration to use it. This prevents conflicts with the main
    working directory.
    """
    demo_dir = "./working/demo_run"
    cache_dir = os.path.join(demo_dir, "cache")
    submission_dir = os.path.join(demo_dir, "submission")

    # Clean up previous run if exists
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)

    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    submission_path = os.path.join(submission_dir, "submission.csv")

    # Patch library.config
    library.config.WORKING_DIR = cache_dir
    library.config.SUBMISSION_DIR = submission_dir
    library.config.SUBMISSION_PATH = submission_path

    # Patch modules that imported constants directly
    library.dataset.WORKING_DIR = cache_dir
    library.image_features.WORKING_DIR = cache_dir
    library.modeling.SUBMISSION_PATH = submission_path
    library.execution.SUBMISSION_PATH = submission_path

    print(f"Demo environment set up at: {demo_dir}")
    return submission_path


def test_image_features():
    """
    Verifies the logic in library.image_features.
    Creates a synthetic image and checks feature extraction.
    """
    print("\n=== Testing Image Features ===")

    # Create a synthetic binary image (100x100)
    # Background: Black (0), Leaf: White (255)
    img = np.zeros((100, 100), dtype=np.uint8)
    # Draw a filled rectangle to simulate a leaf
    cv2.rectangle(img, (30, 20), (70, 80), 255, -1)

    # Test Polarity Correction
    # Case 1: Already correct (leaf is white, background is black)
    corrected = library.image_features.correct_polarity(img)
    # Corner should be black (0)
    assert (
        corrected[0, 0] == 0
    ), "Polarity correction failed: Expected black background."

    # Case 2: Inverted (leaf is black, background is white)
    img_inv = cv2.bitwise_not(img)
    corrected_inv = library.image_features.correct_polarity(img_inv)
    # Should be inverted back to black background
    assert (
        corrected_inv[0, 0] == 0
    ), "Polarity correction failed: Should invert white background."

    # Test Morphometrics Extraction
    features = library.image_features.extract_morphometrics(corrected)

    # Expected Aspect Ratio: width=41 (inclusive), height=61 -> ~0.67
    # Allow some tolerance for CV2 contour approximation
    ar = features["aspect_ratio"]
    print(f"Extracted Aspect Ratio: {ar:.4f}")

    assert (
        0.5 < ar < 0.8
    ), f"Aspect Ratio {ar} out of expected range for synthetic rectangle."
    assert features["solidity"] > 0.9, "Solidity should be high for a rectangle."
    assert "hu_1" in features, "Hu moments missing."

    print("Image features verification passed.")


def test_data_pipeline_and_execution():
    """
    Runs the full data loading, training, and inference pipeline
    using the ExperimentRunner.
    """
    print("\n=== Testing Data Pipeline & Execution ===")

    # Instantiate Runner
    # We set load_cached_data=False to force the processing logic to run
    runner = library.execution.ExperimentRunner(load_cached_data=False)

    # 1. Load Data
    print("Loading data...")
    runner.load_data()

    # Verify Data Integrity
    assert runner.data_train is not None, "Training data not loaded."
    assert runner.data_test is not None, "Test data not loaded."

    # Check Feature Views
    # Global should have 192 features (64*3)
    n_global_feats = runner.data_train["global"].shape[1]
    assert n_global_feats == 192, f"Expected 192 global features, got {n_global_feats}"

    # Check Morphometrics
    # Should have 11 features
    n_morph_feats = runner.data_train["morph"].shape[1]
    assert n_morph_feats == 11, f"Expected 11 morph features, got {n_morph_feats}"

    print(
        f"Data Loaded: Train {runner.data_train['global'].shape}, Test {runner.data_test['global'].shape}"
    )

    # 2. Run Selection Phase (Phase 1)
    # We subsample training data to 200 samples to ensure the demo runs quickly (< 19 mins)
    # The full dataset is small (~900), but 200 is enough to verify the pipeline mechanics.
    print("\nRunning Selection Phase (Subsampled)...")
    runner.run_selection_phase(max_train_samples=200)

    # Verify that experts were selected
    selected = runner.trainer.selected_experts
    assert len(selected) > 0, "No experts were selected during the selection phase."
    print(f"Selected Experts: {list(selected.keys())}")

    # 3. Run Final Inference (Phase 2)
    print("\nRunning Final Inference...")
    submission_df = runner.run_final_inference()

    return submission_df


def verify_submission(df, path):
    """
    Validates the generated submission file.
    """
    print("\n=== Verifying Submission ===")

    # Check file existence
    assert os.path.exists(path), "Submission file was not saved to disk."

    # Check Dimensions
    # Test set has 99 rows (from dataset info)
    # Columns: id + 99 species
    expected_rows = 99
    expected_cols = 100  # id + 99 classes

    assert (
        df.shape[0] == expected_rows
    ), f"Submission has {df.shape[0]} rows, expected {expected_rows}."
    assert (
        df.shape[1] == expected_cols
    ), f"Submission has {df.shape[1]} columns, expected {expected_cols}."

    # Check Probability Range
    feature_cols = [c for c in df.columns if c != "id"]
    probs = df[feature_cols].values

    assert np.all(probs >= 0.0) and np.all(
        probs <= 1.0
    ), "Probabilities out of [0, 1] range."

    # Check IDs
    assert "id" in df.columns, "ID column missing."
    assert df["id"].dtype in [np.int64, int], "ID column should be integer."

    print("Submission verification passed.")


if __name__ == "__main__":
    try:
        # 1. Setup
        sub_path = setup_demo_environment()

        # 2. Unit Test: Image Features
        test_image_features()

        # 3. Integration Test: Pipeline
        df_submission = test_data_pipeline_and_execution()

        # 4. Validation
        verify_submission(df_submission, sub_path)

        print("\nSUCCESS: All demonstrations and verifications completed.")

    except AssertionError as e:
        print(f"\nFAILURE: Assertion failed - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nFAILURE: An unexpected error occurred - {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
