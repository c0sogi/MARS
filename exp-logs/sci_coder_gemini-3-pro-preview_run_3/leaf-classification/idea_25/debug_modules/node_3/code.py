import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data_processing import DensificationManager
from library.execution import ModelExecutor


def create_demo_metadata(num_train=20, num_val=10, num_test=10):
    """
    Creates a small subset of the metadata to allow for rapid execution
    of the pipeline for demonstration purposes.
    """
    print(
        f"Creating demo metadata (Train={num_train}, Val={num_val}, Test={num_test})..."
    )

    # Create demo directory
    demo_dir = "./working/demo_metadata"
    os.makedirs(demo_dir, exist_ok=True)

    # Load original metadata
    train_full = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_full = pd.read_csv(Config.VAL_METADATA_PATH)
    test_full = pd.read_csv(Config.TEST_METADATA_PATH)

    # Sample subsets
    # We use the head to ensure we get existing images (assuming sorted/sequential might help,
    # but random sample with seed is also fine. Head is safer if images are missing,
    # though verification script said 0 missing).

    # Construct train_subset to satisfy StratifiedKFold requirements (min 2 samples per class)
    # Cite debug_lesson_12: Preserve Statistical Invariants in Debug Subsets
    train_subset = pd.DataFrame()
    for _, group in train_full.groupby("species"):
        if len(train_subset) >= num_train:
            break
        # Take 4 samples per class to ensure > 2 for 2-fold CV
        train_subset = pd.concat([train_subset, group.head(4)])

    val_subset = val_full.head(num_val).copy()
    test_subset = test_full.head(num_test).copy()

    # Save subsets
    demo_train_path = os.path.join(demo_dir, "train.csv")
    demo_val_path = os.path.join(demo_dir, "val.csv")
    demo_test_path = os.path.join(demo_dir, "test.csv")

    train_subset.to_csv(demo_train_path, index=False)
    val_subset.to_csv(demo_val_path, index=False)
    test_subset.to_csv(demo_test_path, index=False)

    return demo_train_path, demo_val_path, demo_test_path


def verify_densification_logic():
    """
    Unit test to verify the Manifold Densification logic (Centroid computation).
    """
    print("\nVerifying Densification Logic...")
    manager = DensificationManager()

    # Create dummy data: 1 Sample, 12 Views, 4 Dimensions
    # Views 0, 3, 6, 9 belong to Centroid A
    # Views 1, 4, 7, 10 belong to Centroid B
    # Views 2, 5, 8, 11 belong to Centroid C
    N, V, D = 1, 12, 4
    dummy_features = np.zeros((N, V, D))

    # Fill with specific values to check averaging
    # Centroid A indices: [0, 3, 6, 9] -> Set to 1.0
    for idx in [0, 3, 6, 9]:
        dummy_features[0, idx, :] = 1.0

    # Centroid B indices: [1, 4, 7, 10] -> Set to 2.0
    for idx in [1, 4, 7, 10]:
        dummy_features[0, idx, :] = 2.0

    # Centroid C indices: [2, 5, 8, 11] -> Set to 3.0
    for idx in [2, 5, 8, 11]:
        dummy_features[0, idx, :] = 3.0

    # Compute centroids
    centroids = manager._compute_centroids(dummy_features)

    # Expected Shape: (1, 3, 4)
    assert centroids.shape == (1, 3, 4), f"Incorrect shape: {centroids.shape}"

    # Verify values
    # Centroid 0 (A) should be mean of 1.0s -> 1.0
    assert np.allclose(centroids[0, 0, :], 1.0), "Centroid A computation incorrect"
    # Centroid 1 (B) should be mean of 2.0s -> 2.0
    assert np.allclose(centroids[0, 1, :], 2.0), "Centroid B computation incorrect"
    # Centroid 2 (C) should be mean of 3.0s -> 3.0
    assert np.allclose(centroids[0, 2, :], 3.0), "Centroid C computation incorrect"

    print("Densification logic verified successfully.")


def run_demo_pipeline():
    """
    Executes the full pipeline using the ModelExecutor on the demo dataset.
    """
    print("\nStarting Demo Pipeline Execution...")

    # 1. Initialize Executor
    executor = ModelExecutor()

    # 2. Train Ensemble
    # load_cached_data=False forces the system to process our new demo metadata
    # instead of looking for existing cache files in the working directory.
    print("Training Ensemble...")
    pipelines = executor.train_ensemble(load_cached_data=False)

    # Validation: Check if we got the expected number of models (N_FOLDS=2)
    assert (
        len(pipelines) == Config.N_FOLDS
    ), f"Expected {Config.N_FOLDS} pipelines, got {len(pipelines)}"

    # 3. Generate Submission
    print("Generating Submission...")
    submission_df = executor.generate_submission(pipelines, load_cached_data=False)

    # Validation: Check submission shape
    # We used 5 test samples in configuration
    expected_rows = 5
    # ID column + 99 species columns = 100 columns
    expected_cols = 100

    assert (
        len(submission_df) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(submission_df)}"
    assert (
        submission_df.shape[1] == expected_cols
    ), f"Expected {expected_cols} columns in submission, got {submission_df.shape[1]}"

    # Validation: Check probability constraints
    # Drop 'id' column
    probs = submission_df.drop(columns=["id"]).values
    assert np.all(probs >= 0) and np.all(
        probs <= 1
    ), "Probabilities out of range [0, 1]"

    print("\nPipeline execution successful.")
    print("Sample Submission:")
    print(submission_df.head())


if __name__ == "__main__":
    # ==========================================
    # 1. Global Setup
    # ==========================================
    seed_everything(42)

    # ==========================================
    # 2. Configure for Demo (Optimization)
    # ==========================================
    # We modify the Config class attributes directly to affect the library behavior.

    # Create subset metadata to speed up feature extraction
    # Using very small numbers to ensure it finishes in < 5 mins
    train_path, val_path, test_path = create_demo_metadata(
        num_train=12, num_val=6, num_test=5
    )

    Config.TRAIN_METADATA_PATH = train_path
    Config.VAL_METADATA_PATH = val_path
    Config.TEST_METADATA_PATH = test_path

    # Use a specific working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Reduce folds for speed
    Config.N_FOLDS = 2

    # Reduce batch size if necessary (though 32 is fine for A100)
    Config.BATCH_SIZE_EXTRACTION = 8

    # Disable multiprocessing for data loading in demo script to avoid overhead/complexity
    Config.NUM_WORKERS = 0

    # ==========================================
    # 3. Logic Verification
    # ==========================================
    verify_densification_logic()

    # ==========================================
    # 4. Run Pipeline
    # ==========================================
    # This will:
    # 1. Extract features using DINOv2 and ConvNeXt for the subset
    # 2. Densify the features (12 views -> 3 centroids)
    # 3. Train LDA ensemble
    # 4. Predict on test set
    run_demo_pipeline()

    print("\nDemo completed successfully.")
