import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss

# Filter warnings to keep output clean
warnings.filterwarnings("ignore")

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, format_submission
from library.feature_extraction import DualStreamExtractor
from library.data_manager import DataManager
from library.model_factory import create_pipeline


def main():
    print("Starting Leaf Classification Demo...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Set seed for reproducibility
    seed_everything(42)

    # Override Config parameters for a fast demonstration
    # We limit the dataset to 10 samples per split to ensure quick execution.
    Config.DEBUG_SAMPLE_SIZE = 10
    Config.WORKING_DIR = "./working/demo_execution"
    Config.BATCH_SIZE = 8  # Smaller batch size for the demo

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Configuration configured: Debug Sample Size = {Config.DEBUG_SAMPLE_SIZE}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # ==========================================
    # 2. Feature Extraction (DualStreamExtractor)
    # ==========================================
    print("\n[Step 1] Running DualStreamExtractor...")
    extractor = DualStreamExtractor()

    # Force extraction (load_cached_data=False) to demonstrate the logic
    # This will use the DEBUG_SAMPLE_SIZE to limit processing time
    raw_features = extractor.extract_features(load_cached_data=False)

    # Verification of Feature Extraction
    # Expected rows = N_samples * NUM_ROTATIONS (12)
    # We check the training split
    n_train_samples = min(Config.DEBUG_SAMPLE_SIZE, 712)  # 712 is full train size
    expected_rows = n_train_samples * Config.NUM_ROTATIONS

    dino_dim = 1024  # ViT-Large
    conv_dim = 1536  # ConvNeXt-Large

    print(
        f"Verifying extracted feature shapes for {n_train_samples} training samples..."
    )

    assert raw_features["train_dino"].shape == (
        n_train_samples,
        Config.NUM_ROTATIONS,
        dino_dim,
    ), f"Expected train_dino shape {(n_train_samples, Config.NUM_ROTATIONS, dino_dim)}, got {raw_features['train_dino'].shape}"
    assert raw_features["train_conv"].shape == (
        n_train_samples,
        Config.NUM_ROTATIONS,
        conv_dim,
    ), f"Expected train_conv shape {(n_train_samples, Config.NUM_ROTATIONS, conv_dim)}, got {raw_features['train_conv'].shape}"
    assert raw_features["train_ids"].shape == (
        n_train_samples,
    ), f"Expected train_ids shape {(n_train_samples,)}, got {raw_features['train_ids'].shape}"

    print("Feature extraction verified successfully.")

    # ==========================================
    # 3. Data Management (DataManager)
    # ==========================================
    print("\n[Step 2] Running DataManager (Densification & Structuring)...")
    manager = DataManager()

    # We can load cached data now because extract_features (called above)
    # saves to the same Config.WORKING_DIR
    data = manager.get_data(load_cached_data=True)

    train_X = data["train_X"]
    train_y = data["train_y"]
    test_X = data["test_X"]
    test_ids = data["test_ids"]

    # Verification of Data Structure
    # Train data should be densified: N_samples * CENTROIDS_PER_IMAGE (3)
    expected_train_rows = n_train_samples * Config.CENTROIDS_PER_IMAGE
    total_dim = dino_dim + conv_dim + 192  # 192 is tabular dim

    print(f"Verifying densified data shapes...")

    assert train_X.shape == (
        expected_train_rows,
        total_dim,
    ), f"Expected train_X shape {(expected_train_rows, total_dim)}, got {train_X.shape}"
    assert train_y.shape == (
        expected_train_rows,
    ), f"Expected train_y shape {(expected_train_rows,)}, got {train_y.shape}"

    # Test data should be structured: (N_samples, 3, D)
    n_test_samples = min(Config.DEBUG_SAMPLE_SIZE, 99)
    assert test_X.shape == (
        n_test_samples,
        Config.CENTROIDS_PER_IMAGE,
        total_dim,
    ), f"Expected test_X shape {(n_test_samples, Config.CENTROIDS_PER_IMAGE, total_dim)}, got {test_X.shape}"

    print("Data management verified successfully.")

    # ==========================================
    # 4. Model Training (ModelFactory)
    # ==========================================
    print("\n[Step 3] Building and Training Model Pipeline...")

    # Instantiate pipeline
    pipeline = create_pipeline(dino_dim=dino_dim, conv_dim=conv_dim, tabular_dim=192)

    # Fit pipeline
    # Note: With very few samples (10), PCA n_components=0.99 might result in very few components,
    # or raise an error if n_samples < n_components.
    # However, sklearn PCA handles n_components < min(n_samples, n_features) automatically.
    pipeline.fit(train_X, train_y)

    # Quick sanity check on training set
    train_probs = pipeline.predict_proba(train_X)
    loss = log_loss(train_y, train_probs)
    print(f"Model fitted. Training Log Loss: {loss:.4f}")

    # ==========================================
    # 5. Inference & Submission
    # ==========================================
    print("\n[Step 4] Inference and Submission Generation...")

    # Flatten structured test data for inference
    # (N, 3, D) -> (N*3, D)
    N_test, C, D = test_X.shape
    test_X_flat = test_X.reshape(N_test * C, D)

    # Predict probabilities
    probs_flat = pipeline.predict_proba(test_X_flat)

    # Reshape and average across centroids
    # (N*3, n_classes) -> (N, 3, n_classes) -> mean -> (N, n_classes)
    probs_reshaped = probs_flat.reshape(N_test, C, -1)
    probs_avg = np.mean(probs_reshaped, axis=1)

    # Get class names
    class_names = pipeline.classes_

    # Generate submission file
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    format_submission(test_ids, probs_avg, class_names, output_path=submission_path)

    # Verify submission file
    assert os.path.exists(submission_path), "Submission file was not created."

    sub_df = pd.read_csv(submission_path)
    print(f"Submission file created at: {submission_path}")
    print(f"Submission shape: {sub_df.shape}")

    # Check if ID column exists and count matches
    assert "id" in sub_df.columns, "ID column missing in submission."
    assert (
        len(sub_df) == n_test_samples
    ), f"Expected {n_test_samples} rows, got {len(sub_df)}"

    # Check probability range
    prob_cols = [c for c in sub_df.columns if c != "id"]
    probs = sub_df[prob_cols].values
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
