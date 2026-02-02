import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, calculate_log_loss
from library.feature_extraction import FeatureExtractor
from library.data_manager import DataManager
from library.modeling import StratifiedSelectiveTopologyModel


def run_demo():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    print("--- 1. Configuration Setup ---")

    # Override Config for a fast demo run
    Config.DEBUG_SAMPLE_LIMIT = 10  # Process only 10 images
    Config.N_SPLITS = 2  # 2-Fold CV for speed
    Config.WORKING_DIR = "./working/demo_run/working"
    Config.SUBMISSION_DIR = "./working/demo_run/submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Initialize environment
    seed_everything(Config.SEED)
    Config.setup()

    print(f"Debug Sample Limit: {Config.DEBUG_SAMPLE_LIMIT}")
    print(f"Folds: {Config.N_SPLITS}")
    print(f"Device: {Config.DEVICE}")

    # ==========================================
    # 2. Feature Extraction
    # ==========================================
    print("\n--- 2. Feature Extraction (DINOv2 + ConvNeXt) ---")

    extractor = FeatureExtractor()

    # Extract features for the training set (limited by DEBUG_SAMPLE_LIMIT)
    # We force re-computation (load_cached_data=False) to demonstrate the logic
    raw_train_ids, raw_train_dino, raw_train_conv = extractor.extract_features(
        metadata_path=Config.TRAIN_METADATA,
        cache_prefix="demo_train",
        load_cached_data=False,
    )

    # Validation: Check shapes
    # Expected: (N_samples, 12_views, Feature_Dim)
    n_samples = len(raw_train_ids)
    print(f"Extracted samples: {n_samples}")

    assert (
        n_samples == Config.DEBUG_SAMPLE_LIMIT
    ), f"Expected {Config.DEBUG_SAMPLE_LIMIT} samples, got {n_samples}"

    assert raw_train_dino.shape == (
        n_samples,
        12,
        1024,
    ), f"DINO shape mismatch: {raw_train_dino.shape}"

    assert raw_train_conv.shape == (
        n_samples,
        12,
        1536,
    ), f"ConvNeXt shape mismatch: {raw_train_conv.shape}"

    print("Feature Extraction Validation Passed.")

    # ==========================================
    # 3. Data Management (Manifold Densification)
    # ==========================================
    print("\n--- 3. Data Densification ---")

    manager = DataManager()

    # Pack raw data into tuple expected by manager
    raw_data_tuple = (raw_train_ids, raw_train_dino, raw_train_conv)

    # Create densified dataset
    # This aggregates 12 views into 3 centroids -> 3x data expansion
    densified_data = manager.create_densified_dataset(
        split_name="demo_train",
        raw_data=raw_data_tuple,
        metadata_path=Config.TRAIN_METADATA,
        load_cached_data=False,
    )

    # Validation: Check expansion logic
    # Expected size: 3 * n_samples
    expected_size = n_samples * 3

    assert (
        len(densified_data["ids"]) == expected_size
    ), f"Densified IDs length mismatch. Expected {expected_size}, got {len(densified_data['ids'])}"

    assert densified_data["dino"].shape == (
        expected_size,
        1024,
    ), f"Densified DINO shape mismatch: {densified_data['dino'].shape}"

    assert densified_data["tabular"].shape == (
        expected_size,
        192,
    ), f"Densified Tabular shape mismatch: {densified_data['tabular'].shape}"

    # Check that IDs are tiled correctly (A, B, C blocks)
    # The first N IDs should match the second N IDs
    assert np.array_equal(
        densified_data["ids"][:n_samples],
        densified_data["ids"][n_samples : 2 * n_samples],
    ), "ID tiling logic incorrect."

    print("Data Densification Validation Passed.")

    # ==========================================
    # 4. Modeling (Stratified Selective-Topology)
    # ==========================================
    print("\n--- 4. Model Training (Ensemble LDA) ---")

    model = StratifiedSelectiveTopologyModel()

    # Train the model
    # Note: With only 10 samples and many classes, stratification might warn or adjust,
    # but the code should run.
    model.fit(train_data=densified_data)

    # Check if models were saved
    model_files = os.listdir(os.path.join(Config.WORKING_DIR, "models"))
    print(f"Saved model files: {model_files}")
    assert (
        len(model_files) >= Config.N_SPLITS + 1
    ), "Not all model folds or classes were saved."  # +1 for classes.pkl

    print("Model Training Validation Passed.")

    # ==========================================
    # 5. Inference & Submission
    # ==========================================
    print("\n--- 5. Inference and Submission Generation ---")

    # We use the densified training data as a mock test set for demonstration
    # In a real scenario, this would be densified test data

    # Predict probabilities
    probs, class_names = model.predict_proba(densified_data)

    # Validation: Check prediction shape
    # Should return aggregated predictions for unique samples (N_samples), not densified (3*N)
    assert probs.shape == (
        n_samples,
        len(class_names),
    ), f"Prediction shape mismatch. Expected ({n_samples}, {len(class_names)}), got {probs.shape}"

    # Validation: Check probability properties
    # Row sums should be approximately 1.0 (LDA output)
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Probabilities do not sum to 1."

    print(f"Generated predictions for {len(probs)} unique samples.")

    # Save submission
    model.predict_and_save(densified_data, output_path=Config.SUBMISSION_PATH)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Shape: {df_sub.shape}")
    print(f"Columns: {df_sub.columns[:5].tolist()} ...")

    assert df_sub.shape[0] == n_samples, "Submission row count mismatch."
    assert "id" in df_sub.columns, "Submission missing 'id' column."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
