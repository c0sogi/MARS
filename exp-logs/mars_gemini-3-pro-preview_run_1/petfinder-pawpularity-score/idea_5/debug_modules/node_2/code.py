import sys
import os
import shutil
import numpy as np
import pandas as pd
import torch
import logging

# Ensure library is in path
sys.path.append(".")

from library.config import Config
from library.utils import seed_everything, setup_logger
from library.dataset import PawpularityDataset
from library.feature_extraction import extract_features
from library.stacking import RidgeStacker


def run_demo():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print(">>> Step 1: Configuring environment for demo...")

    # Set seeds for reproducibility
    seed_everything(42)

    # Modify Config for fast execution (Monkey-patching)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10  # Only process 10 images
    Config.N_FOLDS = 3  # Reduce folds for speed
    Config.WORKING_DIR = "./working/demo_output"
    Config.SUBMISSION_PATH = "./working/demo_submission.csv"

    # Ensure working directory is clean
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR)

    # Save original model config to restore later
    ORIGINAL_MODELS = Config.MODELS.copy()

    # For the Feature Extraction demo, we restrict to just one model
    # to save time and memory (avoiding loading 3 large backbones).
    # 'convnext' is selected as a representative CNN.
    DEMO_MODEL_KEY = "convnext"
    Config.MODELS = {DEMO_MODEL_KEY: ORIGINAL_MODELS[DEMO_MODEL_KEY]}

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"Active Model for Extraction Demo: {list(Config.MODELS.keys())}")

    # =========================================================================
    # 2. Dataset Demonstration
    # =========================================================================
    print("\n>>> Step 2: Verifying Dataset Logic...")

    # Instantiate Dataset in 'train' mode
    dataset = PawpularityDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        model_name=DEMO_MODEL_KEY,
        mode="train",
    )

    # Verify subsetting worked
    print(f"Dataset length (subset): {len(dataset)}")
    assert (
        len(dataset) == Config.DEBUG_SAMPLE_SIZE
    ), f"Dataset length {len(dataset)} does not match DEBUG_SAMPLE_SIZE {Config.DEBUG_SAMPLE_SIZE}"

    # Fetch one sample to verify structure
    img, meta, target, sample_id = dataset[0]

    print(f"Sample 0 ID: {sample_id}")
    print(f"Image Shape: {img.shape}")
    print(f"Meta Shape: {meta.shape}")
    print(f"Target: {target}")

    # Assertions
    assert isinstance(img, torch.Tensor)
    # ConvNeXt target size is 224x224
    assert img.shape == (3, 224, 224), f"Unexpected image shape: {img.shape}"
    # There are 12 binary metadata features
    assert meta.shape == (12,), f"Unexpected metadata shape: {meta.shape}"
    assert isinstance(target, torch.Tensor)

    # =========================================================================
    # 3. Feature Extraction Demonstration
    # =========================================================================
    print("\n>>> Step 3: Verifying Feature Extraction (Real Model)...")

    # Run extraction using the real model on the small subset
    # This tests ModelFactory, DataLoader, and caching logic
    features, meta_feats, targets, ids = extract_features(
        model_name=DEMO_MODEL_KEY,
        metadata_path=Config.TRAIN_METADATA_PATH,
        mode="train",
        load_cached_data=False,  # Force re-compute to prove it works
        device=Config.DEVICE,
    )

    print(f"Extracted Features Shape: {features.shape}")

    # Assertions
    expected_dim = ORIGINAL_MODELS[DEMO_MODEL_KEY]["output_dim"]
    assert features.shape == (Config.DEBUG_SAMPLE_SIZE, expected_dim)
    assert meta_feats.shape == (Config.DEBUG_SAMPLE_SIZE, 12)
    assert len(ids) == Config.DEBUG_SAMPLE_SIZE

    # Check that cache files were created
    cache_file = os.path.join(
        Config.WORKING_DIR, f"{DEMO_MODEL_KEY}_train_features.npy"
    )
    assert os.path.exists(cache_file), "Feature cache file was not created"
    print("Feature extraction and caching successful.")

    # =========================================================================
    # 4. Stacking Logic Demonstration (Synthetic Data)
    # =========================================================================
    print("\n>>> Step 4: Verifying Stacking Logic (Synthetic Data)...")

    # Restore full model configuration to demonstrate multi-expert stacking
    Config.MODELS = ORIGINAL_MODELS
    expert_names = list(Config.MODELS.keys())
    print(f"Stacking Experts: {expert_names}")

    # Generate synthetic data to simulate inputs from all 3 experts
    # This avoids the time cost of running inference on CLIP and DINOv2
    n_samples = 50  # Sufficient for CV split
    n_features_meta = 12

    synthetic_features_map = {}
    for name in expert_names:
        dim = Config.MODELS[name]["output_dim"]
        # Create random embeddings
        synthetic_features_map[name] = np.random.randn(n_samples, dim).astype(
            np.float32
        )

    synthetic_meta = np.random.randint(0, 2, (n_samples, n_features_meta)).astype(
        np.float32
    )
    synthetic_targets = np.random.uniform(10, 90, (n_samples,)).astype(np.float32)

    # Initialize Stacker
    # It will pick up the restored Config.MODELS keys
    stacker = RidgeStacker()

    # A. Fit Level-0 (Experts)
    print("Fitting Level-0 Experts with 3-Fold CV...")
    oof_df = stacker.fit_level0(
        synthetic_features_map, synthetic_meta, synthetic_targets
    )

    print("OOF Predictions Head:")
    print(oof_df.head())

    # Verify OOF structure
    assert oof_df.shape == (n_samples, len(expert_names))
    assert not oof_df.isnull().values.any(), "OOF predictions contain NaNs"

    # B. Fit Level-1 (Meta-Learner)
    print("Fitting Level-1 Meta-Learner...")
    stacker.fit_level1(oof_df, synthetic_targets)

    assert stacker.level1_model is not None, "Level-1 model was not trained"

    # C. Predict (Inference)
    print("Running Inference on synthetic test data...")

    # Generate synthetic test data
    n_test = 5
    test_features_map = {}
    for name in expert_names:
        dim = Config.MODELS[name]["output_dim"]
        test_features_map[name] = np.random.randn(n_test, dim).astype(np.float32)

    test_meta = np.random.randint(0, 2, (n_test, n_features_meta)).astype(np.float32)

    final_preds = stacker.predict(test_features_map, test_meta)

    print(f"Final Predictions: {final_preds}")

    # Verify predictions
    assert final_preds.shape == (n_test,)
    # Check clipping logic (1-100)
    assert (final_preds >= 1.0).all() and (
        final_preds <= 100.0
    ).all(), "Predictions contain values outside the valid range [1, 100]"

    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
