import os
import sys
import numpy as np
import pandas as pd
import torch
import glob

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.data_loader import LeafDataset, load_tabular_data
from library.training_manager import train_bagging_ensemble, predict_ensemble


def run_demonstration():
    print("============================================================")
    print("  Leaf Classification: Library Demonstration Script")
    print("============================================================")

    # 1. Setup
    seed_everything(Config.SEED)
    print(f"Device detected: {Config.DEVICE}")

    # Ensure working directory exists for artifacts
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ============================================================
    # Part 1: Verify Data Loading Components
    # ============================================================
    print("\n[Part 1] Verifying Data Loading Components...")

    # A. Test Tabular Data Loading
    print("  Testing load_tabular_data (Debug Mode)...")
    ids, X_tab, y_tab = load_tabular_data(Config.TRAIN_METADATA_PATH, debug=True)

    # Assertions for Tabular Data
    # Debug mode loads head(20)
    assert len(ids) == 20, f"Expected 20 IDs in debug mode, got {len(ids)}"
    assert X_tab.shape == (
        20,
        192,
    ), f"Expected feature shape (20, 192), got {X_tab.shape}"
    assert len(y_tab) == 20, f"Expected 20 labels, got {len(y_tab)}"
    print("  -> Tabular data loaded successfully. Shape verified.")

    # B. Test Image Dataset (LeafDataset)
    print("  Testing LeafDataset (Debug Mode)...")
    # We use a small image size for this quick check to avoid heavy resizing ops
    test_img_size = 224
    dataset = LeafDataset(
        Config.TRAIN_METADATA_PATH, img_size=test_img_size, debug=True
    )

    # Fetch one sample
    images_tensor, label, sample_id = dataset[0]

    # Assertions for Image Data
    # Shape should be (4_views, 3_channels, H, W)
    expected_shape = (4, 3, test_img_size, test_img_size)
    assert (
        images_tensor.shape == expected_shape
    ), f"Expected tensor shape {expected_shape}, got {images_tensor.shape}"
    assert isinstance(images_tensor, torch.Tensor), "Output should be a torch.Tensor"
    print(f"  -> LeafDataset sample verified. Tensor shape: {images_tensor.shape}")

    # ============================================================
    # Part 2: Execute Training Pipeline (Debug Mode)
    # ============================================================
    print("\n[Part 2] Executing Training Pipeline (Debug Mode)...")
    print(
        "  This will run feature extraction (DINOv2/ConvNeXt) and train the ensemble."
    )
    print("  Note: This runs on a small subset (20 samples) with 2 CV folds.")

    # We force load_cached_data=False to demonstrate the extraction logic
    # In a real run, you would likely set this to True to save time on restarts
    try:
        train_bagging_ensemble(load_cached_data=False, debug=True)
    except Exception as e:
        print(f"  !! Training failed with error: {e}")
        raise e

    # Verify that model artifacts were created
    # In debug mode, n_splits=2, so we expect pipeline_fold_0.pkl and pipeline_fold_1.pkl
    expected_models = [
        Config.PIPELINE_FILENAME_TEMPLATE.format(0),
        Config.PIPELINE_FILENAME_TEMPLATE.format(1),
    ]

    for model_path in expected_models:
        if os.path.exists(model_path):
            print(
                f"  -> Verified model artifact exists: {os.path.basename(model_path)}"
            )
        else:
            raise FileNotFoundError(f"Expected model file not found: {model_path}")

    # ============================================================
    # Part 3: Execute Inference and Submission
    # ============================================================
    print("\n[Part 3] Executing Inference and Submission (Debug Mode)...")

    try:
        predict_ensemble(load_cached_data=False, debug=True)
    except Exception as e:
        print(f"  !! Inference failed with error: {e}")
        raise e

    # Verify Submission File
    submission_path = Config.SUBMISSION_PATH
    if os.path.exists(submission_path):
        print(f"  -> Submission file created at: {submission_path}")

        # Validate content format
        df_sub = pd.read_csv(submission_path)
        print(f"  -> Submission shape: {df_sub.shape}")

        # Check columns
        if "id" not in df_sub.columns:
            raise ValueError("Submission file missing 'id' column.")

        # Check values are probabilities
        prob_cols = [c for c in df_sub.columns if c != "id"]
        if df_sub[prob_cols].min().min() < 0 or df_sub[prob_cols].max().max() > 1:
            raise ValueError("Probabilities out of range [0, 1].")

        print("  -> Submission format validated successfully.")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n============================================================")
    print("  Demonstration Completed Successfully")
    print("============================================================")


if __name__ == "__main__":
    run_demonstration()
