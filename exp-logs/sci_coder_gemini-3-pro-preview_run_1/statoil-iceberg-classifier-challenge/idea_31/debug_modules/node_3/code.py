import os
import sys
import numpy as np
import pandas as pd
import torch
import logging

# Import from the provided library
from library.config import Config
from library.utils import set_seed, get_logger
from library.data import process_and_cache_data, IcebergDataset, get_transforms
from library.model import IsovariantResNet18
from library.pipeline import Pipeline


def run_demo():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print("\n[Demo] Setting up configuration for fast execution...")

    # Override Config for speed and isolation
    Config.WORK_DIR = "./working/demo_execution"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORK_DIR, "checkpoints")
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORK_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce computational load for demo
    Config.MAX_EPOCHS_PHASE1 = 1  # Run only 1 epoch per fold
    Config.SWA_EPOCHS = 1  # Run only 1 SWA epoch
    Config.ISOVARIANT_SCALE = 0.01  # Force minimal production epochs
    Config.BATCH_SIZE = 16  # Smaller batch size
    Config.NUM_WORKERS = 2

    # Create directories
    Config.setup_dirs()

    # Set seed
    set_seed(Config.SEED)

    # Setup logger to suppress verbose library logs if needed,
    # though we want to see the pipeline progress.
    logger = get_logger("demo")
    logger.info("Configuration updated for demo run.")

    # =========================================================================
    # 2. Data Loading & Verification
    # =========================================================================
    print("\n[Demo] Verifying Data Loading...")

    # Test raw data processing
    data = process_and_cache_data("train", load_cached_data=False)

    images = data["images"]
    angles = data["angles"]
    labels = data["labels"]

    # Assertions for data integrity
    assert images.ndim == 4, f"Images should be 4D, got {images.ndim}"
    assert images.shape[1:] == (
        75,
        75,
        2,
    ), f"Expected (N, 75, 75, 2), got {images.shape}"
    assert len(images) == len(angles) == len(labels), "Data length mismatch"
    print(
        f"Verified processed data shapes: Images {images.shape}, Angles {angles.shape}"
    )

    # Test Dataset Class
    ds = IcebergDataset(
        images[:32], angles[:32], labels[:32], transform=get_transforms("train")
    )

    # Fetch a single item
    img_t, ang_t, lbl_t = ds[0]

    # Verify Dataset output shapes
    # Image should be upsampled to Config.IMG_SIZE (224) and have 3 channels
    assert img_t.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Dataset image shape mismatch: {img_t.shape}"
    assert ang_t.shape == (1,), f"Dataset angle shape mismatch: {ang_t.shape}"
    assert isinstance(lbl_t, torch.Tensor), "Label should be a tensor"
    print("Verified IcebergDataset item shapes.")

    # =========================================================================
    # 3. Model Instantiation & Forward Pass
    # =========================================================================
    print("\n[Demo] Verifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IsovariantResNet18().to(device)
    model.eval()

    # Create dummy batch
    dummy_img = torch.randn(4, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)
    dummy_ang = torch.randn(4, 1).to(device)

    with torch.no_grad():
        output = model(dummy_img, dummy_ang)

    assert output.shape == (
        4,
        1,
    ), f"Model output shape mismatch. Expected (4, 1), got {output.shape}"
    print("Verified Model forward pass.")

    # Clean up
    del model, dummy_img, dummy_ang, output
    torch.cuda.empty_cache()

    # =========================================================================
    # 4. Pipeline Execution
    # =========================================================================
    print("\n[Demo] Starting Pipeline Execution...")

    pipeline = Pipeline()

    # Phase 1: Calibration
    # This will run 5 folds, 1 epoch each (due to config override)
    print("--- Phase 1: Calibration ---")
    e_opt = pipeline.run_calibration_phase()

    assert isinstance(e_opt, int) and e_opt > 0, "e_opt should be a positive integer"
    print(f"Calibration complete. Optimal Epochs: {e_opt}")

    # Phase 2: Production
    # This will train 5 ensemble models
    print("--- Phase 2: Production ---")
    pipeline.run_production_phase(e_opt)

    # Check if checkpoints were created
    checkpoints = [f for f in os.listdir(Config.CHECKPOINT_DIR) if f.endswith(".pth")]
    assert (
        len(checkpoints) >= 5
    ), f"Expected at least 5 checkpoints, found {len(checkpoints)}"
    print(f"Verified checkpoints created: {len(checkpoints)} files.")

    # Submission
    print("--- Generating Submission ---")
    pipeline.generate_submission()

    # =========================================================================
    # 5. Output Verification
    # =========================================================================
    print("\n[Demo] Verifying Submission File...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Verify shape
    # Test set has 321 rows based on provided metadata info in prompt
    # If the provided test.json has different size, we check against metadata
    df_test_meta = pd.read_csv(Config.TEST_META_PATH)
    expected_rows = len(df_test_meta)

    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Verify columns
    expected_cols = ["id", "is_iceberg"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Verify value range
    probs = df_sub["is_iceberg"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    print("Submission file verified successfully.")
    print("\n[Demo] Execution completed successfully.")


if __name__ == "__main__":
    run_demo()
