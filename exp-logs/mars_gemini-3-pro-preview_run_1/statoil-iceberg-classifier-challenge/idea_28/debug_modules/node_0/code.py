import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, compute_global_stats
from library.dataset import get_dataset, IcebergDataset
from library.model import IcebergResNet18
from library.calibration import run_calibration
from library.production import train_production_models
from library.inference import run_inference


def run_demo():
    print("============================================================")
    print("       Iceberg Classification Pipeline Demo Execution       ")
    print("============================================================")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring Environment for Demo...")

    # Override Config for speed
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce epochs and complexity for fast execution
    Config.BATCH_SIZE = 32
    Config.NUM_WORKERS = 2

    # Phase 1: Calibration (5-Fold CV)
    Config.PHASE1_MAX_EPOCHS = 2  # Run only 2 epochs per fold
    Config.PHASE1_PATIENCE = 1  # Aggressive early stopping

    # Phase 2: Production
    # Note: production.py runs 5 models. We limit the epochs per model.
    # We will set these dynamically based on calibration, but we enforce low limits here
    # to ensure the 'default' or 'fallback' isn't too high if calibration returns high numbers.
    Config.SWA_CYCLES = 1  # Only 1 SWA cycle
    Config.SWA_CYCLE_LEN = 2  # 2 epochs per cycle

    # Create directories
    Config.create_directories()

    # Set Seed
    seed_everything(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print("Configuration updated for speed.")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Data Loading...")

    # Force computation of global stats (usually cached, but good to verify)
    stats = compute_global_stats(load_cached_data=False)
    print(f"Global Stats Computed: {stats}")

    # Load Train Dataset
    train_ds = get_dataset("train", load_cached_data=False)
    print(f"Train Dataset Size: {len(train_ds)}")

    # Verify Item Structure
    img, angle, label = train_ds[0]

    # Assertions
    # Image should be (3, 224, 224) due to Albumentations Resize and ToTensorV2
    # 3 channels: Band1, Band2, Average
    assert img.shape == (3, 224, 224), f"Incorrect image shape: {img.shape}"
    assert isinstance(angle, torch.Tensor), "Angle should be a tensor"
    assert isinstance(label, torch.Tensor), "Label should be a tensor"
    assert img.dtype == torch.float32, "Image dtype should be float32"

    print("Data Loading Verification Passed: Shapes and Types are correct.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = IcebergResNet18().to(device)

    # Create dummy batch
    dummy_img = torch.randn(4, 3, 224, 224).to(device)
    dummy_angle = torch.randn(4).to(device)  # Normalized angles

    # Forward Pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_img, dummy_angle)

    # Assertions
    assert output.shape == (
        4,
        1,
    ), f"Output shape mismatch. Expected (4, 1), got {output.shape}"
    print("Model Verification Passed: Forward pass successful.")

    # -------------------------------------------------------------------------
    # 4. Phase 1: Calibration (Trajectory Discovery)
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Phase 1: Calibration...")

    # This runs 5-Fold CV. With MAX_EPOCHS=2, it should be fast.
    optimal_epochs, milestones, final_lr = run_calibration()

    print(
        f"Calibration Result -> Epochs: {optimal_epochs}, Milestones: {milestones}, LR: {final_lr}"
    )

    # Sanity check results
    assert optimal_epochs > 0, "Optimal epochs should be positive"
    assert isinstance(milestones, list), "Milestones should be a list"

    # -------------------------------------------------------------------------
    # 5. Phase 2: Production (Ensemble Training)
    # -------------------------------------------------------------------------
    print("\n[Step 5] Running Phase 2: Production Training...")

    # Force small epochs for Phase 2 demo regardless of calibration result
    # In a real run, we would use the calibration result directly.
    # Here we clamp it to ensure the demo finishes in time.
    demo_epochs = min(optimal_epochs, 2)

    train_production_models(
        optimal_epochs=demo_epochs, milestones=milestones, final_lr=final_lr
    )

    # Verify Checkpoints
    expected_checkpoints = [f"swa_model_{i}.pth" for i in range(5)]
    for ckpt in expected_checkpoints:
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, ckpt)
        assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"

    print("Production Phase Complete. All 5 SWA models saved.")

    # -------------------------------------------------------------------------
    # 6. Inference
    # -------------------------------------------------------------------------
    print("\n[Step 6] Running Inference...")

    run_inference()

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {df_sub.shape}")
    print(f"First 5 rows:\n{df_sub.head()}")

    # Check constraints
    # Test set has 321 rows (from description)
    # We check if it matches the metadata length
    df_test_meta = pd.read_csv(Config.TEST_META)
    assert len(df_sub) == len(
        df_test_meta
    ), f"Submission length {len(df_sub)} != Test Meta length {len(df_test_meta)}"

    # Check probabilities
    assert df_sub["is_iceberg"].min() >= 0.0, "Probabilities < 0 found"
    assert df_sub["is_iceberg"].max() <= 1.0, "Probabilities > 1 found"

    print("Inference Verification Passed.")

    print("\n============================================================")
    print("           Demo Execution Completed Successfully            ")
    print("============================================================")


if __name__ == "__main__":
    run_demo()
