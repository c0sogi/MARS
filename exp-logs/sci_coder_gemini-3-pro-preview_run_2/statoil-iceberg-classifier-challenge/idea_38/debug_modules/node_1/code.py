import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, setup_logger
from library.model_components import DualPooling, CBAM, SEBlock
from library.model import CRWBN
from library.data_loader import get_dataloaders, process_and_cache_data
from library.train_eval import run_cv_training, generate_submission

if __name__ == "__main__":
    print("Starting CR-WBN Pipeline Demonstration...")

    # ==========================================
    # 1. SETUP & CONFIGURATION OVERRIDE
    # ==========================================
    # Override Config for a fast, minimal execution
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_PATH = os.path.join(Config.WORKING_DIR, "cache", "processed_data.npz")
    Config.MODEL_PATH_TEMPLATE = os.path.join(Config.WORKING_DIR, "tp_wbn_fold_{}.pth")
    Config.SUBMISSION_DIR = "./working/demo_run/submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.CACHE_PATH), exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set hyperparams for speed
    Config.EPOCHS = 1
    Config.NUM_FOLDS = 2  # Run only 2 folds
    Config.BATCH_SIZE = 4
    Config.DEBUG_SIZE = 20  # Very small dataset for demo
    Config.DEBUG = True  # Enable debug flag logic in custom code if present

    # Set seed for reproducibility
    set_seed(42)

    print("Configuration configured for fast demonstration.")

    # ==========================================
    # 2. COMPONENT LOGIC VERIFICATION
    # ==========================================
    print("\nVerifying Model Components...")

    # Test DualPooling
    # Input: (Batch, Channels, H, W) -> Output: (Batch, Channels*2, H/2, W/2)
    dp = DualPooling()
    dummy_input = torch.randn(2, 64, 75, 75)
    out = dp(dummy_input)
    # 75 / 2 = 37 (floor)
    expected_shape = (2, 128, 37, 37)
    assert (
        out.shape == expected_shape
    ), f"DualPooling Failed: Expected {expected_shape}, got {out.shape}"
    print("  DualPooling: OK")

    # Test CBAM
    # Input: (Batch, Channels, H, W) -> Output: Same
    cbam = CBAM(in_planes=64)
    out = cbam(dummy_input)
    assert (
        out.shape == dummy_input.shape
    ), f"CBAM Failed: Expected {dummy_input.shape}, got {out.shape}"
    print("  CBAM: OK")

    # Test SEBlock
    # Input: (Batch, Channels, H, W) -> Output: Same
    se = SEBlock(channels=64)
    out = se(dummy_input)
    assert (
        out.shape == dummy_input.shape
    ), f"SEBlock Failed: Expected {dummy_input.shape}, got {out.shape}"
    print("  SEBlock: OK")

    # Test Full CRWBN Model
    # Input Img: (Batch, 3, 75, 75), Input Angle: (Batch, 1) -> Output: (Batch, 1)
    model = CRWBN()
    dummy_img = torch.randn(4, 3, 75, 75)
    dummy_angle = torch.randn(4, 1)
    out = model(dummy_img, dummy_angle)
    assert out.shape == (4, 1), f"CRWBN Failed: Expected (4, 1), got {out.shape}"
    print("  CRWBN Model: OK")

    # ==========================================
    # 3. DATA LOADING VERIFICATION
    # ==========================================
    print("\nVerifying Data Loading...")

    # This triggers process_and_cache_data internally
    # We use debug=True to slice the arrays inside get_dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Check Train Loader
    img_batch, angle_batch, label_batch = next(iter(train_loader))

    # Verify Shapes
    assert img_batch.shape == (
        Config.BATCH_SIZE,
        3,
        75,
        75,
    ), f"Train Batch Image Shape Mismatch: {img_batch.shape}"
    assert angle_batch.shape[0] == Config.BATCH_SIZE, "Train Batch Angle Count Mismatch"
    assert label_batch.shape[0] == Config.BATCH_SIZE, "Train Batch Label Count Mismatch"

    print(
        f"  Train Loader Batch: Images {img_batch.shape}, Angles {angle_batch.shape}, Labels {label_batch.shape}"
    )
    print("  Data Loading: OK")

    # ==========================================
    # 4. TRAINING PIPELINE EXECUTION
    # ==========================================
    print("\nExecuting Training Pipeline (Debug Mode)...")

    # Run the CV training loop provided in the library
    # This will train for 1 epoch on 2 folds using the tiny debug dataset
    run_cv_training(debug=True)

    # Verify artifacts
    for fold in range(Config.NUM_FOLDS):
        model_path = Config.MODEL_PATH_TEMPLATE.format(fold)
        assert os.path.exists(
            model_path
        ), f"Model file for fold {fold} not found at {model_path}"

    print("  Training Pipeline: OK (Models saved)")

    # ==========================================
    # 5. INFERENCE PIPELINE EXECUTION
    # ==========================================
    print("\nExecuting Inference Pipeline (Debug Mode)...")

    # Generate submission using the trained models
    generate_submission(debug=True)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "id" in df_sub.columns and "is_iceberg" in df_sub.columns
    ), "Submission columns missing"
    assert (
        len(df_sub) == Config.DEBUG_SIZE
    ), f"Submission length mismatch. Expected {Config.DEBUG_SIZE}, got {len(df_sub)}"

    print(f"  Submission Generated: {Config.SUBMISSION_PATH}")
    print("  Inference Pipeline: OK")

    print("\nAll demonstration steps completed successfully.")
