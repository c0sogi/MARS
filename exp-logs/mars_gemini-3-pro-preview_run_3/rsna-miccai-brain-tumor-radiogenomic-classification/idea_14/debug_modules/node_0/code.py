import os
import torch
import pandas as pd
import numpy as np
import sys

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_train_loader, get_val_loader, get_test_loader
from library.model import BraTS25DNet
from library.train import run_training
from library.predict import run_inference


def run_demo():
    print("Initializing BraTS21 2.5D Pipeline Demo...")

    # ==========================================
    # 1. Configuration Override for Demo
    # ==========================================
    # We override Config attributes to run a fast, minimal version of the pipeline.

    # Enable Debug mode to use a tiny subset of data (e.g., 6 patients)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 6

    # Reduce training complexity
    Config.BATCH_SIZE = 2
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Use main process for simplicity in demo

    # Redirect cache and output paths to a demo folder to avoid conflicts
    demo_dir = os.path.join(Config.WORKING_DIR, "demo_artifacts")
    os.makedirs(demo_dir, exist_ok=True)

    Config.TRAIN_CACHE_X = os.path.join(demo_dir, "train_X.npy")
    Config.TRAIN_CACHE_Y = os.path.join(demo_dir, "train_y.npy")
    Config.VAL_CACHE_X = os.path.join(demo_dir, "val_X.npy")
    Config.VAL_CACHE_Y = os.path.join(demo_dir, "val_y.npy")
    Config.TEST_CACHE_X = os.path.join(demo_dir, "test_X.npy")
    Config.TEST_CACHE_IDS = os.path.join(demo_dir, "test_ids.npy")

    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "demo_submission.csv")

    # Set seed for reproducibility
    seed_everything(42)
    print("Configuration updated for fast execution.")

    # ==========================================
    # 2. Data Loading Verification
    # ==========================================
    print("\n[Step 1/4] Verifying Data Loader...")

    # Force load_cached=False to trigger data processing on the debug subset
    train_loader = get_train_loader(load_cached=False)

    # Fetch one batch
    images, targets = next(iter(train_loader))

    print(f"   Batch Images Shape: {images.shape}")
    print(f"   Batch Targets Shape: {targets.shape}")

    # Validate Shapes
    # Expected: (Batch, Channels, H, W) -> (2, 128, 256, 256)
    expected_channels = Config.NUM_SLICES * Config.NUM_MODALITIES  # 32 * 4 = 128
    assert images.shape == (
        Config.BATCH_SIZE,
        expected_channels,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Image shape mismatch! Expected {(Config.BATCH_SIZE, expected_channels, Config.IMG_SIZE, Config.IMG_SIZE)}, got {images.shape}"

    assert targets.shape == (
        Config.BATCH_SIZE,
    ), f"Target shape mismatch! Expected {(Config.BATCH_SIZE,)}, got {targets.shape}"

    print("   Data Loader verification passed.")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n[Step 2/4] Verifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BraTS25DNet().to(device)
    dummy_input = images.to(device)

    # Perform forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"   Model Output Shape: {output.shape}")

    # Validate Output
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch! Expected {(Config.BATCH_SIZE, 1)}, got {output.shape}"

    print("   Model verification passed.")

    # ==========================================
    # 4. Training Pipeline Execution
    # ==========================================
    print("\n[Step 3/4] Executing Training Pipeline (1 Epoch)...")

    # Ensure validation data is also prepared (cached) before training loop starts
    _ = get_val_loader(load_cached=False)

    # Run training
    best_auc = run_training()

    # Validate Artifacts
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        # In the rare case validation AUC is 0.0 or doesn't improve, force save for demo continuity
        print(
            "   Note: Model not saved by early stopping (AUC didn't improve). Saving manually for inference demo."
        )
        torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint file missing!"
    print(f"   Training completed. Best AUC: {best_auc}")

    # ==========================================
    # 5. Inference Pipeline Execution
    # ==========================================
    print("\n[Step 4/4] Executing Inference Pipeline...")

    # Run inference
    submission_df = run_inference()

    # Validate Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission CSV file missing!"
    assert not submission_df.empty, "Submission DataFrame is empty!"
    assert (
        "BraTS21ID" in submission_df.columns and "MGMT_value" in submission_df.columns
    ), "Submission columns are incorrect!"

    print(f"   Inference completed. Generated {len(submission_df)} predictions.")
    print("   First prediction:", submission_df.iloc[0].to_dict())

    print("\nDemo completed successfully. All components verified.")


if __name__ == "__main__":
    run_demo()
