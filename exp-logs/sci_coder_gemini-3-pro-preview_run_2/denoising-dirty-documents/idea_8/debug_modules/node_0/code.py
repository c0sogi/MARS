import os
import shutil
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything
from library.dataset import DenoisingDataset
from library.model import CoConvNeXtUNet
from library.inference import predict_full_image
from library.train import run_training


def run_demo():
    # 1. Setup Configuration for Demo
    print(">>> Setting up configuration for demo run...")

    # Define a specific working directory for this demo to avoid conflicts
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config paths and parameters for speed and isolation
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.MODEL_PATH = os.path.join(demo_dir, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Speed optimizations
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.PATCHES_PER_IMAGE = 2  # Only grab 2 patches per image for training
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Create cache dir
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Verify Dataset Logic
    print("\n>>> Verifying Dataset Logic...")

    # Test Train Dataset
    train_ds = DenoisingDataset(Config.TRAIN_CSV, mode="train", load_cached_data=True)
    if len(train_ds) > 0:
        sample_noisy, sample_clean = train_ds[0]

        # Assertions
        assert isinstance(
            sample_noisy, torch.Tensor
        ), "Train noisy sample must be a Tensor"
        assert isinstance(
            sample_clean, torch.Tensor
        ), "Train clean sample must be a Tensor"
        assert sample_noisy.shape == (
            Config.IN_CHANNELS,
            Config.PATCH_SIZE,
            Config.PATCH_SIZE,
        ), f"Expected train shape {(Config.IN_CHANNELS, Config.PATCH_SIZE, Config.PATCH_SIZE)}, got {sample_noisy.shape}"
        assert (
            sample_clean.shape == sample_noisy.shape
        ), "Noisy and Clean shapes must match"
        print("Train dataset verification passed.")
    else:
        print("Warning: Train dataset is empty, skipping verification.")

    # Test Val Dataset (Full Images)
    val_ds = DenoisingDataset(Config.VAL_CSV, mode="val", load_cached_data=True)
    if len(val_ds) > 0:
        val_noisy, val_clean, val_id = val_ds[0]
        assert len(val_noisy.shape) == 3, "Val noisy image should be (C, H, W)"
        assert val_noisy.shape == val_clean.shape, "Val noisy and clean shapes mismatch"
        assert isinstance(val_id, str), "Image ID should be a string"
        print("Validation dataset verification passed.")

    # 3. Verify Model Logic
    print("\n>>> Verifying Model Logic...")
    model = CoConvNeXtUNet(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        base_filters=Config.BASE_FILTERS,
    ).to(device)

    # Create dummy input: (Batch, Channels, Height, Width)
    dummy_input = torch.randn(
        2, Config.IN_CHANNELS, Config.PATCH_SIZE, Config.PATCH_SIZE
    ).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    assert (
        output.shape == dummy_input.shape
    ), f"Model output shape mismatch. Expected {dummy_input.shape}, got {output.shape}"
    print("Model forward pass verification passed.")

    # 4. Verify Inference Tiling Logic
    print("\n>>> Verifying Inference Tiling Logic...")
    # Create a random image tensor larger than patch size
    large_h, large_w = 300, 300
    dummy_large_img = torch.rand(Config.IN_CHANNELS, large_h, large_w).to(device)

    with torch.no_grad():
        tiled_output = predict_full_image(
            model,
            dummy_large_img,
            patch_size=Config.PATCH_SIZE,
            overlap_ratio=0.5,
            device=device,
        )

    assert tiled_output.shape == (
        Config.IN_CHANNELS,
        large_h,
        large_w,
    ), f"Tiled inference output shape mismatch. Expected {(Config.IN_CHANNELS, large_h, large_w)}, got {tiled_output.shape}"
    print("Inference tiling verification passed.")

    # 5. Run Full Training Pipeline (Integration Test)
    print("\n>>> Running Full Training Pipeline (Debug Mode)...")
    # This runs training, validation, saves checkpoint, and generates submission
    # debug=True subsets the data significantly
    run_training(load_cached_data=True, debug=True)

    # 6. Verify Outputs
    print("\n>>> Verifying Pipeline Outputs...")

    # Check Model Checkpoint
    if os.path.exists(Config.MODEL_PATH):
        print(f"Model checkpoint found at {Config.MODEL_PATH}")
    else:
        raise FileNotFoundError(
            f"Model checkpoint not generated at {Config.MODEL_PATH}"
        )

    # Check Submission File
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file found at {Config.SUBMISSION_PATH}")

        # Validate CSV content
        df = pd.read_csv(Config.SUBMISSION_PATH)
        assert (
            "id" in df.columns and "value" in df.columns
        ), "Submission CSV missing required columns"
        assert len(df) > 0, "Submission CSV is empty"

        # Check value range
        assert (
            df["value"].min() >= 0 and df["value"].max() <= 1.05
        ), "Pixel values should be approximately within [0, 1]"

        print(f"Submission CSV validated. Rows: {len(df)}")
        print("First 5 rows:")
        print(df.head())
    else:
        raise FileNotFoundError(
            f"Submission file not generated at {Config.SUBMISSION_PATH}"
        )

    print("\n>>> Demo Completed Successfully!")


if __name__ == "__main__":
    run_demo()
