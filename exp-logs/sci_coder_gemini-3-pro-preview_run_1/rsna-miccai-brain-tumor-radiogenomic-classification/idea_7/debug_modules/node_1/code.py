import os
import shutil
import pandas as pd
import torch
import numpy as np

# Import provided library modules
from library import config
from library import utils
from library import data_loader
from library import model
from library import train


def main():
    print("=== Glioblastoma Subtype Prediction Pipeline Demo ===\n")

    # ---------------------------------------------------------
    # 1. Setup & Configuration Override
    # ---------------------------------------------------------
    # Define a temporary directory for this execution to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    print(f"1. Configuring environment (Working Dir: {DEMO_DIR})...")

    # Override config parameters for a fast demonstration
    config.WORKING_DIR = DEMO_DIR
    config.EPOCHS = 2  # Run only 2 epochs
    config.BATCH_SIZE = 2  # Small batch size for mini-dataset
    config.NUM_WORKERS = 0  # Disable multiprocessing for tiny data overhead
    config.PATIENCE = 1  # Aggressive early stopping for demo

    # Ensure reproducibility
    utils.set_seed(42)

    # ---------------------------------------------------------
    # 2. Create Mini-Datasets
    # ---------------------------------------------------------
    print("2. Creating mini-datasets for rapid testing...")

    # Read original metadata
    df_train_full = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val_full = pd.read_csv(config.VAL_METADATA_PATH)

    # Take a tiny subset (4 training samples, 2 validation samples)
    df_train_mini = df_train_full.head(4).copy()
    df_val_mini = df_val_full.head(2).copy()

    # Save mini metadata to the demo directory
    mini_train_path = os.path.join(DEMO_DIR, "mini_train.csv")
    mini_val_path = os.path.join(DEMO_DIR, "mini_val.csv")

    df_train_mini.to_csv(mini_train_path, index=False)
    df_val_mini.to_csv(mini_val_path, index=False)

    # Point config to these new mini files
    config.TRAIN_METADATA_PATH = mini_train_path
    config.VAL_METADATA_PATH = mini_val_path
    # Point test to val just to ensure file existence if accessed (though not used in training loop)
    config.TEST_METADATA_PATH = mini_val_path

    print(
        f"   Created mini_train.csv ({len(df_train_mini)} rows) and mini_val.csv ({len(df_val_mini)} rows)."
    )

    # ---------------------------------------------------------
    # 3. Verify Data Processing Logic
    # ---------------------------------------------------------
    print("3. Verifying Data Loader and Montage Construction...")

    # Manually trigger dataset processing to check shapes
    # This reads DICOMs, creates the montage, and saves .npy files
    images, targets, ids = data_loader.process_dataset(
        df_train_mini, "demo_check", load_cached=False
    )

    # Calculate expected dimensions
    # Montage creates a grid of slices.
    # Height = SLICE_SIZE * GRID_SIZE (224 * 2 = 448)
    # Channels = Number of modalities (3: FLAIR, T1wCE, T2w)
    expected_h = config.SLICE_SIZE * config.GRID_SIZE
    expected_w = config.SLICE_SIZE * config.GRID_SIZE
    expected_c = len(config.SELECTED_MODALITIES)
    expected_n = len(df_train_mini)

    print(f"   Output Images Shape: {images.shape}")
    print(f"   Output Targets Shape: {targets.shape}")

    # Assertions
    assert images.shape == (
        expected_n,
        expected_h,
        expected_w,
        expected_c,
    ), f"Image shape mismatch! Expected ({expected_n}, {expected_h}, {expected_w}, {expected_c}), got {images.shape}"
    assert targets.shape == (expected_n,), "Targets shape mismatch!"
    assert ids.shape == (expected_n,), "IDs shape mismatch!"

    # Check value ranges (should be normalized to [0, 1] or close to it if empty)
    print(f"   Pixel Value Range: [{images.min():.4f}, {images.max():.4f}]")
    assert (
        images.min() >= 0.0 and images.max() <= 1.0 + 1e-6
    ), "Images not properly normalized to [0,1]"

    print("   Data processing logic verified successfully.")

    # ---------------------------------------------------------
    # 4. Verify Model Architecture
    # ---------------------------------------------------------
    print("4. Verifying Model Architecture...")

    # Instantiate model
    # We use pretrained=False to speed up initialization for the demo
    net = model.MontageEfficientNet(
        model_name="efficientnet_b0", pretrained=False, num_classes=1
    )

    # Create dummy input tensor
    # Model expects (Batch, Channels, Height, Width)
    # Note: The DataLoader (via Albumentations) handles the permute from (H,W,C) to (C,H,W).
    # Here we simulate the tensor after the DataLoader.
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, expected_h, expected_w)

    # Forward pass
    output = net(dummy_input)

    print(f"   Input Shape: {dummy_input.shape}")
    print(f"   Output Shape: {output.shape}")

    assert output.shape == (
        batch_size,
        1,
    ), f"Model output shape mismatch. Expected ({batch_size}, 1), got {output.shape}"
    print("   Model architecture verified successfully.")

    # ---------------------------------------------------------
    # 5. Run Training Loop (Integration Test)
    # ---------------------------------------------------------
    print("5. Running Training Loop (Integration Test)...")

    # run_training handles dataloader creation (using our modified config paths),
    # model init, training loop, validation, and saving.
    # We set load_cached=False to force it to process our new mini metadata files
    # instead of looking for old cached files in the working dir.
    best_auc = train.run_training(load_cached=False)

    print(f"   Training completed. Best AUC: {best_auc}")

    # Verify the checkpoint file was created
    checkpoint_path = os.path.join(DEMO_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"   Checkpoint found at: {checkpoint_path}")

        # Verify loading
        loaded_state = utils.load_checkpoint(checkpoint_path, net)
        assert (
            "model_state_dict" in loaded_state
        ), "Checkpoint corrupted: missing model_state_dict"
        print("   Checkpoint loaded and verified.")
    else:
        raise FileNotFoundError("Training finished but 'best_model.pth' was not found.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
