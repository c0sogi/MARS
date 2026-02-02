import os
import shutil
import numpy as np
import torch
import pandas as pd
import cv2

# Import from provided library files
from library.config import GlobalConfig, StreamAConfig, StreamBConfig
from library.utils import seed_everything, pad_image, d4_transform, d4_inverse_transform
from library.dataset import DenoisingDataset, get_dataloader
from library.model import UNet
from library.train import train_model
from library.inference import generate_submission


def run_demo():
    print("=== Starting Denoising Pipeline Demo ===")

    # --- 1. Configuration & Setup ---
    # Redirect working directory to a demo folder to avoid clutter/conflicts
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Monkey-patch GlobalConfig to use the demo directory
    # This ensures all cache files, models, and submissions go to the demo folder
    GlobalConfig.WORKING_DIR = demo_dir
    GlobalConfig.SUBMISSION_DIR = demo_dir
    GlobalConfig.SUBMISSION_FILE = os.path.join(demo_dir, "demo_submission.csv")

    # Set a specific seed for reproducibility
    seed_everything(42)
    print(f"Working Directory set to: {GlobalConfig.WORKING_DIR}")

    # --- 2. Verify Utilities ---
    print("\n=== Verifying Utilities ===")

    # Test Padding Logic
    # Create a 100x100 image. Modulus 16 -> Next multiple is 112.
    dummy_img = np.random.rand(100, 100).astype(np.float32)
    padded_img, pad_info = pad_image(dummy_img, modulus=16)

    expected_h, expected_w = 112, 112
    assert padded_img.shape == (
        expected_h,
        expected_w,
    ), f"Padding failed. Expected {(expected_h, expected_w)}, got {padded_img.shape}"
    print("✓ pad_image logic verified.")

    # Test D4 Transform & Inverse Logic
    # Transform k=1 (Rot90)
    trans_img = d4_transform(dummy_img, k=1)
    # Inverse k=1 (Rot270)
    restored_img = d4_inverse_transform(trans_img, k=1)

    # Check if restored matches original (using approximate equality for float precision)
    assert np.allclose(
        dummy_img, restored_img
    ), "D4 Inverse Transform failed to restore original image."
    print("✓ D4 transform/inverse logic verified.")

    # --- 3. Verify Dataset & DataLoader ---
    print("\n=== Verifying Dataset & DataLoader ===")

    # Initialize Dataset (Train mode)
    # This will trigger cache generation in the demo_dir
    # We use StreamAConfig which defines the patch size for training
    train_ds = DenoisingDataset(
        mode="train", stream_config=StreamAConfig, load_cached_data=False
    )

    assert len(train_ds) > 0, "Training dataset is empty."
    print(f"Dataset initialized with {len(train_ds)} samples.")

    # Check item structure and shapes
    noisy, clean, img_id = train_ds[0]

    # Expected: (1, PatchSize, PatchSize) for train
    patch_size = StreamAConfig.PATCH_SIZE
    assert noisy.shape == (
        1,
        patch_size,
        patch_size,
    ), f"Incorrect noisy tensor shape. Expected (1, {patch_size}, {patch_size}), got {noisy.shape}"
    assert clean.shape == (
        1,
        patch_size,
        patch_size,
    ), f"Incorrect clean tensor shape. Expected (1, {patch_size}, {patch_size}), got {clean.shape}"
    print("✓ Dataset item shapes verified.")

    # Check DataLoader functionality
    train_loader = get_dataloader(
        mode="train", stream_config=StreamAConfig, batch_size=2
    )
    batch_noisy, batch_clean, batch_ids = next(iter(train_loader))
    assert batch_noisy.shape[0] == 2, "DataLoader batch size mismatch."
    print("✓ DataLoader functioning.")

    # --- 4. Verify Model Architecture ---
    print("\n=== Verifying Model Architecture ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate Stream A Model (Context Specialist)
    model = UNet(
        depth=StreamAConfig.DEPTH,
        encoder_filters=StreamAConfig.ENCODER_FILTERS,
        bottleneck_filters=StreamAConfig.BOTTLENECK_FILTERS,
        bottleneck_depth=StreamAConfig.BOTTLENECK_DEPTH,
        in_channels=1,
        out_channels=1,
    ).to(device)

    # Create dummy input tensor (Batch, Channel, H, W)
    # Using 320x320 (Stream A patch size)
    dummy_input = torch.randn(1, 1, 320, 320).to(device)

    # Forward pass
    output = model(dummy_input)

    assert (
        output.shape == dummy_input.shape
    ), f"Model output shape mismatch. Expected {dummy_input.shape}, got {output.shape}"
    print("✓ Model forward pass verified.")

    # --- 5. Verify Training Loop ---
    print("\n=== Verifying Training Loop ===")

    # Train for 1 epoch with debug=True (runs only a few batches)
    # We use seed 42 because it is listed in GlobalConfig.STREAM_A_SEEDS.
    # The inference script specifically looks for models with these seeds.
    train_seed = 42
    print(f"Training Stream A (Seed {train_seed}) in debug mode...")

    best_rmse = train_model(
        stream_config=StreamAConfig, seed=train_seed, epochs=1, debug=True
    )

    # Verify model file creation
    model_path = os.path.join(
        GlobalConfig.WORKING_DIR, f"{StreamAConfig.NAME}_seed_{train_seed}.pth"
    )
    assert os.path.exists(model_path), f"Model file was not saved at {model_path}"
    print(f"✓ Training complete. Model saved. Best RMSE: {best_rmse:.4f}")

    # --- 6. Verify Inference & Submission ---
    print("\n=== Verifying Inference & Submission ===")

    # Run submission generation in debug mode (processes only a few images)
    # This relies on the model saved in the previous step.
    # Since we only trained one seed, the ensemble will consist of just that one model.
    generate_submission(load_cached_data=False, debug=True)

    # Verify submission file existence
    assert os.path.exists(GlobalConfig.SUBMISSION_FILE), "Submission file not created."

    # Verify submission content format
    df_sub = pd.read_csv(GlobalConfig.SUBMISSION_FILE)
    print(f"Submission file generated with {len(df_sub)} rows.")

    assert (
        "id" in df_sub.columns and "value" in df_sub.columns
    ), "Submission columns missing."
    assert len(df_sub) > 0, "Submission file is empty."

    # Check value range (should be [0, 1] due to Sigmoid activation)
    min_val, max_val = df_sub["value"].min(), df_sub["value"].max()
    assert (
        0 <= min_val and max_val <= 1
    ), f"Values out of range [0, 1]. Range: [{min_val}, {max_val}]"

    print("✓ Submission format verified.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
