import os
import torch
import numpy as np
import pandas as pd
import warnings

# Import library components
from library.config import Config
from library.utils import seed_everything, calculate_rmse, save_submission
from library.network_modules import CoordinateAttention, ASPP, ResidualBlock
from library.model import CACResUNet
from library.dataset import TextDenoisingDataset
from library.train import run_training
from library.inference import predict_tiled, apply_tta

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Library Usage Demonstration ===\n")

    # --- 1. Configuration & Setup ---
    print("--- 1. Configuring Environment ---")
    seed_everything(42)

    # Override Config parameters for a fast, isolated demo run
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Hyperparameters for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.PATCHES_PER_IMAGE = 2  # Reduce patches per image (default 100)
    Config.DEBUG_SUBSET_SIZE = 4  # Use only 4 images for debug training
    Config.PATCH_SIZE = 64  # Smaller patches
    Config.TILE_OVERLAP = 0.25  # Less overlap for faster inference demo

    # Initialize the new directories
    Config.setup()
    print(f"Working Directory: {Config.WORKING_DIR}")
    print("Configuration updated for rapid execution.")

    # --- 2. Network Components Verification ---
    print("\n--- 2. Verifying Network Modules ---")
    device = torch.device("cpu")  # Use CPU for simple shape checks

    # Test Coordinate Attention
    dummy_input = torch.randn(2, 32, 64, 64).to(device)
    ca = CoordinateAttention(in_channels=32).to(device)
    out_ca = ca(dummy_input)
    assert out_ca.shape == (
        2,
        32,
        64,
        64,
    ), f"CoordinateAttention shape mismatch: {out_ca.shape}"
    print("CoordinateAttention: OK")

    # Test ASPP
    aspp = ASPP(in_channels=32, out_channels=16).to(device)
    out_aspp = aspp(dummy_input)
    assert out_aspp.shape == (2, 16, 64, 64), f"ASPP shape mismatch: {out_aspp.shape}"
    print("ASPP: OK")

    # Test Residual Block
    res_block = ResidualBlock(in_channels=32, out_channels=64, stride=2).to(device)
    out_res = res_block(dummy_input)
    assert out_res.shape == (
        2,
        64,
        32,
        32,
    ), f"ResidualBlock shape mismatch: {out_res.shape}"
    print("ResidualBlock: OK")

    # --- 3. Full Model Verification ---
    print("\n--- 3. Verifying CACResUNet Model ---")
    model = CACResUNet().to(device)
    # Input: (Batch, 1, H, W)
    dummy_img = torch.randn(1, 1, 128, 128).to(device)
    out_model = model(dummy_img)
    assert out_model.shape == (
        1,
        1,
        128,
        128,
    ), f"Model output shape mismatch: {out_model.shape}"
    print("CACResUNet Forward Pass: OK")

    # --- 4. Dataset Verification ---
    print("\n--- 4. Verifying Dataset Loading ---")
    # Initialize Train Dataset
    # Note: This will process/cache the first few images defined in metadata
    ds_train = TextDenoisingDataset(
        Config.TRAIN_METADATA, mode="train", load_cached_data=True
    )

    # Verify Length: num_images * patches_per_image
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    expected_len = len(df_train) * Config.PATCHES_PER_IMAGE
    assert (
        len(ds_train) == expected_len
    ), f"Dataset length mismatch. Expected {expected_len}, got {len(ds_train)}"

    # Verify Sample Shape (Patches)
    noisy_patch, clean_patch = ds_train[0]
    assert noisy_patch.shape == (
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), "Train patch shape incorrect"
    assert clean_patch.shape == (
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), "Train label shape incorrect"
    print("Train Dataset: OK")

    # Initialize Val Dataset
    ds_val = TextDenoisingDataset(
        Config.VAL_METADATA, mode="val", load_cached_data=True
    )
    # Verify Sample Shape (Full Image)
    noisy_val, clean_val, _ = ds_val[0]
    # Shape should be (1, H, W)
    assert noisy_val.ndim == 3 and noisy_val.shape[0] == 1, "Val image shape incorrect"
    print("Val Dataset: OK")

    # --- 5. Training Pipeline Verification ---
    print("\n--- 5. Verifying Training Pipeline ---")
    # Run a short training session (1 epoch, debug subset)
    # This tests the DataLoaders, Loss, Optimizer, and Checkpointing
    run_training(debug=True, num_epochs=1, patience=1, load_cached_data=True)

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved."
    print("Training execution completed successfully.")

    # --- 6. Inference Logic Verification ---
    print("\n--- 6. Verifying Inference Logic ---")
    # Load the trained model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Create a synthetic large image to test tiling (e.g., 200x200)
    # Patch size is 64, so this forces the tiling logic to work
    large_img = torch.randn(1, 1, 200, 200).to(device)

    # Test Tiled Prediction
    with torch.no_grad():
        pred_tiled = predict_tiled(model, large_img, patch_size=64, overlap=0.25)

    assert pred_tiled.shape == (
        1,
        1,
        200,
        200,
    ), f"Tiled prediction shape mismatch: {pred_tiled.shape}"
    print("Tiled Inference: OK")

    # Test Test-Time Augmentation (TTA)
    with torch.no_grad():
        pred_tta = apply_tta(
            model, large_img, patch_size=64, overlap=0.25, device=device
        )

    assert pred_tta.shape == (
        1,
        1,
        200,
        200,
    ), f"TTA prediction shape mismatch: {pred_tta.shape}"
    print("TTA Inference: OK")

    # --- 7. Utility Verification ---
    print("\n--- 7. Verifying Utilities ---")
    # Test RMSE
    rmse_val = calculate_rmse(np.array([0, 1]), np.array([0, 0]))
    assert np.isclose(rmse_val, 0.70710678), "RMSE calculation incorrect"
    print("RMSE Calculation: OK")

    # Test Submission Generation
    # Create mock predictions
    mock_preds = {
        "img_A": np.zeros((2, 2)),  # 4 pixels
        "img_B": np.ones((2, 2)),  # 4 pixels
    }
    save_submission(mock_preds, Config.SUBMISSION_PATH)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check row count: 2 images * 4 pixels = 8 rows
    assert len(df_sub) == 8, f"Submission row count mismatch: {len(df_sub)}"
    # Check ID format (e.g., img_A_1_1)
    assert df_sub.iloc[0]["id"] == "img_A_1_1", "Submission ID format incorrect"
    print("Submission File Generation: OK")

    print("\n=== All Tests Passed Successfully ===")


if __name__ == "__main__":
    main()
