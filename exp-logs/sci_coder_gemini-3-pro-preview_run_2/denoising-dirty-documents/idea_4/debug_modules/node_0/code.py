import os
import torch
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, load_checkpoint
from library.dataset import DenoisingDataset
from library.model import ResUNetPlusPlus
from library.train import run_training
from library.inference import predict_tiled, apply_tta


def run_demo():
    print("=== Starting Demonstration of Denoising Pipeline ===")

    # -------------------------------------------------------------------------
    # 1. Setup Configuration for Fast Demo Run
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for demo...")

    # Override Config defaults to ensure the demo runs quickly (within minutes)
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.PATCHES_PER_IMAGE = 2  # Only extract 2 patches per image (vs 100 default)
    Config.BATCH_SIZE = 4  # Small batch size
    Config.PATIENCE = 1  # Early stopping threshold

    # Use a specific working directory for the demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "model.pth")
    Config.SUBMISSION_FILE_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Initialize directories and seeds
    Config.initialize()
    seed_everything(Config.SEED)

    print(
        f"Configured: Epochs={Config.NUM_EPOCHS}, Patches/Img={Config.PATCHES_PER_IMAGE}, Device={Config.DEVICE}"
    )

    # -------------------------------------------------------------------------
    # 2. Verify Dataset Logic
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Dataset Logic...")

    # Initialize Train Dataset
    # load_cached_data=False forces the dataset to read images via cv2 and process them,
    # ensuring the image loading pipeline is working correctly.
    train_ds = DenoisingDataset(mode="train", load_cached_data=False)

    # Check length: (Number of images in metadata) * (Patches per image)
    # Train metadata has 92 images.
    expected_len = len(train_ds.data) * Config.PATCHES_PER_IMAGE
    print(f"Train Dataset Length: {len(train_ds)} (Expected: {expected_len})")
    assert len(train_ds) == expected_len, "Dataset length mismatch"

    # Check Item Shapes (Patching)
    sample_noisy, sample_target = train_ds[0]
    print(f"Sample Patch Shape: {sample_noisy.shape}")

    # Shape should be (C, H, W) -> (1, PATCH_SIZE, PATCH_SIZE)
    assert sample_noisy.shape == (
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), "Incorrect patch shape"
    assert sample_target.shape == (
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), "Incorrect target shape"

    # Initialize Val Dataset (Full images)
    val_ds = DenoisingDataset(mode="val")
    val_noisy, val_target = val_ds[0]
    print(f"Validation Image Shape: {val_noisy.shape}")

    # Validation returns full images, shape varies but should be (1, H, W)
    # We check dimensionality and channel count
    assert (
        val_noisy.dim() == 3 and val_noisy.size(0) == 1
    ), "Incorrect validation image format"
    assert (
        val_target.dim() == 3 and val_target.size(0) == 1
    ), "Incorrect validation target format"

    # -------------------------------------------------------------------------
    # 3. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Model Architecture...")
    model = ResUNetPlusPlus().to(Config.DEVICE)

    # Create dummy input batch (B, C, H, W)
    dummy_input = torch.randn(2, 1, Config.PATCH_SIZE, Config.PATCH_SIZE).to(
        Config.DEVICE
    )

    # Test Training Mode (Deep Supervision)
    model.train()
    train_out = model(dummy_input)

    # Should return a list of 4 tensors (levels 0_1 to 0_4) because DEEP_SUPERVISION is True
    assert isinstance(
        train_out, list
    ), "Model in train mode should return a list (Deep Supervision)"
    assert len(train_out) == 4, "Model should return 4 outputs for deep supervision"
    assert train_out[-1].shape == (
        2,
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), "Output shape mismatch"
    print("Forward pass (Train Mode - Deep Supervision) successful.")

    # Test Eval Mode
    model.eval()
    eval_out = model(dummy_input)

    # Should return a single tensor (the final output)
    assert isinstance(
        eval_out, torch.Tensor
    ), "Model in eval mode should return a tensor"
    assert eval_out.shape == (
        2,
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), "Eval output shape mismatch"
    print("Forward pass (Eval Mode) successful.")

    # -------------------------------------------------------------------------
    # 4. Run Training Loop
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running Training Loop (Reduced)...")
    # This function handles data loading, model init, training loop, validation,
    # checkpoint saving, and finally generates the submission file.
    run_training(num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE)

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    print("Training execution completed.")

    # -------------------------------------------------------------------------
    # 5. Verify Inference Components
    # -------------------------------------------------------------------------
    print("\n[Step 5] Verifying Inference Components...")

    # Load the trained model using the utility function
    model = ResUNetPlusPlus().to(Config.DEVICE)
    load_checkpoint(model, Config.MODEL_SAVE_PATH, device=Config.DEVICE)
    model.eval()

    # Create a dummy large image to test tiling logic
    # Size 200x200 is larger than patch size 64, forcing the tiling logic to engage
    large_img = torch.randn(1, 1, 200, 200).to(Config.DEVICE)

    # Test Tiled Prediction
    pred_tiled = predict_tiled(model, large_img, tile_size=64, overlap=16)
    print(f"Tiled Prediction Shape: {pred_tiled.shape}")
    assert pred_tiled.shape == (
        1,
        1,
        200,
        200,
    ), "Tiled prediction output shape mismatch"

    # Test Test-Time Augmentation (TTA)
    pred_tta = apply_tta(model, large_img)
    print(f"TTA Prediction Shape: {pred_tta.shape}")
    assert pred_tta.shape == (1, 1, 200, 200), "TTA prediction output shape mismatch"

    # -------------------------------------------------------------------------
    # 6. Verify Submission File
    # -------------------------------------------------------------------------
    print("\n[Step 6] Verifying Submission File...")
    assert os.path.exists(Config.SUBMISSION_FILE_PATH), "Submission file not found"

    df_sub = pd.read_csv(Config.SUBMISSION_FILE_PATH)
    print(f"Submission rows: {len(df_sub)}")
    print("Header:", df_sub.columns.tolist())

    # Check columns
    assert (
        "id" in df_sub.columns and "value" in df_sub.columns
    ), "Submission columns missing"
    assert len(df_sub) > 0, "Submission file is empty"

    # Check format of first ID to ensure it matches 'imageID_row_col'
    first_id = df_sub.iloc[0]["id"]
    parts = str(first_id).split("_")
    assert len(parts) >= 3, f"Invalid ID format in submission: {first_id}"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
