import os
import sys
import torch
import numpy as np
import pandas as pd

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

from library.config import Config

# --- 1. Configuration Overrides for Speed and Demonstration ---
# We modify the Config class attributes before importing other modules.
# This ensures that the demonstration runs quickly (1 epoch, tiny model, subset of data).
Config.SEED = 42
Config.DEBUG = True  # Use a subset of the dataset
Config.DEBUG_SAMPLES = 20  # Limit to 20 samples
Config.NUM_EPOCHS = 1  # Train for only 1 epoch
Config.BATCH_SIZE = 4  # Small batch size
Config.FEATURES = [16, 32]  # Reduced feature depth for faster computation
Config.TTA_ENABLED = False  # Disable Test-Time Augmentation for speed
Config.WORKING_DIR = "./working/demo_run"
Config.MODEL_CHECKPOINT = os.path.join(Config.WORKING_DIR, "demo_model.pth")
Config.SUBMISSION_FILE = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

# Create the working directory
os.makedirs(Config.WORKING_DIR, exist_ok=True)

# Import library modules after configuration update
from library.utils import seed_everything, pad_image, unpad_image, calculate_rmse
from library.dataset import get_dataloaders, DenoisingDataset
from library.model import ResUNet
from library.train import train_one_epoch, validate
from library.inference import generate_submission, predict_with_tta


def main():
    print("=== Starting Denoising Pipeline Demonstration ===")

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # --- 2. Verify Utility Functions ---
    print("\n[1/5] Verifying Utility Functions...")

    # Test Image Padding (must be divisible by 32)
    h, w = 50, 50
    dummy_img = np.random.rand(h, w).astype(np.float32)
    padded = pad_image(dummy_img, factor=32)

    # Next multiple of 32 after 50 is 64
    assert padded.shape == (
        64,
        64,
    ), f"Padding failed. Expected (64, 64), got {padded.shape}"

    # Test Image Unpadding
    unpadded = unpad_image(padded, (h, w))
    assert unpadded.shape == (
        h,
        w,
    ), f"Unpadding failed. Expected ({h}, {w}), got {unpadded.shape}"
    assert np.allclose(
        dummy_img, unpadded
    ), "Unpadded image content does not match original."

    # Test RMSE Calculation
    # RMSE of [1, 1] and [0, 2] -> MSE = ((1-0)^2 + (1-2)^2)/2 = 1 -> RMSE = 1
    rmse_val = calculate_rmse(np.array([1.0, 1.0]), np.array([0.0, 2.0]))
    assert np.isclose(
        rmse_val, 1.0
    ), f"RMSE calculation error. Expected 1.0, got {rmse_val}"

    print("Utils verified.")

    # --- 3. Verify Data Loading ---
    print("\n[2/5] Verifying Data Loading...")

    # Initialize DataLoaders
    # num_workers=0 is used here to avoid multiprocessing overhead in this short script
    train_loader, val_loader, test_loader = get_dataloaders(
        train_batch_size=Config.BATCH_SIZE,
        val_batch_size=1,
        test_batch_size=1,
        num_workers=0,
    )

    # Verify Train Batch Structure (Patches)
    try:
        noisy_batch, clean_batch = next(iter(train_loader))
        print(f"Train Batch Shape: {noisy_batch.shape}")

        expected_shape = (Config.BATCH_SIZE, 1, Config.PATCH_SIZE, Config.PATCH_SIZE)
        assert (
            noisy_batch.shape == expected_shape
        ), f"Train batch shape mismatch: {noisy_batch.shape}"
        assert (
            clean_batch.shape == expected_shape
        ), f"Train target shape mismatch: {clean_batch.shape}"
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # Verify Validation Batch Structure (Full Images)
    try:
        v_noisy, v_clean = next(iter(val_loader))
        # Validation batch size is 1, dimensions vary per image
        assert v_noisy.shape[0] == 1, "Validation batch size should be 1"
        assert v_noisy.ndim == 4, "Validation tensor should be 4D (B, C, H, W)"
    except StopIteration:
        raise AssertionError("Validation loader is empty!")

    print("Data loading verified.")

    # --- 4. Verify Model Architecture ---
    print("\n[3/5] Verifying Model Architecture...")

    # Instantiate the model using the reduced features from Config
    model = ResUNet(features=Config.FEATURES).to(device)

    # Perform a dummy forward pass to check dimensions
    dummy_input = torch.randn(2, 1, 128, 128).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    assert (
        output.shape == dummy_input.shape
    ), f"Model I/O mismatch. In: {dummy_input.shape}, Out: {output.shape}"
    print("Model instantiated and forward pass successful.")

    # --- 5. Verify Training Loop ---
    print("\n[4/5] Verifying Training Loop...")

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Run training for one epoch
    print(f"Training for {Config.NUM_EPOCHS} epoch(s)...")
    avg_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Epoch 1 Train Loss: {avg_loss:.6f}")

    assert avg_loss >= 0, "Training loss should be non-negative"
    assert not np.isnan(avg_loss), "Training loss is NaN"

    # Save the model checkpoint (required for the inference step)
    torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)
    print(f"Model saved to {Config.MODEL_CHECKPOINT}")

    # Run validation
    val_rmse = validate(model, val_loader, device)
    print(f"Validation RMSE: {val_rmse:.6f}")

    print("Training loop verified.")

    # --- 6. Verify Inference and Submission ---
    print("\n[5/5] Verifying Inference and Submission...")

    # Manually test TTA wrapper
    model.eval()
    with torch.no_grad():
        tta_out = predict_with_tta(model, dummy_input)
        assert tta_out.shape == dummy_input.shape, "TTA output shape mismatch"

    # Generate Submission using the library function
    # This tests loading the saved checkpoint and processing the test set
    generate_submission(
        checkpoint_path=Config.MODEL_CHECKPOINT,
        output_file=Config.SUBMISSION_FILE,
        device=device,
        use_tta=Config.TTA_ENABLED,
        batch_size=1,
    )

    # Verify the output file exists and has content
    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError("Submission file was not created.")

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission file created with {len(df_sub)} rows.")

    assert (
        "id" in df_sub.columns and "value" in df_sub.columns
    ), "Submission columns missing"
    assert len(df_sub) > 0, "Submission file is empty"

    print("Inference pipeline verified.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
