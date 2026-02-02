import os
import shutil
import torch
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, pad_to_multiple, unpad, rmse_loss
from library.dataset import get_dataloaders
from library.model import UNet
from library.train import run_training
from library.predict import generate_submission, predict_with_tta_ensemble


def main():
    print(">>> Starting Library Usage Demonstration...")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    # We override the default Config settings to run a fast, lightweight demo.

    # Set a fixed seed for reproducibility
    seed_everything(42)

    # Define a specific working directory for this demo
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config paths and parameters
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "demo_submission.csv")

    # Enable Debug mode to use a small subset of data (fast loading)
    Config.DEBUG = True
    Config.MAX_DEBUG_SAMPLES = 20

    # Training parameters for speed
    Config.EPOCHS = 1
    Config.SEEDS = [42]  # Train only one model instead of an ensemble of 5
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for this small script

    print(f"Configuration configured. Working directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Utility Functions...")

    # Test Padding Logic
    # Create a tensor that is NOT divisible by 16 (e.g., 50x50)
    # Shape: (Batch, Channel, Height, Width)
    x = torch.randn(1, 1, 50, 50)
    padded_x, padding_info = pad_to_multiple(x, divisor=16)

    # 50 -> next multiple of 16 is 64
    expected_shape = (1, 1, 64, 64)
    assert (
        padded_x.shape == expected_shape
    ), f"Padding failed. Expected {expected_shape}, got {padded_x.shape}"

    # Test Unpadding Logic
    unpadded_x = unpad(padded_x, padding_info)
    assert (
        unpadded_x.shape == x.shape
    ), f"Unpadding failed. Expected {x.shape}, got {unpadded_x.shape}"
    assert torch.allclose(x, unpadded_x), "Unpadded tensor content mismatch."

    # Test RMSE Loss
    t1 = torch.ones(10)
    t2 = torch.zeros(10)
    loss = rmse_loss(t1, t2)
    # sqrt(mean((1-0)^2)) = 1.0
    assert torch.isclose(
        loss, torch.tensor(1.0)
    ), f"RMSE calculation error. Expected 1.0, got {loss.item()}"

    print("Utils verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Verify Dataset and DataLoader
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Data Loading...")

    # Initialize DataLoaders (this will trigger cache creation in the demo dir)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify Train Loader
    try:
        batch = next(iter(train_loader))
        noisy_imgs, clean_imgs, ids = batch

        # Check dimensions: (Batch, Channel, Patch, Patch)
        # Config.PATCH_SIZE is 320
        assert noisy_imgs.ndim == 4
        assert noisy_imgs.shape[2] == Config.PATCH_SIZE
        assert noisy_imgs.shape[3] == Config.PATCH_SIZE
        assert clean_imgs.shape == noisy_imgs.shape

        # Check value normalization [0, 1]
        assert (
            noisy_imgs.min() >= 0.0 and noisy_imgs.max() <= 1.0
        ), "Train input out of range."
        assert (
            clean_imgs.min() >= 0.0 and clean_imgs.max() <= 1.0
        ), "Train target out of range."

        print(f"Train loader verified. Batch shape: {noisy_imgs.shape}")
    except StopIteration:
        raise AssertionError("Train loader is empty.")

    # Verify Validation Loader (Batch size 1, full image size)
    try:
        val_batch = next(iter(val_loader))
        v_noisy, v_clean, v_ids = val_batch
        assert v_noisy.shape[0] == 1, "Validation batch size must be 1."
        print(f"Validation loader verified. Image shape: {v_noisy.shape}")
    except StopIteration:
        raise AssertionError("Validation loader is empty.")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Model Architecture...")

    device = Config.DEVICE
    model = UNet(n_channels=1, n_classes=1).to(device)

    # Perform a forward pass with the training batch
    noisy_input = noisy_imgs.to(device)
    with torch.no_grad():
        output = model(noisy_input)

    # Check output shape
    assert (
        output.shape == noisy_input.shape
    ), f"Model output shape mismatch. Expected {noisy_input.shape}, got {output.shape}"

    # Check Sigmoid activation (values must be in [0, 1])
    assert (
        output.min() >= 0.0 and output.max() <= 1.0
    ), "Model output values out of range [0, 1]."

    print("Model instantiated and forward pass verified.")

    # -------------------------------------------------------------------------
    # 5. Run Training Pipeline
    # -------------------------------------------------------------------------
    print("\n>>> Running Training Pipeline (1 Epoch)...")

    # Execute training
    # This uses the overridden Config values (1 epoch, 1 seed)
    run_training(epochs=Config.EPOCHS, debug=Config.DEBUG)

    # Verify that the model file was saved
    expected_model_path = os.path.join(
        Config.WORKING_DIR, f"model_seed_{Config.SEEDS[0]}.pth"
    )
    assert os.path.exists(
        expected_model_path
    ), f"Model file not found: {expected_model_path}"

    print(f"Training complete. Model saved to {expected_model_path}")

    # -------------------------------------------------------------------------
    # 6. Run Prediction and Submission Generation
    # -------------------------------------------------------------------------
    print("\n>>> Running Prediction Pipeline...")

    # Verify single inference with TTA before full generation
    model.eval()
    # Load the trained weights
    model.load_state_dict(torch.load(expected_model_path, map_location=device))

    # Get a test image
    test_batch = next(iter(test_loader))
    t_noisy, t_ids = test_batch

    # Run TTA inference
    pred_tta = predict_with_tta_ensemble([model], t_noisy, device)

    assert pred_tta.shape == t_noisy.shape, "TTA Prediction shape mismatch."
    print("Inference with TTA verified.")

    # Generate full submission file
    print("Generating submission file...")
    generate_submission()

    # Verify submission file existence and format
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    assert "id" in df_sub.columns, "Column 'id' missing in submission."
    assert "value" in df_sub.columns, "Column 'value' missing in submission."

    # Check that we have rows
    assert len(df_sub) > 0, "Submission file is empty."

    # Check value validity
    assert (
        df_sub["value"].min() >= 0.0 and df_sub["value"].max() <= 1.0
    ), "Submission values contain invalid intensities (outside 0-1)."

    print(f"Submission generated successfully at {Config.SUBMISSION_PATH}")
    print(f"Total pixels predicted: {len(df_sub)}")

    print("\n>>> Demonstration Completed Successfully.")


if __name__ == "__main__":
    main()
