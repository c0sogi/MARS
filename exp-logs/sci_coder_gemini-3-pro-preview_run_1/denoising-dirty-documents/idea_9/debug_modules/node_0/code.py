import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import library components
from library.config import Config
from library.utils import seed_everything, calculate_rmse
from library.dataset import get_dataloaders, get_test_dataloader
from library.models import UNet
from library.train import train_all_models
from library.inference import load_trained_models, predict_with_tta, generate_submission


def main():
    print("Starting Demo Script...")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Set a specific working directory for this demo to avoid conflicts
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config attributes
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "demo_submission.csv")

    # Limit data size
    Config.MAX_TRAIN_SAMPLES = 16  # Just enough for a few batches
    Config.MAX_VAL_SAMPLES = 8
    Config.BATCH_SIZE = 4

    # Minimize training duration
    Config.NUM_EPOCHS = 1
    Config.T_MAX = 1

    # Define a lightweight stream for the demo
    # Depth 2, small patch size, few channels
    DEMO_STREAM = {
        "name": "model",
        "patch_size": 64,
        "depth": 2,
        "base_channels": 8,
        "num_models": 1,
        "seeds": [42],
    }
    Config.STREAMS = [DEMO_STREAM]

    # Create a dummy test metadata file with only 1 image to speed up submission generation
    original_test_df = pd.read_csv(Config.TEST_METADATA)
    demo_test_df = original_test_df.head(1)
    demo_test_metadata_path = os.path.join(DEMO_DIR, "test.csv")
    demo_test_df.to_csv(demo_test_metadata_path, index=False)
    Config.TEST_METADATA = demo_test_metadata_path

    # Re-run setup to ensure directories exist
    Config.setup()
    seed_everything(Config.SEED)

    print("Configuration updated.")

    # -------------------------------------------------------------------------
    # 2. Verify Utils
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utils...")

    # Test RMSE calculation
    y_true = np.array([0.0, 1.0, 0.5])
    y_pred = np.array([0.0, 1.0, 0.5])
    rmse = calculate_rmse(y_true, y_pred)
    assert rmse == 0.0, f"RMSE should be 0.0 for identical arrays, got {rmse}"

    y_pred_off = np.array([1.0, 0.0, 1.5])  # Errors: 1, 1, 1 -> MSE=1 -> RMSE=1
    rmse_off = calculate_rmse(y_true, y_pred_off)
    assert np.isclose(rmse_off, 1.0), f"RMSE should be 1.0, got {rmse_off}"
    print("Utils verification passed.")

    # -------------------------------------------------------------------------
    # 3. Verify Dataset Pipeline
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Dataset Pipeline...")

    # Generate dataloaders
    # load_cached_data=False forces loading from disk and creating new cache in demo dir
    train_loader, val_loader = get_dataloaders(DEMO_STREAM, load_cached_data=False)

    # Check Train Loader
    assert len(train_loader) > 0, "Train loader is empty."
    inputs, targets, ids = next(iter(train_loader))

    # Verify shapes: (Batch, Channels, Height, Width)
    expected_shape = (
        Config.BATCH_SIZE,
        1,
        DEMO_STREAM["patch_size"],
        DEMO_STREAM["patch_size"],
    )
    assert (
        inputs.shape == expected_shape
    ), f"Input shape mismatch. Expected {expected_shape}, got {inputs.shape}"
    assert (
        targets.shape == expected_shape
    ), f"Target shape mismatch. Expected {expected_shape}, got {targets.shape}"

    print(f"Train batch shape verified: {inputs.shape}")

    # Check Test Loader
    test_loader = get_test_dataloader(load_cached_data=False)
    assert (
        len(test_loader) == 1
    ), f"Test loader should have 1 image (subset), got {len(test_loader)}"
    test_img, test_id = next(iter(test_loader))
    # Test loader returns full images (batch size 1), shape (1, 1, H, W)
    assert (
        test_img.ndim == 4 and test_img.shape[0] == 1
    ), f"Test image shape incorrect: {test_img.shape}"

    print("Dataset verification passed.")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    model = UNet(
        n_channels=1,
        n_classes=1,
        depth=DEMO_STREAM["depth"],
        base_channels=DEMO_STREAM["base_channels"],
    )

    # Forward pass with dummy data
    dummy_input = torch.randn(2, 1, 64, 64)
    output = model(dummy_input)

    assert (
        output.shape == dummy_input.shape
    ), f"Model output shape mismatch. Expected {dummy_input.shape}, got {output.shape}"
    print("Model forward pass verified.")

    # -------------------------------------------------------------------------
    # 5. Verify Training Loop
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Session...")

    # Train the model (this saves the model to disk)
    trained_model_paths = train_all_models(load_cached_data=True)

    assert len(trained_model_paths) == 1, "Should have trained exactly 1 model."
    model_path = trained_model_paths[0]
    assert os.path.exists(model_path), f"Model file not found at {model_path}"

    print(f"Training complete. Model saved to {model_path}")

    # -------------------------------------------------------------------------
    # 6. Verify Inference & Submission
    # -------------------------------------------------------------------------
    print("\n[6] Running Inference and Submission Generation...")

    # Load the trained model
    device = torch.device(Config.DEVICE)
    models = load_trained_models(device)
    assert len(models) == 1, "Should have loaded 1 model."

    # Test TTA prediction on a single tensor
    dummy_test_input = torch.randn(1, 1, 64, 64).to(device)
    tta_output = predict_with_tta(models[0], dummy_test_input)
    assert tta_output.shape == dummy_test_input.shape, "TTA output shape mismatch."

    # Generate Submission (using the 1-image test set configured earlier)
    generate_submission()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Check content format
    with open(Config.SUBMISSION_PATH, "r") as f:
        header = f.readline().strip()
        first_line = f.readline().strip()

    assert header == "id,value", f"Invalid header: {header}"
    assert len(first_line.split(",")) == 2, f"Invalid row format: {first_line}"

    print(f"Submission generated successfully at {Config.SUBMISSION_PATH}")
    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
