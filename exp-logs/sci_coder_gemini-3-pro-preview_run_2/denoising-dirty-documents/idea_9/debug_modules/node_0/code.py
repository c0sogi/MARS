import os
import torch
import numpy as np
import pandas as pd
import shutil
import warnings

# Import from the provided library
from library.config import Config
from library.dataset import DenoisingDataset
from library.model import CoRes2NetUNet
from library.trainer import Trainer
from library.inference import InferenceEngine
from library.utils import set_seed, calculate_rmse


def run_demo():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    print("--- Starting Demo Execution ---")

    # 1. Setup Configuration
    # Use debug=True to reduce epochs (2) and patches per image (10) for speed.
    config = Config(debug=True)

    # Redirect working directories to a demo folder to isolate outputs
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    config.WORKING_DIR = demo_dir
    config.CHECKPOINT_PATH = os.path.join(demo_dir, "checkpoint.pth")
    config.SUBMISSION_DIR = demo_dir
    config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Ensure cache directory exists within new working dir
    os.makedirs(os.path.join(demo_dir, "cache", "train"), exist_ok=True)
    os.makedirs(os.path.join(demo_dir, "cache", "val"), exist_ok=True)
    os.makedirs(os.path.join(demo_dir, "cache", "test"), exist_ok=True)

    # Set seed for reproducibility
    set_seed(config.SEED)
    config.print_config()

    # 2. Dataset Verification
    print("\n[1/5] Verifying Dataset...")
    train_ds = DenoisingDataset("train", config)

    # Check length
    assert len(train_ds) > 0, "Training dataset is empty."

    # Check item structure
    noisy, clean = train_ds[0]
    assert isinstance(noisy, torch.Tensor), "Dataset should return torch tensors."
    assert isinstance(clean, torch.Tensor), "Dataset should return torch tensors."

    # Check shapes (C, H, W) -> (1, 128, 128) based on Config.PATCH_SIZE
    expected_shape = (1, config.PATCH_SIZE, config.PATCH_SIZE)
    assert (
        noisy.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {noisy.shape}"
    assert (
        clean.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {clean.shape}"

    # Check value ranges (Normalized 0-1)
    assert (
        noisy.min() >= 0.0 and noisy.max() <= 1.0
    ), "Noisy image values out of range [0, 1]."
    assert (
        clean.min() >= 0.0 and clean.max() <= 1.0
    ), "Clean image values out of range [0, 1]."
    print("Dataset verification passed.")

    # 3. Model Verification
    print("\n[2/5] Verifying Model Architecture...")
    model = CoRes2NetUNet(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        base_filters=config.BASE_FILTERS,
    ).to(config.DEVICE)

    # Create dummy input batch (Batch Size=2, Channels=1, H=128, W=128)
    dummy_input = torch.randn(2, 1, config.PATCH_SIZE, config.PATCH_SIZE).to(
        config.DEVICE
    )

    # Forward pass
    try:
        output = model(dummy_input)
    except Exception as e:
        raise AssertionError(f"Model forward pass failed: {e}")

    # Check output shape
    assert (
        output.shape == dummy_input.shape
    ), f"Model output shape mismatch. Expected {dummy_input.shape}, got {output.shape}"
    print("Model verification passed.")

    # 4. Training Loop (Demo)
    print("\n[3/5] Running Training Loop (Debug Mode)...")
    trainer = Trainer(config)

    # Run fit (runs for 2 epochs due to debug=True)
    trainer.fit()

    # Verify checkpoint creation
    assert os.path.exists(config.CHECKPOINT_PATH), "Checkpoint file was not created."
    print("Training loop completed and checkpoint saved.")

    # 5. Inference and Submission
    print("\n[4/5] Running Inference and Submission Generation...")
    inference_engine = InferenceEngine(config)

    # Run Validation
    val_rmse = inference_engine.validate()
    print(f"Validation RMSE: {val_rmse:.6f}")
    assert isinstance(
        val_rmse, (float, np.floating)
    ), "Validation RMSE should be a float."

    # Generate Submission
    inference_engine.generate_submission()

    # Verify submission file
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created."

    # basic content check
    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    assert (
        "id" in df_sub.columns and "value" in df_sub.columns
    ), "Submission file missing required columns."
    assert len(df_sub) > 0, "Submission file is empty."
    print("Inference and submission generation passed.")

    # 6. Utility Verification
    print("\n[5/5] Verifying Utilities...")
    y_true = np.array([0.0, 1.0, 0.5])
    y_pred = np.array([0.0, 1.0, 0.5])
    rmse_perfect = calculate_rmse(y_true, y_pred)
    assert rmse_perfect == 0.0, "RMSE calculation for identical arrays should be 0."

    y_pred_off = np.array([1.0, 0.0, 1.5])  # Errors: 1, 1, 1 -> MSE=1 -> RMSE=1
    rmse_off = calculate_rmse(y_true, y_pred_off)
    assert np.isclose(
        rmse_off, 1.0
    ), f"RMSE calculation incorrect. Expected 1.0, got {rmse_off}"
    print("Utility verification passed.")

    print("\n--- Demo Execution Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
