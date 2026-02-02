import os
import sys
import torch
import numpy as np
import pandas as pd

# Ensure the current directory is in the path so library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config, seed_everything
from library.utils import calculate_rmse
from library.dataset import DenoisingDataset
from library.model import AttentionUNet
from library.train import train_single_model
from library.inference import generate_submission


def main():
    print("Starting Denoising Pipeline Demonstration...")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Set seed for reproducibility
    seed_everything(42)

    # Modify Config for a fast demonstration
    print("Configuring for fast execution...")
    Config.NUM_MODELS = 1  # Train only one model instead of the ensemble
    Config.BATCH_SIZE = 2  # Small batch size for debug dataset
    Config.NUM_WORKERS = 0  # Disable multiprocessing to avoid overhead in short run

    # Ensure directories exist (Config.setup() does this, but good to be sure)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # 2. Logic Verification: Utils
    # -------------------------------------------------------------------------
    print("Verifying Utility Functions...")
    # Test RMSE calculation
    y_true = np.array([0.0, 0.5, 1.0])
    y_pred = np.array([0.0, 0.5, 1.0])
    rmse_perfect = calculate_rmse(y_true, y_pred)
    assert rmse_perfect == 0.0, f"Expected RMSE 0.0, got {rmse_perfect}"

    y_pred_bad = np.array([1.0, 1.5, 2.0])  # Error of 1.0 everywhere
    rmse_bad = calculate_rmse(y_true, y_pred_bad)
    assert np.isclose(rmse_bad, 1.0), f"Expected RMSE 1.0, got {rmse_bad}"
    print("Utils verified.")

    # -------------------------------------------------------------------------
    # 3. Logic Verification: Dataset
    # -------------------------------------------------------------------------
    print("Verifying Dataset...")

    # Test Train Dataset (Debug mode loads first 10 samples)
    train_ds = DenoisingDataset(split="train", debug=True)
    assert len(train_ds) > 0, "Train dataset is empty"

    # Check item structure: (noisy_patch, clean_patch)
    t_noisy, t_clean = train_ds[0]

    # Assertions for Train
    assert torch.is_tensor(t_noisy), "Train noisy image is not a tensor"
    assert torch.is_tensor(t_clean), "Train clean image is not a tensor"
    assert t_noisy.ndim == 3, f"Expected 3 dims (C,H,W), got {t_noisy.ndim}"
    # Check patch size (Config.PATCH_SIZE is 128)
    assert t_noisy.shape[1] == Config.PATCH_SIZE, "Incorrect patch height"
    assert t_noisy.shape[2] == Config.PATCH_SIZE, "Incorrect patch width"

    # Test Val Dataset (Should return full images)
    val_ds = DenoisingDataset(split="val", debug=True)
    v_noisy, v_clean = val_ds[0]
    assert v_noisy.shape == v_clean.shape, "Mismatch in val image shapes"

    # Test Test Dataset (Should return noisy image and ID)
    test_ds = DenoisingDataset(split="test", debug=True)
    test_noisy, test_id = test_ds[0]
    assert isinstance(test_id, str), "Test ID should be a string"
    assert test_noisy.ndim == 3, "Test image should be a tensor (C,H,W)"

    print("Datasets verified.")

    # -------------------------------------------------------------------------
    # 4. Logic Verification: Model
    # -------------------------------------------------------------------------
    print("Verifying Model Architecture...")
    device = torch.device(Config.DEVICE)
    model = AttentionUNet(n_channels=1, n_classes=1).to(device)

    # Create dummy input: Batch=2, Channel=1, Height=128, Width=128
    dummy_input = torch.randn(2, 1, 128, 128).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape
    assert (
        output.shape == dummy_input.shape
    ), f"Model output shape {output.shape} does not match input {dummy_input.shape}"
    print("Model architecture verified.")

    # -------------------------------------------------------------------------
    # 5. Execution: Training
    # -------------------------------------------------------------------------
    print("\nExecuting Training Demo (1 Model, 2 Epochs)...")

    # Train model 0. debug=True forces num_epochs=2 and uses the debug dataset subset.
    train_single_model(model_index=0, debug=True)

    # Verify artifact creation
    expected_model_path = os.path.join(Config.WORKING_DIR, "model_0.pth")
    if not os.path.exists(expected_model_path):
        raise FileNotFoundError(
            f"Training failed to produce model file at {expected_model_path}"
        )

    print("Training demo completed successfully.")

    # -------------------------------------------------------------------------
    # 6. Execution: Inference
    # -------------------------------------------------------------------------
    print("\nExecuting Inference Demo...")

    # Generate submission using the trained model. debug=True uses test subset.
    generate_submission(debug=True)

    # Verify submission file
    expected_sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if not os.path.exists(expected_sub_path):
        raise FileNotFoundError(
            f"Inference failed to produce submission file at {expected_sub_path}"
        )

    # Verify submission content format
    df = pd.read_csv(expected_sub_path)
    print(f"Submission generated with {len(df)} rows.")

    required_cols = {"id", "value"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Submission missing required columns. Found: {df.columns}")

    # Check that values are within valid range [0, 1] (allowing for slight float errors)
    # Note: Model output is sigmoid, so it is strictly (0, 1).
    min_val = df["value"].min()
    max_val = df["value"].max()

    if min_val < 0.0 or max_val > 1.0:
        print(
            f"Warning: Submission values out of expected range [0, 1]. Range: [{min_val}, {max_val}]"
        )

    print("Inference demo completed successfully.")
    print("\nAll demonstration steps passed.")


if __name__ == "__main__":
    main()
