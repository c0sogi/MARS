import sys
import os
import torch
import numpy as np
import pandas as pd
import shutil

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.dataset import DenoisingDataset
from library.model import ResUNetPlusPlus
from library.train import run_training
from library.inference import generate_submission, predict_tiled
from library.utils import set_seed, calculate_rmse


def main():
    print("=== Denoising Task Implementation Demo ===\n")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Override Config parameters for a fast, lightweight run
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.PATCHES_PER_IMAGE = 5  # Reduce sampling density for speed
    Config.DEBUG = True  # Enable debug mode (uses subset of data)
    Config.DEBUG_SAMPLES = 5  # Only use 5 images for training/val

    # Redirect output to a separate demo directory to avoid clutter/conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure directories exist (since we changed WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.join(Config.WORKING_DIR, "cache"), exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(42)
    print("    Configuration complete. Working directory set to:", Config.WORKING_DIR)

    # ---------------------------------------------------------
    # 2. Dataset Demonstration & Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying DenoisingDataset...")

    # Instantiate dataset in training mode
    # We force load_cached_data=False to demonstrate raw loading logic
    train_ds = DenoisingDataset(
        metadata_file=Config.TRAIN_METADATA,
        mode="train",
        patches_per_image=Config.PATCHES_PER_IMAGE,
        load_cached_data=False,
    )

    # Validation
    print(f"    Dataset initialized. Total samples (patches): {len(train_ds)}")
    assert len(train_ds) > 0, "Training dataset should not be empty."

    # Fetch a single sample
    noisy_patch, residual_patch = train_ds[0]

    # Verify shapes and types
    print(
        f"    Sample 0 Shapes -> Noisy: {noisy_patch.shape}, Residual: {residual_patch.shape}"
    )
    assert isinstance(noisy_patch, torch.Tensor), "Dataset output should be a Tensor."
    assert noisy_patch.shape == (
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), f"Expected noisy patch shape (1, {Config.PATCH_SIZE}, {Config.PATCH_SIZE})"
    assert residual_patch.shape == (
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), f"Expected residual patch shape (1, {Config.PATCH_SIZE}, {Config.PATCH_SIZE})"

    print("    Dataset verification passed.")

    # ---------------------------------------------------------
    # 3. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying ResUNetPlusPlus Model...")

    model = ResUNetPlusPlus()
    # Create a dummy batch: (Batch_Size, Channels, Height, Width)
    dummy_input = torch.randn(2, 1, 128, 128)

    # Test Train Mode (Deep Supervision enabled)
    model.train()
    out_train = model(dummy_input)
    print("    Forward pass (Train Mode) successful.")
    assert isinstance(
        out_train, list
    ), "Model in train mode should return a list (Deep Supervision)."
    assert (
        len(out_train) == 4
    ), "Deep Supervision should return 4 outputs (levels 0_1 to 0_4)."
    assert out_train[-1].shape == (2, 1, 128, 128), "Final output shape mismatch."

    # Test Eval Mode (Inference only)
    model.eval()
    with torch.no_grad():
        out_eval = model(dummy_input)
    print("    Forward pass (Eval Mode) successful.")
    assert isinstance(
        out_eval, torch.Tensor
    ), "Model in eval mode should return a single Tensor."
    assert out_eval.shape == (2, 1, 128, 128), "Inference output shape mismatch."

    print("    Model verification passed.")

    # ---------------------------------------------------------
    # 4. Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n[4] Running Training Loop (Debug Mode)...")

    # Run training using the library function
    # This handles data loading, model init, optimizer, loop, and saving
    run_training(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        debug=Config.DEBUG,
        load_cached_data=False,
    )

    # Verify model artifact creation
    assert os.path.exists(
        Config.MODEL_PATH
    ), f"Model file not found at {Config.MODEL_PATH}"
    print(f"    Training complete. Model saved to {Config.MODEL_PATH}")

    # ---------------------------------------------------------
    # 5. Inference Logic Verification
    # ---------------------------------------------------------
    print("\n[5] Verifying Inference Utilities...")

    # Test the Tiled Prediction logic specifically
    # Create a large random image (larger than tile size)
    large_h, large_w = 600, 600
    large_img = torch.randn(1, large_h, large_w)

    # Use CPU for this quick check to avoid moving model back and forth if unnecessary
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    tiled_output = predict_tiled(model, large_img, tile_size=128, overlap=0.25)

    print(
        f"    Tiled prediction input: {large_img.shape}, output: {tiled_output.shape}"
    )
    assert tiled_output.shape == (
        1,
        large_h,
        large_w,
    ), "Tiled output shape does not match input shape."
    print("    Tiled prediction verification passed.")

    # ---------------------------------------------------------
    # 6. Full Submission Generation
    # ---------------------------------------------------------
    print("\n[6] Generating Submission File...")

    # This function loads the model from Config.MODEL_PATH and processes the test set
    generate_submission()

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Check content format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission generated with {len(df_sub)} rows.")

    # Basic schema check
    assert (
        "id" in df_sub.columns and "value" in df_sub.columns
    ), "Submission CSV missing required columns."
    assert len(df_sub) > 0, "Submission CSV is empty."

    # Check value range (should be roughly 0-1, though noise might push slightly out before clipping)
    # The inference code clips to [0, 1], so we verify that.
    vals = df_sub["value"].values
    assert vals.min() >= 0 and vals.max() <= 1, "Submission values out of range [0, 1]."

    print("    Submission generation verification passed.")

    # ---------------------------------------------------------
    # 7. Metric Calculation Verification
    # ---------------------------------------------------------
    print("\n[7] Verifying Metric Calculation...")

    y_true = np.array([0.0, 1.0, 0.5])
    y_pred_perfect = np.array([0.0, 1.0, 0.5])
    y_pred_bad = np.array([1.0, 0.0, 1.5])

    rmse_perfect = calculate_rmse(y_true, y_pred_perfect)
    rmse_bad = calculate_rmse(y_true, y_pred_bad)

    print(f"    RMSE (Perfect): {rmse_perfect:.4f}")
    print(f"    RMSE (Bad):     {rmse_bad:.4f}")

    assert np.isclose(rmse_perfect, 0.0), "RMSE for identical arrays should be 0."
    assert rmse_bad > 0, "RMSE for different arrays should be > 0."

    print("    Metric verification passed.")

    print("\n=== All demonstrations completed successfully ===")


if __name__ == "__main__":
    main()
