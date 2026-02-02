import os
import sys
import shutil
import numpy as np
import torch
import pandas as pd
import cv2
from pathlib import Path

# Import provided library modules
from library.config import Config
from library.utils import rle_encode, f05_score, get_boundary_mask
from library.model import WSDN_ABS
from library.loss import JointLoss
from library.dataset import InkDataset, InferenceDataset
import library.train
import library.inference


def run_demo():
    print("--- Starting Demo Script ---")

    # --- 1. Setup & Configuration Overrides ---
    print("\n[1] Setting up environment and configuration...")

    # Define demo working directory
    DEMO_DIR = Path("./working/demo_execution")
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Attempt to copy cached data to speed up execution
    # We look for cached .npy files in working/demo_cache/ or similar locations mentioned in the prompt
    CACHE_SOURCE = Path("./working/demo_cache")
    if CACHE_SOURCE.exists():
        print(f"Copying cached files from {CACHE_SOURCE} to {DEMO_DIR}...")
        for f in CACHE_SOURCE.glob("*.npy"):
            dest = DEMO_DIR / f.name
            if not dest.exists():
                # Symlink is faster, but copy ensures safety. Using copy here.
                shutil.copy(f, dest)

    # Create dummy normalization stats to skip computation
    # Mean ~100, Std ~20 is typical for this data based on EDA
    stats_path = DEMO_DIR / "normalization_stats.npy"
    if not stats_path.exists():
        np.save(stats_path, np.array([100.0, 20.0]))

    # Override Config for Speed
    Config.WORKING_DIR = DEMO_DIR
    Config.SEED = 42
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.SAMPLES_PER_EPOCH = 10  # Very few samples
    Config.BATCH_SIZE = 2  # Small batch
    Config.PATCH_SIZE = 64  # Small patches (faster convolution)
    Config.INFERENCE_STRIDE = 32  # Overlap for inference
    Config.MODEL_CHANNELS = 8  # Tiny model
    Config.DILATION_RATES = [1, 2]  # Shallow backbone
    Config.USE_TTA = False  # Disable TTA for speed
    Config.THRESHOLD_STEP = 0.5  # Coarse threshold search
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set seeds
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    # --- 2. Verify Utilities ---
    print("\n[2] Verifying Utility Functions...")

    # Test RLE Encoding
    # Mask: 0 1 1 0
    #       1 1 0 0
    # Flattened: 0 1 1 0 1 1 0 0
    # Indices (1-based): 2,3, 5,6. Runs: Start 2 len 2, Start 5 len 2.
    dummy_mask = np.array([[0, 1, 1, 0], [1, 1, 0, 0]], dtype=np.uint8)
    rle_out = rle_encode(dummy_mask)
    expected_rle = "2 2 5 2"
    assert (
        rle_out == expected_rle
    ), f"RLE failed. Got {rle_out}, expected {expected_rle}"
    print("  -> rle_encode passed.")

    # Test F0.5 Score
    preds = np.array([1, 0, 1, 1])
    labels = np.array([1, 0, 0, 1])
    # TP=2, FP=1, FN=0. Precision=2/3, Recall=2/2=1.
    # F0.5 = (1.25 * 0.66 * 1) / (0.25 * 0.66 + 1) = 0.833 / 1.166 ~= 0.714
    score = f05_score(preds, labels)
    assert 0.7 < score < 0.72, f"F0.5 score incorrect. Got {score}"
    print("  -> f05_score passed.")

    # Test Boundary Mask
    # 5x5 image with 3x3 square in middle
    b_mask_in = np.zeros((5, 5), dtype=np.uint8)
    b_mask_in[1:4, 1:4] = 1
    boundary = get_boundary_mask(b_mask_in)
    # Boundary should be the edge of the square
    assert boundary.sum() > 0, "Boundary mask is empty."
    assert boundary.shape == (5, 5), "Boundary mask shape mismatch."
    print("  -> get_boundary_mask passed.")

    # --- 3. Verify Dataset & Model ---
    print("\n[3] Verifying Dataset and Model...")

    # Initialize Dataset
    # We use 'train' split. Ensure metadata exists (it is provided in environment).
    ds = InkDataset(split="train", load_cached_data=True)
    assert len(ds) == Config.SAMPLES_PER_EPOCH, "Dataset length mismatch."

    # Get a batch
    vol, targets = ds[0]
    assert vol.shape == (
        Config.Z_DIM,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), f"Volume shape mismatch: {vol.shape}"
    assert targets["mask"].shape == (
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), f"Mask shape mismatch: {targets['mask'].shape}"

    # Initialize Model
    model = WSDN_ABS(
        in_channels=Config.Z_DIM,
        model_channels=Config.MODEL_CHANNELS,
        dilation_rates=Config.DILATION_RATES,
    )

    # Forward Pass
    # Create a batch dimension
    input_tensor = vol.unsqueeze(0)  # (1, 65, 64, 64)
    outputs = model(input_tensor)

    assert "mask" in outputs and "boundary" in outputs, "Model output keys missing."
    assert outputs["mask"].shape == (
        1,
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), f"Model output shape mismatch: {outputs['mask'].shape}"
    print("  -> Model forward pass passed.")

    # Loss Calculation
    criterion = JointLoss()
    loss = criterion(outputs, {k: v.unsqueeze(0) for k, v in targets.items()})
    assert not torch.isnan(loss), "Loss is NaN."
    assert loss.item() > 0, "Loss should be positive."
    print(f"  -> Loss calculation passed. Loss: {loss.item():.4f}")

    # --- 4. Execute Training ---
    print("\n[4] Executing Training Loop...")

    # We call the library's train function directly.
    # It uses the Config we modified.
    library.train.train()

    best_model_path = Config.WORKING_DIR / "best_model.pth"
    assert best_model_path.exists(), "Training failed to produce best_model.pth"
    print("  -> Training completed successfully.")

    # --- 5. Execute Inference ---
    print("\n[5] Executing Inference Pipeline...")

    # We call the library's inference function directly.
    library.inference.inference()

    submission_path = Path("./submission.csv")
    assert submission_path.exists(), "Inference failed to produce submission.csv"

    df_sub = pd.read_csv(submission_path)
    assert not df_sub.empty, "Submission file is empty."
    assert (
        "Id" in df_sub.columns and "Predicted" in df_sub.columns
    ), "Submission columns mismatch."
    print("  -> Inference completed successfully.")
    print(f"  -> Submission Head:\n{df_sub.head()}")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
