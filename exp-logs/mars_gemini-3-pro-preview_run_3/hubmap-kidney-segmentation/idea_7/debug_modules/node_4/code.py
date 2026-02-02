import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
import shutil
from library.config import Config
from library.utils import set_seed, rle_encode, rle_decode
from library.dataset import HubmapDataset
from library.model import build_model
from library.train import train_model
from library.inference import run_inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting HuBMAP FTU Detection Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Fast Demo Execution
    # -------------------------------------------------------------------------
    print("[1/7] Configuring environment for demo run...")

    # Override Config parameters to ensure speed and isolation
    Config.IDEA_NAME = "demo_run"
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 4  # Use very few samples

    # Update paths to use a specific demo working directory
    Config.WORKING_DIR = os.path.join("./working", Config.IDEA_NAME)
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Optimize training parameters for speed (1 epoch, small tiles)
    Config.PHASE1 = {
        "EPOCHS": 1,
        "BATCH_SIZE": 2,
        "TILE_SIZE": 256,  # Reduced from 512
        "LR": 1e-4,
    }
    Config.PHASE2 = {
        "EPOCHS": 1,
        "BATCH_SIZE": 2,
        "TILE_SIZE": 256,  # Reduced from 768
        "LR": 1e-4,
    }

    # Inference settings
    Config.INFERENCE_OVERLAP = 0.1  # Reduce overlap for speed

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("      Configuration updated. Working dir:", Config.WORKING_DIR)

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions (RLE)
    # -------------------------------------------------------------------------
    print("\n[2/7] Verifying RLE encoding/decoding logic...")

    # Create a synthetic 10x10 binary mask
    # Pattern: A 3x3 square in the middle
    dummy_mask = np.zeros((10, 10), dtype=np.uint8)
    dummy_mask[3:6, 3:6] = 1

    # Encode
    rle_str = rle_encode(dummy_mask)

    # Decode
    decoded_mask = rle_decode(rle_str, (10, 10))

    # Assert equality
    if not np.array_equal(dummy_mask, decoded_mask):
        raise AssertionError("RLE Encode -> Decode failed: Masks do not match.")

    print("      RLE verification passed.")

    # -------------------------------------------------------------------------
    # 3. Verify Dataset Loading
    # -------------------------------------------------------------------------
    print("\n[3/7] Verifying Dataset pipeline...")

    # Load metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH).head(Config.DEBUG_SAMPLE_SIZE)

    # Instantiate Dataset
    dataset = HubmapDataset(
        metadata_df=df_train,
        phase="train",
        image_size=256,
        samples_per_epoch=2,  # Small number of samples
        load_cached_data=True,
    )

    # Fetch one item
    img_tensor, mask_tensor = dataset[0]

    # Verify Shapes
    # Image: (3, 256, 256), Mask: (1, 256, 256)
    if img_tensor.shape != (3, 256, 256):
        raise AssertionError(f"Incorrect image tensor shape: {img_tensor.shape}")
    if mask_tensor.shape != (1, 256, 256):
        raise AssertionError(f"Incorrect mask tensor shape: {mask_tensor.shape}")

    # Verify Types
    if not isinstance(img_tensor, torch.Tensor) or not isinstance(
        mask_tensor, torch.Tensor
    ):
        raise AssertionError("Dataset did not return torch Tensors.")

    print(f"      Dataset verification passed. Image shape: {img_tensor.shape}")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4/7] Verifying Model architecture...")

    model = build_model()

    # Create dummy input batch (Batch Size=2, Channels=3, H=256, W=256)
    dummy_input = torch.randn(2, 3, 256, 256)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input)

    # Check Deep Supervision output
    # Unet++ with deep_supervision=True returns a list of tensors
    if not isinstance(outputs, list):
        raise AssertionError("Model output is not a list (Deep Supervision expected).")

    # Check shape of the final output (index 0 usually corresponds to final layer in some implementations,
    # but SMP Unet++ usually returns [output_L1, output_L2, ..., output_final].
    # Actually, SMP implementation returns a list where the last element is often the final prediction
    # or ordered by depth. Let's check dimensions of all outputs.
    # They should all be (B, Classes, H, W) because SMP upsamples them.

    for i, out in enumerate(outputs):
        if out.shape != (2, 1, 256, 256):
            raise AssertionError(f"Model output {i} has incorrect shape: {out.shape}")

    print(f"      Model verification passed. Deep supervision outputs: {len(outputs)}")

    # -------------------------------------------------------------------------
    # 5. Run Training Loop
    # -------------------------------------------------------------------------
    print("\n[5/7] Running Training Loop (Demo Mode)...")

    # We call the provided train_model function.
    # It uses Config settings we overrode earlier.
    try:
        train_model()
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")

    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise AssertionError("Training finished but 'best_model.pth' was not created.")

    print("      Training finished successfully. Model saved.")

    # -------------------------------------------------------------------------
    # 6. Run Inference
    # -------------------------------------------------------------------------
    print("\n[6/7] Running Inference on Test Set...")

    # We call the provided run_inference function.
    try:
        run_inference()
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError("Inference finished but 'submission.csv' was not created.")

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    if len(sub_df) != len(test_df):
        raise AssertionError(
            f"Submission has {len(sub_df)} rows, expected {len(test_df)}."
        )

    if list(sub_df.columns) != ["id", "predicted"]:
        raise AssertionError(f"Submission columns incorrect: {sub_df.columns}")

    print(f"      Inference finished. Submission generated with {len(sub_df)} rows.")

    # -------------------------------------------------------------------------
    # 7. Final Success Message
    # -------------------------------------------------------------------------
    print("\n[7/7] Demo Completed Successfully!")
    print(f"      Artifacts stored in: {Config.WORKING_DIR}")


if __name__ == "__main__":
    main()
