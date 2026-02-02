import os
import torch
import pandas as pd
import numpy as np
import warnings
import sys

# Import library components
from library.config import Config
from library.dataset import InkDataset, get_transforms
from library.architecture import MIPUNet
from library.engine import train_model
from library.inference import create_submission
from library.utils import seed_everything


def run_demo():
    # 1. Setup and Configuration
    # --------------------------
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    print("Initializing Vesuvius Ink Detection Demo...")

    # Set seed for reproducibility
    seed_everything(42)

    # Override Config parameters for a fast demonstration run
    print("Overriding configuration for speed...")
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.WORKING_DIR = "./working/demo_run"  # Separate working dir
    Config.SUBMISSION_PATH = "./working/demo_submission.csv"

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Dataset Verification
    # -----------------------
    print("\n[1/5] Verifying Dataset Logic...")

    # Instantiate dataset with training transforms
    ds = InkDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        transform=get_transforms("train"),
        load_cached_data=True,  # Use caching to speed up if available
    )

    # Check length
    print(f"    Dataset size: {len(ds)}")
    if len(ds) == 0:
        raise ValueError("Dataset is empty. Check metadata files.")

    # Get one sample
    image, mask = ds[0]

    # Assertions
    # Image should be (C, H, W). Config.IN_CHANNELS is 1.
    assert image.ndim == 3, f"Image tensor should be 3D, got {image.ndim}"
    assert (
        image.shape[0] == Config.IN_CHANNELS
    ), f"Expected {Config.IN_CHANNELS} channel(s), got {image.shape[0]}"
    assert (
        image.shape[1] == Config.TILE_SIZE
    ), f"Expected height {Config.TILE_SIZE}, got {image.shape[1]}"
    assert (
        image.shape[2] == Config.TILE_SIZE
    ), f"Expected width {Config.TILE_SIZE}, got {image.shape[2]}"

    # Mask should be (1, H, W)
    assert mask.ndim == 3, f"Mask tensor should be 3D, got {mask.ndim}"
    assert mask.shape == (
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), f"Unexpected mask shape: {mask.shape}"

    print("    Dataset shapes verified successfully.")

    # 3. Architecture Verification
    # ----------------------------
    print("\n[2/5] Verifying Model Architecture...")

    # Instantiate model
    model = MIPUNet(
        encoder_name=Config.ENCODER_NAME,
        encoder_weights=None,  # Skip downloading weights for speed/offline check if needed, or use defaults
        in_channels=Config.IN_CHANNELS,
        classes=Config.CLASSES,
    )
    model.eval()

    # Create dummy input batch (B, C, H, W)
    dummy_input = torch.randn(2, Config.IN_CHANNELS, Config.TILE_SIZE, Config.TILE_SIZE)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape: (B, Classes, H, W)
    expected_shape = (2, Config.CLASSES, Config.TILE_SIZE, Config.TILE_SIZE)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print("    Model architecture forward pass successful.")

    # 4. Training Loop Execution
    # --------------------------
    print("\n[3/5] Executing Training Loop (1 Epoch)...")

    # train_model handles dataloading, training, validation, and saving the best model
    best_model_path = train_model(load_cached_data=True)

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Training finished but model file not found at {best_model_path}"
        )

    print(f"    Training complete. Model saved to: {best_model_path}")

    # 5. Inference Execution
    # ----------------------
    print("\n[4/5] Executing Inference Pipeline...")

    # Run inference on test set defined in metadata/test.csv
    create_submission(
        model_path=best_model_path,
        submission_output_path=Config.SUBMISSION_PATH,
        test_metadata_path=Config.TEST_METADATA_PATH,
        threshold=0.5,
        load_cached_data=True,
    )

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Inference finished but submission file not found at {Config.SUBMISSION_PATH}"
        )

    print(f"    Inference complete. Submission saved to: {Config.SUBMISSION_PATH}")

    # 6. Submission Validation
    # ------------------------
    print("\n[5/5] Validating Submission File...")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    expected_cols = ["Id", "Predicted"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Check content
    if len(df_sub) > 0:
        sample_rle = df_sub.iloc[0]["Predicted"]
        # RLE should be a string of space-separated numbers
        if pd.notna(sample_rle) and len(str(sample_rle)) > 0:
            rle_parts = str(sample_rle).split()
            # Basic check: RLE must be pairs, so length must be even
            assert (
                len(rle_parts) % 2 == 0
            ), "RLE string does not contain pairs of values (length is odd)."
            # Check if values are numeric
            try:
                _ = [int(x) for x in rle_parts[:10]]  # Check first few
            except ValueError:
                raise AssertionError("RLE string contains non-numeric characters.")
        print(f"    Submission file format verified. Rows: {len(df_sub)}")
    else:
        print("    Submission file is empty (no test fragments found).")

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    run_demo()
