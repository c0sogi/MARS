import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch

# Import library modules
# We assume the library files are in ./library/ relative to this script
from library import config
from library import utils
from library import dataset
from library import model
from library import engine


def main():
    print("Starting Demonstration Script...")

    # =========================================================================
    # 1. SETUP & CONFIGURATION PATCHING
    # =========================================================================
    # We modify the config module at runtime to run a fast demo (1 epoch, small data)

    # Define paths for demo artifacts
    demo_dir = os.path.join(config.BASE_DIR, "working", "demo_execution")

    # Clean up stale cache to prevent loading mismatched data
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Create mini metadata files to speed up data loading
    # We will sample 50 training images, 20 validation images, and 20 test images
    print("Creating mini-datasets for demonstration...")

    full_train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    full_val_df = pd.read_csv(config.VAL_METADATA_PATH)
    full_test_df = pd.read_csv(config.TEST_METADATA_PATH)

    mini_train_df = full_train_df.head(50).copy()
    mini_val_df = full_val_df.head(20).copy()
    mini_test_df = full_test_df.head(20).copy()

    mini_train_path = os.path.join(demo_dir, "mini_train.csv")
    mini_val_path = os.path.join(demo_dir, "mini_val.csv")
    mini_test_path = os.path.join(demo_dir, "mini_test.csv")

    mini_train_df.to_csv(mini_train_path, index=False)
    mini_val_df.to_csv(mini_val_path, index=False)
    mini_test_df.to_csv(mini_test_path, index=False)

    # Patch the config module
    print("Patching library.config for fast execution...")
    config.TRAIN_METADATA_PATH = mini_train_path
    config.VAL_METADATA_PATH = mini_val_path
    config.TEST_METADATA_PATH = mini_test_path

    config.WORKING_DIR = demo_dir
    config.CHECKPOINT_DIR = demo_dir
    config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    config.SUBMISSION_PATH = os.path.join(config.SUBMISSION_DIR, "submission_demo.csv")

    # Reduce compute load
    config.EPOCHS = 1
    config.SEEDS = [42]  # Single seed
    config.BATCH_SIZE = 8
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    config.PATIENCE = 1

    # Ensure directories exist
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Set global seed
    utils.set_seed(42)

    # =========================================================================
    # 2. VERIFY MODEL ARCHITECTURE
    # =========================================================================
    print("\n--- Verifying Model Architecture ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate model
    net = model.WideSERes2NeXt(
        num_classes=config.NUM_CLASSES,
        stages=config.MODEL_PARAMS["stages"],
        cardinality=config.MODEL_PARAMS["cardinality"],
        base_width=config.MODEL_PARAMS["base_width"],
        res2net_scale=config.MODEL_PARAMS["res2net_scale"],
        se_reduction=config.MODEL_PARAMS["se_reduction"],
        use_gap=config.MODEL_PARAMS["use_gap"],
        dropout_rate=config.MODEL_PARAMS["dropout_rate"],
    ).to(device)

    # Create dummy input: Batch Size 2, 3 Channels, 32x32 Image
    dummy_input = torch.randn(2, 3, 32, 32).to(device)

    # Forward pass
    output = net(dummy_input)

    # Check output shape (Batch Size, Num Classes)
    expected_shape = (2, 1)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"
    print("Model forward pass successful. Output shape verified.")

    # =========================================================================
    # 3. VERIFY DATASET LOADING
    # =========================================================================
    print("\n--- Verifying Dataset Loading ---")

    # Initialize dataset with mini metadata
    # Note: We use load_cached_data=False to force loading from disk for verification
    ds = dataset.CactusDataset(
        metadata_path=config.TRAIN_METADATA_PATH,
        phase="train",
        transform=dataset.get_transforms("train"),
        load_cached_data=False,
    )

    # Check length
    assert len(ds) == 50, f"Dataset length mismatch. Expected 50, got {len(ds)}"

    # Check item retrieval
    img, label, img_id = ds[0]

    # Check image tensor shape (Channels, Height, Width)
    assert img.shape == (
        3,
        32,
        32,
    ), f"Image tensor shape mismatch. Expected (3, 32, 32), got {img.shape}"
    assert isinstance(img, torch.Tensor), "Image is not a torch.Tensor"
    assert isinstance(
        label, (float, np.float32)
    ), f"Label type mismatch. Got {type(label)}"

    print("Dataset loading and transformation verified.")

    # =========================================================================
    # 4. RUN TRAINING AND INFERENCE ENGINE
    # =========================================================================
    print("\n--- Running Training and Inference Engine ---")

    # This will run training for 1 epoch on the mini dataset and generate predictions
    # engine.run() uses the patched config values
    engine.run()

    # =========================================================================
    # 5. VERIFY SUBMISSION
    # =========================================================================
    print("\n--- Verifying Submission ---")

    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(config.SUBMISSION_PATH)

    # Check columns
    assert "id" in sub_df.columns, "Submission missing 'id' column"
    assert "has_cactus" in sub_df.columns, "Submission missing 'has_cactus' column"

    # Check length (should match mini test set size)
    assert (
        len(sub_df) == 20
    ), f"Submission length mismatch. Expected 20, got {len(sub_df)}"

    # Check values are probabilities (0 to 1)
    # Note: Since we use TTA and Sigmoid, values should be strictly within [0, 1]
    assert sub_df["has_cactus"].min() >= 0.0, "Probabilities below 0 found"
    assert sub_df["has_cactus"].max() <= 1.0, "Probabilities above 1 found"

    print("Submission file verified successfully.")
    print("\nDemonstration complete!")


if __name__ == "__main__":
    main()
