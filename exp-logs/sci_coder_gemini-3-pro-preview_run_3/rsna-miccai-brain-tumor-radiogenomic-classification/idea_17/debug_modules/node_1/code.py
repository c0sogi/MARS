import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Import from the provided library
from library.config import Config
from library.utils import generate_strided_indices
from library.dataset import SSVEDataset
from library.model import SSVEModel
from library.train import run_training
from library.predict import run_inference


def setup_demo_environment():
    """
    Sets up a lightweight environment for the demo by creating subset metadata
    and overriding Config paths. This ensures the code runs in seconds/minutes
    rather than hours.
    """
    print("[Demo] Setting up demo environment...")

    # Define demo directories
    demo_work_dir = "./working/demo_execution"
    demo_meta_dir = "./working/demo_execution/metadata"
    os.makedirs(demo_work_dir, exist_ok=True)
    os.makedirs(demo_meta_dir, exist_ok=True)

    # 1. Create Subset Metadata
    # We read the original metadata, take the top 4 rows, and save to demo location.
    # This forces the Dataset class to only process these 4 patients.
    for split in ["train", "val", "test"]:
        src_path = os.path.join(Config.METADATA_DIR, f"{split}.parquet")
        dst_path = os.path.join(demo_meta_dir, f"{split}.parquet")

        if os.path.exists(src_path):
            df = pd.read_parquet(src_path)
            # Take a tiny subset (e.g., 4 samples)
            subset_df = df.head(4).copy()
            subset_df.to_parquet(dst_path, index=False)
            print(f"  Created subset metadata for {split}: {len(subset_df)} samples")
        else:
            print(f"  Warning: Source metadata {src_path} not found.")

    # 2. Override Config Global Settings for Speed
    print("[Demo] Overriding Config parameters...")
    Config.WORKING_DIR = demo_work_dir
    Config.METADATA_DIR = demo_meta_dir
    Config.TRAIN_META_PATH = os.path.join(demo_meta_dir, "train.parquet")
    Config.VAL_META_PATH = os.path.join(demo_meta_dir, "val.parquet")
    Config.TEST_META_PATH = os.path.join(demo_meta_dir, "test.parquet")
    Config.MODEL_PATH = os.path.join(demo_work_dir, "best_model.pth")
    Config.SUBMISSION_DIR = demo_work_dir
    Config.SUBMISSION_PATH = os.path.join(demo_work_dir, "demo_submission.csv")

    # Clean up stale model artifacts to prevent "ghost" bugs
    if os.path.exists(Config.MODEL_PATH):
        os.remove(Config.MODEL_PATH)

    # Hyperparameters for Demo
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # Re-run setup to ensure new dirs exist
    Config.setup()


def verify_utils():
    """
    Verifies the logic of utility functions.
    """
    print("\n[Verification] Testing Utility Functions...")

    # Test generate_strided_indices
    # Scenario: 100 slices. We expect 32 slices total (16 View A, 16 View B)
    # The function samples from 10% (10) to 90% (90).
    num_files = 100
    indices = generate_strided_indices(num_files)

    assert "view_a" in indices and "view_b" in indices, "Missing views in indices dict"
    assert (
        len(indices["view_a"]) == 16
    ), f"View A should have 16 slices, got {len(indices['view_a'])}"
    assert (
        len(indices["view_b"]) == 16
    ), f"View B should have 16 slices, got {len(indices['view_b'])}"

    # Verify disjointness (View A is even indices of the linspace, View B is odd)
    # Note: In the implementation, they come from the same linspace array.
    # We just check they are integers and within range.
    all_indices = indices["view_a"] + indices["view_b"]
    assert all(0 <= i < num_files for i in all_indices), "Indices out of bounds"

    print("  generate_strided_indices: OK")


def verify_model():
    """
    Verifies the model architecture and forward pass.
    """
    print("\n[Verification] Testing Model Architecture...")

    model = SSVEModel()
    model.eval()

    # Create dummy input: (Batch=2, Channels=64, Height=256, Width=256)
    # Channels = 16 slices * 4 modalities = 64
    dummy_input = torch.randn(2, 64, 256, 256)

    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape: (Batch, 1)
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"
    print("  SSVEModel Forward Pass: OK")


def verify_dataset_loading():
    """
    Verifies that the dataset loads data correctly using the subset metadata.
    """
    print("\n[Verification] Testing Dataset Loading...")

    # Initialize dataset (this will trigger processing and caching of the 4 subset samples)
    # We set load_cached_data=False to force the processing logic to run for the demo
    ds = SSVEDataset(mode="train", load_cached_data=False)

    assert len(ds) == 4, f"Expected 4 samples in subset, found {len(ds)}"

    # Fetch one item
    img, target = ds[0]

    # Check shapes
    # Train mode returns (64, 256, 256) - one view
    assert img.shape == (
        64,
        256,
        256,
    ), f"Expected image shape (64, 256, 256), got {img.shape}"
    assert isinstance(target.item(), float), "Target should be a float"

    print("  SSVEDataset (Train): OK")


def run_demo_pipeline():
    """
    Executes the training and inference pipeline using the library functions.
    """
    print("\n[Pipeline] Starting Training Demo...")
    # Run training (uses Config.NUM_EPOCHS=1, Config.BATCH_SIZE=2)
    # We force reload to ensure the subset data is used
    run_training(load_cached_data=True)

    print("\n[Pipeline] Starting Inference Demo...")
    # Run inference on test subset
    run_inference(load_cached_data=True)

    # Verify Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"\n[Output] Submission generated at {Config.SUBMISSION_PATH}")
        print(sub_df)

        # Check format
        assert "BraTS21ID" in sub_df.columns
        assert "MGMT_value" in sub_df.columns
        assert len(sub_df) == 4, "Submission should have 4 rows (matching test subset)"
        print("  Submission Format: OK")
    else:
        raise FileNotFoundError("Submission file was not generated.")


if __name__ == "__main__":
    # 1. Setup Environment (Subset Data & Config Override)
    setup_demo_environment()

    # 2. Verify Components
    verify_utils()
    verify_model()
    verify_dataset_loading()

    # 3. Run Full Pipeline
    run_demo_pipeline()

    print("\n" + "=" * 40)
    print(" DEMONSTRATION COMPLETE")
    print("=" * 40)
