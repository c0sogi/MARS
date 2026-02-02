import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.dataset import extract_patches, DenoisingDataset
from library.network import CAResDnCNN
from library.trainer import run_curriculum_training, set_seed
from library.inference import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Library Demonstration ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Define a specific working directory for this demo to avoid conflicts
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config parameters for speed
    Config.WORKING_DIR = demo_dir
    Config.CHECKPOINT_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "demo_submission.csv")

    # Cache files for demo
    Config.CACHE_FILE_STAGE_1 = os.path.join(demo_dir, "train_patches_s1.npy")
    Config.CACHE_TARGETS_STAGE_1 = os.path.join(demo_dir, "train_targets_s1.npy")
    Config.CACHE_FILE_VAL = os.path.join(demo_dir, "val_patches.npy")
    Config.CACHE_TARGETS_VAL = os.path.join(demo_dir, "val_targets.npy")

    # Training Hyperparameters for Demo
    Config.BATCH_SIZE = 8
    Config.MAX_EPOCHS_STAGE_1 = 1  # Run only 1 epoch
    Config.MAX_EPOCHS_STAGE_2 = 0  # Skip stage 2

    # Data Processing: Use very large stride to get few patches
    Config.STRIDE_STAGE_1 = 300
    Config.VAL_STRIDE = 300

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration updated.")

    # -------------------------------------------------------------------------
    # 2. Dataset & Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Dataset Logic...")

    # Test patch extraction
    # We expect this to create .npy files in the demo directory
    patches, targets = extract_patches(
        metadata_path=Config.TRAIN_METADATA_PATH,
        stride=Config.STRIDE_STAGE_1,
        patch_size=Config.PATCH_SIZE,
        cache_patches_path=Config.CACHE_FILE_STAGE_1,
        cache_targets_path=Config.CACHE_TARGETS_STAGE_1,
        load_cached_data=False,  # Force extraction
        is_test=False,
    )

    # Assertions
    assert isinstance(patches, np.ndarray), "Patches should be a numpy array"
    assert isinstance(targets, np.ndarray), "Targets should be a numpy array"
    assert patches.ndim == 3, f"Patches should be (N, H, W), got {patches.shape}"
    assert patches.shape == targets.shape, "Input and Target shapes must match"

    print(f"Extracted {len(patches)} patches with shape {patches.shape[1:]}.")

    # Test Dataset Class
    dataset = DenoisingDataset(patches, targets, augment=True)
    assert len(dataset) == len(patches)

    # Test __getitem__
    sample_in, sample_tar = dataset[0]
    assert torch.is_tensor(sample_in), "Dataset should return tensors"
    assert torch.is_tensor(sample_tar), "Dataset should return tensors"
    # Shape check: (C, H, W) where C=1
    assert sample_in.shape == (
        1,
        Config.PATCH_SIZE,
        Config.PATCH_SIZE,
    ), f"Unexpected input shape: {sample_in.shape}"

    print("Dataset verification successful.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = CAResDnCNN(
        in_channels=Config.IN_CHANNELS,
        out_channels=Config.OUT_CHANNELS,
        num_features=32,  # Reduced features for demo speed
        num_blocks=2,  # Reduced blocks for demo speed
    ).to(device)

    # Create dummy input: Batch=2, Channel=1, H=50, W=50
    dummy_input = torch.randn(2, 1, 50, 50).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Assertions
    assert (
        output.shape == dummy_input.shape
    ), f"Output shape {output.shape} does not match input shape {dummy_input.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    print("Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Verification
    # -------------------------------------------------------------------------
    print("\n[4] Running Training Demo (1 Epoch)...")

    # We rely on the overridden Config to keep this short
    # This function internally initializes the model (using Config params) and runs training
    # Note: It will re-instantiate the model using Config.NUM_FEATURES/BLOCKS.
    # We'll let it use the defaults from Config (64/20) which is fine for 1 epoch on a few patches.

    run_curriculum_training()

    # Verify Checkpoint
    assert os.path.exists(Config.CHECKPOINT_PATH), "Checkpoint file was not created."
    print(f"Training complete. Checkpoint saved at {Config.CHECKPOINT_PATH}")

    # -------------------------------------------------------------------------
    # 5. Inference Verification
    # -------------------------------------------------------------------------
    print("\n[5] Running Inference Demo...")

    # Create a mini test metadata file to avoid processing the whole test set
    full_test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    mini_test_df = full_test_df.head(3)  # Take top 3 images
    mini_test_path = os.path.join(demo_dir, "mini_test.csv")
    mini_test_df.to_csv(mini_test_path, index=False)

    # Point Config to mini test set
    Config.TEST_METADATA_PATH = mini_test_path

    # Run inference
    generate_submission()

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Check content format
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(submission_df.columns) == [
        "id",
        "value",
    ], "Incorrect submission columns"
    assert len(submission_df) > 0, "Submission file is empty"

    # Check value range
    min_val = submission_df["value"].min()
    max_val = submission_df["value"].max()
    assert (
        min_val >= 0 and max_val <= 1
    ), f"Values out of range [0, 1]: {min_val}, {max_val}"

    print(f"Inference complete. Submission generated with {len(submission_df)} pixels.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
