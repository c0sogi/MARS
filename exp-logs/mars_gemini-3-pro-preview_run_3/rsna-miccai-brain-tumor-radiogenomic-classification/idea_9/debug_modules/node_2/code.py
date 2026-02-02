import os
import shutil
import pandas as pd
import torch
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import set_seed
from library.model import Stabilized25DNet
from library.train import run_training
from library.predict import generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Sets up a lightweight environment for the demo by creating a subset of metadata
    and configuring the Config class to use temporary directories and fewer epochs.
    """
    print("Setting up demo environment...")

    # Define paths
    base_metadata_dir = "./metadata"
    demo_metadata_dir = "./working/demo_metadata"
    demo_working_dir = "./working/demo_run"

    os.makedirs(demo_metadata_dir, exist_ok=True)
    os.makedirs(demo_working_dir, exist_ok=True)

    # 1. Create Subset Metadata
    # We take the first 6 samples for train, 4 for val, and 4 for test
    # to ensure the code runs quickly but exercises the data loading logic.
    splits = {"train": 6, "val": 4, "test": 4}

    for split, n_samples in splits.items():
        src_path = os.path.join(base_metadata_dir, f"{split}.parquet")
        dst_path = os.path.join(demo_metadata_dir, f"{split}.parquet")

        if os.path.exists(src_path):
            df = pd.read_parquet(src_path)
            # Take a small subset
            df_subset = df.head(n_samples).copy()
            df_subset.to_parquet(dst_path, index=False)
            print(f"Created {split} subset with {len(df_subset)} samples.")
        else:
            raise FileNotFoundError(f"Original metadata not found at {src_path}")

    # 2. Override Config parameters for the demo
    print("Overriding Config parameters...")
    Config.METADATA_DIR = demo_metadata_dir
    Config.WORKING_DIR = demo_working_dir
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size for the demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure the working directory exists (Config creates the original one on import)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Clean up stale artifacts to prevent ghost bugs
    # Cite debug_lesson_8: Invalidate Stale Output Artifacts to Prevent "Ghost" Bugs
    stale_model = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(stale_model):
        os.remove(stale_model)


def verify_model_architecture():
    """
    Instantiates the model and performs a forward pass with dummy data
    to verify input/output dimensions.
    """
    print("\nVerifying model architecture...")

    set_seed(Config.SEED)
    device = torch.device("cpu")  # Use CPU for simple verification

    model = Stabilized25DNet().to(device)
    model.eval()

    # Input shape: (Batch, Channels, H, W)
    # Channels = 32 slices * 4 modalities = 128
    batch_size = 2
    dummy_input = torch.randn(
        batch_size, Config.IN_CHANNELS, Config.IMAGE_SIZE, Config.IMAGE_SIZE
    ).to(device)

    try:
        with torch.no_grad():
            output = model(dummy_input)

        # Check output shape
        expected_shape = (batch_size, 1)
        assert (
            output.shape == expected_shape
        ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

        print("Model architecture verification passed.")

    except Exception as e:
        raise AssertionError(f"Model verification failed: {e}")


def run_demo_training():
    """
    Runs the training pipeline using the subset data.
    """
    print("\nRunning demo training...")

    # Run training (this will load data, cache it, and train for 1 epoch)
    # We set patience=1 to ensure it doesn't run longer than necessary even if we increased epochs
    run_training(num_epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE, patience=1)

    # Verify checkpoint creation
    expected_checkpoint = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(expected_checkpoint):
        raise AssertionError(
            f"Training failed to produce checkpoint at {expected_checkpoint}"
        )

    print(f"Training demo complete. Checkpoint saved to {expected_checkpoint}")


def run_demo_inference():
    """
    Runs the inference pipeline using the trained model and subset test data.
    """
    print("\nRunning demo inference...")

    # Generate submission
    # We force load_cached_data=False for test to demonstrate the processing pipeline
    # (though in a real run, caching is preferred)
    generate_submission(load_cached_data=False, batch_size=Config.BATCH_SIZE)

    # Verify submission file
    submission_path = "./submission/submission.csv"
    if not os.path.exists(submission_path):
        raise AssertionError("Inference failed to produce submission.csv")

    # Verify submission content
    df = pd.read_csv(submission_path)
    required_cols = ["BraTS21ID", "MGMT_value"]

    if not all(col in df.columns for col in required_cols):
        raise AssertionError(
            f"Submission file missing required columns. Found: {df.columns}"
        )

    if len(df) == 0:
        raise AssertionError("Submission file is empty.")

    # Check if probabilities are valid
    if df["MGMT_value"].min() < 0 or df["MGMT_value"].max() > 1:
        raise AssertionError("Predicted probabilities are out of range [0, 1].")

    print("Inference demo complete. Submission generated successfully.")
    print(f"Sample predictions:\n{df.head()}")


if __name__ == "__main__":
    # 1. Setup
    setup_demo_environment()

    # 2. Verify Model Logic
    verify_model_architecture()

    # 3. Run Training Demo
    # This tests dataset loading, preprocessing, caching, and the training loop
    run_demo_training()

    # 4. Run Inference Demo
    # This tests the prediction pipeline and submission generation
    run_demo_inference()

    print("\nAll demo steps completed successfully.")
