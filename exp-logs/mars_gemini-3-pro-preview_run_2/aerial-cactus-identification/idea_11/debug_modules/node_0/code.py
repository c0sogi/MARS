import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, get_device, save_submission
from library.dataset import CactusDataset, get_transforms
from library.model import NarrowSEResNet
from library.train import train_model
from library.inference import run_inference


def main():
    print("Initializing Demonstration...")

    # 1. Configuration Overrides for Demo
    # We modify Config attributes in-memory to ensure a fast run
    # and to isolate outputs to a demo directory.
    Config.WORK_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = os.path.join(Config.WORK_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Speed up settings
    Config.EPOCHS = 2
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Small subset for speed
    Config.BATCH_SIZE = 10  # Small batch for the small subset
    Config.SEEDS = [42]  # Single seed
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny dataset

    print(f"Working Directory: {Config.WORK_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")

    # 2. Dataset Verification
    print("\n--- Verifying Dataset Logic ---")

    # Instantiate Train Dataset
    train_ds = CactusDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        split="train",
        transform=get_transforms("train"),
        load_cached_data=False,  # Force reload to test logic
        debug=Config.DEBUG,
    )

    # Check Length
    print(f"Train Dataset Length: {len(train_ds)}")
    assert (
        len(train_ds) == Config.DEBUG_SUBSET_SIZE
    ), f"Expected {Config.DEBUG_SUBSET_SIZE} samples in debug mode, got {len(train_ds)}"

    # Check Item Structure
    img, label, img_id = train_ds[0]
    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Label: {label}")

    assert img.shape == (
        3,
        32,
        32,
    ), f"Expected image shape (3, 32, 32), got {img.shape}"
    assert isinstance(label, torch.Tensor), "Label should be a torch.Tensor"
    assert isinstance(img_id, str), "ID should be a string"

    print("Dataset verification passed.")

    # 3. Model Verification
    print("\n--- Verifying Model Architecture ---")
    device = get_device()
    model = NarrowSEResNet().to(device)

    # Create dummy batch
    dummy_input = torch.randn(4, 3, 32, 32).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (4, 1), f"Expected output shape (4, 1), got {output.shape}"

    print("Model verification passed.")

    # 4. Training Pipeline Demonstration
    print("\n--- Running Training Pipeline ---")
    seed = Config.SEEDS[0]

    # Run training (this handles loop, validation, and saving)
    train_model(seed=seed, debug=Config.DEBUG)

    # Verify Checkpoint
    expected_model_path = os.path.join(Config.WORK_DIR, f"model_seed_{seed}.pth")
    if os.path.exists(expected_model_path):
        print(f"Model successfully saved to: {expected_model_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {expected_model_path}")

    # 5. Inference Pipeline Demonstration
    print("\n--- Running Inference Pipeline ---")

    # Run inference (this handles loading model, TTA, and saving submission)
    run_inference(debug=Config.DEBUG)

    # Verify Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file successfully created at: {Config.SUBMISSION_PATH}")

        # Validate content
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission shape: {df_sub.shape}")
        print("First 3 rows:")
        print(df_sub.head(3))

        assert list(df_sub.columns) == [
            "id",
            "has_cactus",
        ], "Submission columns mismatch"
        assert (
            len(df_sub) == Config.DEBUG_SUBSET_SIZE
        ), f"Expected {Config.DEBUG_SUBSET_SIZE} rows in submission (debug mode), got {len(df_sub)}"
    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # 6. Utility Verification
    print("\n--- Verifying Utilities ---")

    # Test manual submission saving
    dummy_ids = ["test_1.jpg", "test_2.jpg"]
    dummy_probs = [0.1, 0.9]
    dummy_sub_path = os.path.join(Config.SUBMISSION_DIR, "dummy_submission.csv")

    save_submission(dummy_ids, dummy_probs, dummy_sub_path)

    assert os.path.exists(dummy_sub_path), "save_submission failed to create file"
    df_dummy = pd.read_csv(dummy_sub_path)
    assert len(df_dummy) == 2, "save_submission wrote incorrect number of rows"
    print("Utility verification passed.")

    print("\n========================================")
    print("SUCCESS: All demonstrations completed.")
    print("========================================")


if __name__ == "__main__":
    main()
