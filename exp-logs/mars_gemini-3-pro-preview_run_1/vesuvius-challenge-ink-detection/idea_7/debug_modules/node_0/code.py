import os
import sys
import shutil
import warnings
import numpy as np
import pandas as pd
import torch

# Suppress warnings
warnings.filterwarnings("ignore")

# Import library modules
from library import config, utils, dataset, model, train, inference


def setup_demo_environment():
    """
    Sets up a temporary environment for the demo to run quickly.
    Creates mini-datasets and overrides config paths.
    """
    print("--- Setting up Demo Environment ---")

    # Define demo directories
    demo_dir = os.path.join("./working", "demo_execution")
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Create subdirectories as expected by the library
    os.makedirs(os.path.join(demo_dir, "cache"), exist_ok=True)
    os.makedirs(os.path.join(demo_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(demo_dir, "predictions"), exist_ok=True)

    # Create Mini Metadata (Top 5 samples) to speed up execution
    # We read the existing metadata and save a subset to the demo folder
    meta_files = {
        "train": config.TRAIN_METADATA_PATH,
        "val": config.VAL_METADATA_PATH,
        "test": config.TEST_METADATA_PATH,
    }

    new_paths = {}

    for key, path in meta_files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Take top 5 or less
            df_mini = df.head(5).copy()
            new_path = os.path.join(demo_dir, f"{key}_mini.csv")
            df_mini.to_csv(new_path, index=False)
            new_paths[key] = new_path
            print(f"Created mini {key} metadata with {len(df_mini)} samples.")
        else:
            # Fallback for test if not present (though it should be)
            new_paths[key] = path

    # --- Monkey Patch Config ---
    # We modify the config module variables at runtime to point to our demo environment
    config.WORKING_DIR = demo_dir
    config.CACHE_DIR = os.path.join(demo_dir, "cache")
    config.CHECKPOINT_DIR = os.path.join(demo_dir, "checkpoints")
    config.PREDICTIONS_DIR = os.path.join(demo_dir, "predictions")
    config.SUBMISSION_PATH = os.path.join(demo_dir, "submission_demo.csv")

    config.TRAIN_METADATA_PATH = new_paths["train"]
    config.VAL_METADATA_PATH = new_paths["val"]
    config.TEST_METADATA_PATH = new_paths["test"]

    # Reduce training parameters for speed
    config.NUM_EPOCHS = 1
    config.BATCH_SIZE = 2  # Small batch for mini dataset

    print("Config updated for demo execution.")
    print("-" * 30)


def verify_utils():
    """
    Verifies utility functions.
    """
    print("--- Verifying Utils ---")

    # 1. Test RLE Encode
    # Pattern: 0 1 1 1 0 1
    # Indices: 1 2 3 4 5 6 (1-based)
    # Ink at: 2, 3, 4 and 6
    # Run 1: Start 2, Length 3
    # Run 2: Start 6, Length 1
    # Expected RLE: "2 3 6 1"
    mask = np.array([[0, 1, 1], [1, 0, 1]], dtype=np.uint8)  # Flattened: 0 1 1 1 0 1
    rle = utils.rle_encode(mask)
    assert rle == "2 3 6 1", f"RLE Encoding failed. Expected '2 3 6 1', got '{rle}'"
    print("RLE Encode logic verified.")

    # 2. Test F-Beta Score (Beta=0.5)
    # F0.5 weights precision higher than recall.
    y_true = np.array([1, 0, 1, 1])
    y_pred = np.array([1, 1, 0, 1])
    # TP = 2 (Indices 0, 3)
    # FP = 1 (Index 1)
    # FN = 1 (Index 2)
    # Beta = 0.5, Beta^2 = 0.25
    # Numerator = (1.25) * 2 = 2.5
    # Denominator = (1.25 * 2) + (0.25 * 1) + 1 = 2.5 + 0.25 + 1 = 3.75
    # Score = 2.5 / 3.75 = 0.6666...
    score = utils.calculate_fbeta(y_true, y_pred, beta=0.5)
    assert abs(score - (2.5 / 3.75)) < 1e-5, f"F-Beta calculation failed. Got {score}"
    print("F-Beta calculation verified.")
    print("-" * 30)


def verify_dataset_and_model():
    """
    Verifies Dataset loading and Model forward pass.
    """
    print("--- Verifying Dataset and Model ---")

    # 1. Dataset
    # Initialize dataset with mini metadata
    ds = dataset.InkDataset(
        config.TRAIN_METADATA_PATH, mode="train", load_cached_data=True
    )

    if len(ds) == 0:
        print("Warning: Mini dataset is empty. Skipping dataset verification.")
        return

    # Fetch one sample
    vol, label = ds[0]

    # Verify shapes
    # Volume: (65, 512, 512)
    # Label: (1, 512, 512)
    assert vol.shape == (
        config.Z_DIM,
        config.PATCH_SIZE,
        config.PATCH_SIZE,
    ), f"Volume shape mismatch. Got {vol.shape}"
    assert label.shape == (
        1,
        config.PATCH_SIZE,
        config.PATCH_SIZE,
    ), f"Label shape mismatch. Got {label.shape}"

    print(
        f"Dataset loaded successfully. Volume shape: {vol.shape}, Label shape: {label.shape}"
    )

    # 2. Model
    net = model.HDNet().to(config.DEVICE)

    # Create a dummy batch (Batch Size 2)
    dummy_input = torch.stack([vol, vol]).to(config.DEVICE)  # (2, 65, 512, 512)

    # Forward pass
    with torch.no_grad():
        output = net(dummy_input)

    # Verify output shape: (Batch, 1, 512, 512)
    expected_shape = (2, 1, config.PATCH_SIZE, config.PATCH_SIZE)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print("Model forward pass verified.")
    print("-" * 30)


def run_pipeline_demo():
    """
    Runs the Training and Inference pipeline.
    """
    print("--- Running Full Pipeline Demo ---")

    # 1. Training
    # We run for 1 epoch on the mini dataset.
    print("Step 1: Training...")
    try:
        train.run_training(num_epochs=config.NUM_EPOCHS, load_cached_data=True)
    except Exception as e:
        raise RuntimeError(f"Training failed: {e}")

    # Verify checkpoint creation
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(
        checkpoint_path
    ), "Checkpoint file 'best_model.pth' was not created."
    print("Training completed and checkpoint saved.")

    # 2. Inference
    # Generates submission using the trained model
    print("Step 2: Inference...")
    try:
        inference.generate_submission(load_cached_data=True)
    except Exception as e:
        raise RuntimeError(f"Inference failed: {e}")

    # Verify submission file
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not created."

    # Check submission content
    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    required_cols = ["Id", "Predicted"]
    assert all(
        col in df_sub.columns for col in required_cols
    ), "Submission file missing required columns."

    print(f"Inference completed. Submission saved to {config.SUBMISSION_PATH}")
    print("Preview of submission:")
    print(df_sub.head())
    print("-" * 30)


def main():
    # Ensure reproducibility
    utils.set_seed(42)

    # Setup
    setup_demo_environment()

    # Verification
    verify_utils()
    verify_dataset_and_model()

    # Execution
    run_pipeline_demo()

    print("\nSUCCESS: All demonstrations completed without error.")


if __name__ == "__main__":
    main()
