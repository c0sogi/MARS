import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil

# --- 1. Configuration Override ---
# We modify the Config class attributes BEFORE importing other library modules
# to ensure they pick up the changes.
from library.config import Config

# Define a demo working directory
DEMO_DIR = "./working/demo_execution"
if os.path.exists(DEMO_DIR):
    shutil.rmtree(DEMO_DIR)
os.makedirs(DEMO_DIR, exist_ok=True)

# Override Paths
Config.WORKING_DIR = DEMO_DIR
Config.SUBMISSION_DIR = DEMO_DIR
Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "demo_submission.csv")

# Create Mini Metadata files to speed up data loading and inference
# We read the original metadata, take the head, and save to demo dir
original_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
original_val = pd.read_csv(Config.VAL_METADATA_PATH)
original_test = pd.read_csv(Config.TEST_METADATA_PATH)

# Use only 2 samples for train, 1 for val, 1 for test
mini_train_path = os.path.join(DEMO_DIR, "mini_train.csv")
mini_val_path = os.path.join(DEMO_DIR, "mini_val.csv")
mini_test_path = os.path.join(DEMO_DIR, "mini_test.csv")

original_train.head(2).to_csv(mini_train_path, index=False)
original_val.head(1).to_csv(mini_val_path, index=False)
original_test.head(1).to_csv(mini_test_path, index=False)

# Update Config to point to mini metadata
Config.TRAIN_METADATA_PATH = mini_train_path
Config.VAL_METADATA_PATH = mini_val_path
Config.TEST_METADATA_PATH = mini_test_path

# Override Hyperparameters for Speed
Config.EPOCHS = 1
Config.NUM_ENSEMBLE_MODELS = 1
Config.BATCH_SIZE = 4
Config.PATCH_SIZE = 32  # Smaller patches
Config.STRIDE = 200  # Large stride -> very few patches
Config.MODEL_DEPTH = 2  # Shallow model
Config.MODEL_FILTERS = 16  # Narrow model
Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data
Config.AUGMENTATION = False  # Disable aug for speed

print("Configuration overridden for rapid demonstration.")

# --- 2. Import Library Modules ---
# Now we import the rest, which will use the modified Config
from library.utils import seed_everything
from library.data_loader import get_dataloaders
from library.architecture import ResDnCNN
from library.trainer import ModelTrainer
from library.inference import generate_submission


def run_demo():
    # Set seed for reproducibility
    seed_everything(42)

    # --- Step 1: Data Loading ---
    print("\n[Step 1] Initializing DataLoaders...")
    # load_cached_data=False forces extraction from our new mini metadata
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Verification
    assert len(train_loader) > 0, "Train loader is empty!"

    # Check shape of a single batch
    # Note: With high stride, we might get very few patches.
    # If len(train_loader) is small, this iteration is safe.
    noisy_imgs, residual_targets = next(iter(train_loader))
    print(f"Batch Shape: {noisy_imgs.shape}")

    # Expected: [Batch, Channels, Patch, Patch]
    # Batch size might be smaller than Config.BATCH_SIZE if it's the last batch
    assert noisy_imgs.shape[1:] == (1, Config.PATCH_SIZE, Config.PATCH_SIZE)
    assert residual_targets.shape[1:] == (1, Config.PATCH_SIZE, Config.PATCH_SIZE)
    print("Data shapes verified.")

    # --- Step 2: Model Instantiation ---
    print("\n[Step 2] Initializing Model...")
    model = ResDnCNN(
        depth=Config.MODEL_DEPTH,
        filters=Config.MODEL_FILTERS,
        input_channels=Config.INPUT_CHANNELS,
        output_channels=Config.OUTPUT_CHANNELS,
    )

    # Verification: Forward pass
    dummy_input = torch.randn(2, 1, Config.PATCH_SIZE, Config.PATCH_SIZE)
    dummy_out = model(dummy_input)
    assert (
        dummy_out.shape == dummy_input.shape
    ), f"Output shape mismatch: {dummy_out.shape}"
    print("Model forward pass verified.")

    # --- Step 3: Training ---
    print("\n[Step 3] Starting Training (1 Epoch)...")
    trainer = ModelTrainer(model)

    # Train model_0
    best_loss = trainer.train(train_loader, val_loader, model_id=0)
    print(f"Training complete. Best Loss: {best_loss:.6f}")

    # Verification: Checkpoint existence
    checkpoint_path = os.path.join(Config.WORKING_DIR, "model_0.pth")
    assert os.path.exists(checkpoint_path), "Model checkpoint not found!"
    print(f"Checkpoint verified at {checkpoint_path}")

    # --- Step 4: Inference ---
    print("\n[Step 4] Generating Submission...")
    # This will use the trained model_0 from Config.WORKING_DIR
    # and the mini_test.csv defined in Config.TEST_METADATA_PATH
    generate_submission()

    # Verification: Submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found!"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(df_sub)} rows.")

    # Check columns
    assert "id" in df_sub.columns
    assert "value" in df_sub.columns

    # Check value range (should be roughly 0-1, though regression can go slightly outside)
    # Just checking type here
    assert pd.api.types.is_numeric_dtype(df_sub["value"])

    # Check ID format (e.g., "110_1_1")
    sample_id = df_sub.iloc[0]["id"]
    assert "_" in str(sample_id), f"Invalid ID format: {sample_id}"

    print("Submission format verified.")
    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    run_demo()
