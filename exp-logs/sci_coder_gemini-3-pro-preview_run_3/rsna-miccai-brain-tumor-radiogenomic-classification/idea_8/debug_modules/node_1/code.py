import os
import shutil
import pandas as pd
import numpy as np
import torch
import sys

# Import from the provided library
from library import config, data, model, trainer


def run_demo():
    print("Initializing Demonstration...")

    # ==========================================
    # 1. Setup & Configuration Override
    # ==========================================
    # We override config parameters to create a lightweight execution environment
    # suitable for a quick demonstration.

    DEMO_DIR = "./working/demo_execution"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Update Config Paths
    config.WORKING_DIR = DEMO_DIR
    config.CACHE_DIR = DEMO_DIR
    config.TRAIN_CACHE_X = os.path.join(DEMO_DIR, "cached_train_X.npy")
    config.TRAIN_CACHE_Y = os.path.join(DEMO_DIR, "cached_train_y.npy")
    config.VAL_CACHE_X = os.path.join(DEMO_DIR, "cached_val_X.npy")
    config.VAL_CACHE_Y = os.path.join(DEMO_DIR, "cached_val_y.npy")
    config.TEST_CACHE_X = os.path.join(DEMO_DIR, "cached_test_X.npy")
    config.TEST_CACHE_IDS = os.path.join(DEMO_DIR, "cached_test_ids.npy")
    config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    config.SUBMISSION_PATH = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    # Clear stale cache files to prevent dimension mismatch
    # The demo uses different dimensions (16 channels) than the cached full run (128 channels).
    for cache_path in [
        config.TRAIN_CACHE_X,
        config.TRAIN_CACHE_Y,
        config.VAL_CACHE_X,
        config.VAL_CACHE_Y,
        config.TEST_CACHE_X,
        config.TEST_CACHE_IDS,
    ]:
        if os.path.exists(cache_path):
            os.remove(cache_path)
            print(f"Removed stale cache: {cache_path}")

    # Update Hyperparameters for Speed
    config.IMG_SIZE = 128  # Reduced from 256
    config.NUM_SLICES = 4  # Reduced from 32
    config.NUM_MODALITIES = 4  # Fixed (FLAIR, T1w, T1wCE, T2w)
    config.INPUT_CHANNELS = config.NUM_SLICES * config.NUM_MODALITIES  # 16
    config.BATCH_SIZE = 4
    config.EPOCHS = 2
    config.NUM_WORKERS = 2  # Reduced workers

    print(
        f"Configuration updated: Image Size={config.IMG_SIZE}, Slices={config.NUM_SLICES}, Batch={config.BATCH_SIZE}"
    )

    # ==========================================
    # 2. Create Mini-Metadata (Data Subsetting)
    # ==========================================
    # Processing the full dataset takes time. We create a subset of the metadata.
    print("\nCreating mini-datasets for demonstration...")

    meta_demo_dir = os.path.join(DEMO_DIR, "metadata")
    os.makedirs(meta_demo_dir, exist_ok=True)

    # Load original metadata
    train_df = pd.read_parquet(config.TRAIN_META_PATH)
    val_df = pd.read_parquet(config.VAL_META_PATH)
    test_df = pd.read_parquet(config.TEST_META_PATH)

    # Sample subset (ensure we have enough for a batch)
    mini_train = train_df.head(8).copy()
    mini_val = val_df.head(4).copy()
    mini_test = test_df.head(4).copy()

    # Save mini metadata
    mini_train_path = os.path.join(meta_demo_dir, "train.parquet")
    mini_val_path = os.path.join(meta_demo_dir, "val.parquet")
    mini_test_path = os.path.join(meta_demo_dir, "test.parquet")

    mini_train.to_parquet(mini_train_path)
    mini_val.to_parquet(mini_val_path)
    mini_test.to_parquet(mini_test_path)

    # Point config to mini metadata
    config.TRAIN_META_PATH = mini_train_path
    config.VAL_META_PATH = mini_val_path
    config.TEST_META_PATH = mini_test_path

    print(
        f"Mini-metadata created with {len(mini_train)} train, {len(mini_val)} val, {len(mini_test)} test samples."
    )

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\nVerifying Model Architecture...")

    # Set seed for reproducibility
    model.set_seed(config.SEED)

    # Instantiate model
    net = model.Stabilized25DNet()

    # Create dummy input: (Batch, Channels, Height, Width)
    # Channels = NUM_SLICES * NUM_MODALITIES = 4 * 4 = 16
    dummy_input = torch.randn(
        2, config.INPUT_CHANNELS, config.IMG_SIZE, config.IMG_SIZE
    )

    # Forward pass
    output = net(dummy_input)

    # Check output shape: (Batch, 1)
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"
    print("Model forward pass successful. Output shape verified.")

    # ==========================================
    # 4. Training Demonstration
    # ==========================================
    print("\nStarting Training Demonstration...")

    # Initialize Trainer
    # Note: Trainer uses config.DEVICE internally
    my_trainer = trainer.Trainer(learning_rate=1e-3)

    # Run training
    # This will implicitly call data.get_dataloaders, which will process the mini-dataset
    # and cache it to DEMO_DIR.
    my_trainer.train(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)

    # Verify model artifact creation
    assert os.path.exists(config.MODEL_SAVE_PATH), "Best model file was not saved!"
    print(f"Training finished. Model saved to {config.MODEL_SAVE_PATH}")

    # ==========================================
    # 5. Inference Demonstration
    # ==========================================
    print("\nStarting Inference Demonstration...")

    # Run prediction
    my_trainer.predict(batch_size=config.BATCH_SIZE)

    # Verify submission file
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not generated!"

    # Check submission content
    sub_df = pd.read_csv(config.SUBMISSION_PATH)
    print("Submission Head:")
    print(sub_df.head())

    # Validate structure
    assert "BraTS21ID" in sub_df.columns, "Submission missing BraTS21ID column"
    assert "MGMT_value" in sub_df.columns, "Submission missing MGMT_value column"
    assert len(sub_df) == len(
        mini_test
    ), f"Expected {len(mini_test)} predictions, got {len(sub_df)}"
    assert sub_df["MGMT_value"].dtype == float, "MGMT_value should be float"

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    run_demo()
