import os
import pandas as pd
import torch
import numpy as np
import shutil
from library import config, utils, data, model, train


def run_demo():
    # ==========================================
    # 1. Setup & Configuration Overrides
    # ==========================================
    print("Setting up configuration for demo...")

    # Create a separate working directory for the demo
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override config parameters for speed and isolation
    config.WORKING_DIR = demo_dir
    config.EPOCHS = 1
    config.BATCH_SIZE = 2
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data
    config.PRETRAINED = False  # Avoid downloading weights
    config.SUBMISSION_PATH = os.path.join(demo_dir, "demo_submission.csv")

    # Update cache paths to point to the demo directory
    # Note: We must explicitly update these because they were defined at import time
    config.CACHE_TRAIN_IMAGES = os.path.join(demo_dir, "cached_train_images.npy")
    config.CACHE_TRAIN_LABELS = os.path.join(demo_dir, "cached_train_labels.npy")
    config.CACHE_VAL_IMAGES = os.path.join(demo_dir, "cached_val_images.npy")
    config.CACHE_VAL_LABELS = os.path.join(demo_dir, "cached_val_labels.npy")
    config.CACHE_TEST_IMAGES = os.path.join(demo_dir, "cached_test_images.npy")
    config.CACHE_TEST_IDS = os.path.join(demo_dir, "cached_test_ids.npy")

    # Set seeds
    utils.seed_everything(42)

    # ==========================================
    # 2. Data Subsetting
    # ==========================================
    print("Creating data subsets...")

    # Load original metadata
    df_train_orig = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_test_orig = pd.read_csv(config.TEST_METADATA_PATH)

    # Create tiny subsets (4 samples each)
    df_train_subset = df_train_orig.head(4).copy()
    df_val_subset = (
        df_train_orig.iloc[4:8].copy()
        if len(df_train_orig) > 8
        else df_train_orig.tail(2).copy()
    )
    df_test_subset = df_test_orig.head(4).copy()

    # Save subsets to temporary files
    subset_train_path = os.path.join(demo_dir, "subset_train.csv")
    subset_val_path = os.path.join(demo_dir, "subset_val.csv")
    subset_test_path = os.path.join(demo_dir, "subset_test.csv")

    df_train_subset.to_csv(subset_train_path, index=False)
    df_val_subset.to_csv(subset_val_path, index=False)
    df_test_subset.to_csv(subset_test_path, index=False)

    # Point config to these new metadata files
    config.TRAIN_METADATA_PATH = subset_train_path
    config.VAL_METADATA_PATH = subset_val_path
    config.TEST_METADATA_PATH = subset_test_path

    print(
        f"Subset metadata created. Train: {len(df_train_subset)}, Val: {len(df_val_subset)}, Test: {len(df_test_subset)}"
    )

    # ==========================================
    # 3. Model Logic Verification
    # ==========================================
    print("Verifying model architecture...")

    device = torch.device("cpu")  # Use CPU for simple shape check
    net = model.SIRVEfficientNet().to(device)

    # Check if stem was modified correctly (9 channels)
    # The first layer in EfficientNet is usually named 'conv_stem'
    first_conv = net.backbone.conv_stem
    assert (
        first_conv.in_channels == 9
    ), f"Expected 9 input channels, got {first_conv.in_channels}"

    # Dummy Forward Pass
    # Input shape: (Batch, Channels, Height, Width)
    dummy_input = torch.randn(2, 9, 224, 224).to(device)
    dummy_output = net(dummy_input)

    assert dummy_output.shape == (
        2,
        1,
    ), f"Expected output shape (2, 1), got {dummy_output.shape}"
    print("Model architecture verified.")

    # ==========================================
    # 4. Training Loop Demonstration
    # ==========================================
    print("Starting training demonstration...")

    # We force load_cached_data=False to trigger the data processing logic on our subset
    best_model_path = train.run_training(load_cached_data=False)

    if not os.path.exists(best_model_path):
        raise FileNotFoundError(
            f"Training failed to produce model file at {best_model_path}"
        )

    print(f"Training demo complete. Model saved to {best_model_path}")

    # ==========================================
    # 5. Inference Demonstration
    # ==========================================
    print("Starting inference demonstration...")

    train.predict_and_submit(best_model_path, load_cached_data=False)

    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Inference failed to produce submission file at {config.SUBMISSION_PATH}"
        )

    # Verify submission content
    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    assert len(df_sub) == len(
        df_test_subset
    ), "Submission file row count does not match test subset"
    assert (
        "BraTS21ID" in df_sub.columns and "MGMT_value" in df_sub.columns
    ), "Submission columns missing"

    print("Inference demo complete.")
    print("All demonstrations passed successfully.")


if __name__ == "__main__":
    run_demo()
