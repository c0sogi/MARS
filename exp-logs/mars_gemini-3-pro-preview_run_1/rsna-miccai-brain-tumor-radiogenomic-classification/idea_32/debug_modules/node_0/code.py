import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import (
    TRAIN_METADATA_PATH,
    WORKING_DIR,
    IMG_SIZE,
    INPUT_CHANNELS,
    SEED,
    DEVICE,
)
from library.data_processing import process_dataset
from library.dataset import BraTSDataset, get_transforms
from library.model import RNVSNetwork
from library.trainer import run_training
from library.utils import seed_everything


def demo_data_processing():
    print("\n=== Demo: Data Processing ===")

    # Load metadata
    if not os.path.exists(TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {TRAIN_METADATA_PATH}")

    df = pd.read_csv(TRAIN_METADATA_PATH)

    # Use a tiny subset for speed (5 subjects)
    subset_df = df.head(5).copy()
    print(f"Processing subset of {len(subset_df)} subjects...")

    # Define temporary cache paths for this demo
    demo_cache_dir = os.path.join(WORKING_DIR, "demo_cache")
    os.makedirs(demo_cache_dir, exist_ok=True)

    cache_ids = os.path.join(demo_cache_dir, "ids.npy")
    cache_imgs = os.path.join(demo_cache_dir, "imgs.npy")
    cache_lbls = os.path.join(demo_cache_dir, "lbls.npy")

    # Clean up previous demo runs
    if os.path.exists(cache_imgs):
        os.remove(cache_imgs)

    # Run processing
    ids, imgs, lbls = process_dataset(
        subset_df,
        cache_ids,
        cache_imgs,
        cache_lbls,
        load_cached_data=False,  # Force processing
        debug=False,
    )

    # Verification
    print(f"Processed IDs shape: {ids.shape}")
    print(f"Processed Images shape: {imgs.shape}")
    print(f"Processed Labels shape: {lbls.shape}")

    # Assertions
    assert len(ids) == 5, "Number of IDs should be 5"
    assert imgs.shape == (
        5,
        IMG_SIZE,
        IMG_SIZE,
        INPUT_CHANNELS,
    ), f"Image shape mismatch. Expected (5, {IMG_SIZE}, {IMG_SIZE}, {INPUT_CHANNELS}), got {imgs.shape}"
    assert lbls.shape == (5,), "Labels shape should be (5,)"

    print("Data processing verification passed.")
    return imgs, lbls, ids


def demo_dataset_logic(imgs, lbls, ids):
    print("\n=== Demo: Dataset & Structured Dropout ===")

    # 1. Standard Dataset
    dataset = BraTSDataset(
        images=imgs,
        labels=lbls,
        ids=ids,
        transform=get_transforms("train"),
        input_dropout_prob=0.0,
    )

    img_tensor, label_tensor = dataset[0]
    print(f"Item 0 Image Tensor Shape: {img_tensor.shape}")
    print(f"Item 0 Label Tensor Shape: {label_tensor.shape}")

    assert img_tensor.shape == (
        INPUT_CHANNELS,
        IMG_SIZE,
        IMG_SIZE,
    ), "Tensor channel-first format incorrect"
    assert label_tensor.shape == (1,), "Label tensor should be shape (1,)"

    # 2. Test Structured Input Dropout Logic
    # We force prob=1.0 so dropout ALWAYS happens
    dropout_dataset = BraTSDataset(
        images=imgs,
        labels=lbls,
        ids=ids,
        transform=get_transforms("train"),
        input_dropout_prob=1.0,
    )

    print("Testing Structured Input Dropout (Prob=1.0)...")
    # Fetch an item
    dropped_img, _ = dropout_dataset[0]

    # Logic: Either Center (channels 3-5) is 0 OR Periphery (0-2, 6-8) is 0
    center_sum = torch.sum(torch.abs(dropped_img[3:6, :, :]))
    periph_sum = torch.sum(torch.abs(dropped_img[0:3, :, :])) + torch.sum(
        torch.abs(dropped_img[6:9, :, :])
    )

    is_center_dropped = center_sum == 0
    is_periph_dropped = periph_sum == 0

    print(f"Center Sum: {center_sum:.4f}, Periphery Sum: {periph_sum:.4f}")

    if is_center_dropped:
        print("-> Center triplet was dropped.")
    elif is_periph_dropped:
        print("-> Peripheral triplets were dropped.")
    else:
        # Note: It's theoretically possible for the original image to be all zeros,
        # but unlikely given we selected valid subjects.
        # If the image was empty, both sums would be 0.
        if torch.sum(torch.abs(dropped_img)) == 0:
            print("-> Image was empty (all zeros).")
        else:
            raise AssertionError(
                "Structured Dropout failed: Neither center nor periphery was zeroed out."
            )

    print("Dataset logic verification passed.")


def demo_model_architecture():
    print("\n=== Demo: Model Architecture ===")

    model = RNVSNetwork(input_dropout_prob=0.0)
    model.eval()

    # Check first layer modification
    first_conv = model.backbone.conv_stem
    print(f"First Conv In-Channels: {first_conv.in_channels}")

    assert (
        first_conv.in_channels == INPUT_CHANNELS
    ), f"Model first layer should accept {INPUT_CHANNELS} channels, got {first_conv.in_channels}"

    # Forward pass check
    dummy_input = torch.randn(2, INPUT_CHANNELS, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Forward pass output shape: {output.shape}")
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"

    print("Model architecture verification passed.")


def demo_full_pipeline():
    print("\n=== Demo: Full Training Pipeline (Debug Mode) ===")

    # run_training with debug=True runs on a small subset (50 samples) for limited epochs
    # We limit to 1 epoch for this demo to be fast.
    try:
        best_auc = run_training(debug=True, epochs=1)
        print(f"Training finished. Best AUC: {best_auc}")

        # Check if submission file was generated
        submission_path = "./submission/submission.csv"
        if os.path.exists(submission_path):
            print(f"Submission file found at {submission_path}")
            df_sub = pd.read_csv(submission_path)
            print("Submission head:")
            print(df_sub.head())
            assert not df_sub.empty, "Submission file is empty"
            assert (
                "BraTS21ID" in df_sub.columns and "MGMT_value" in df_sub.columns
            ), "Submission columns incorrect"
        else:
            raise FileNotFoundError("Submission file was not generated.")

    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        raise e


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(SEED)

    # 1. Verify Data Processing
    imgs, lbls, ids = demo_data_processing()

    # 2. Verify Dataset & Transforms
    demo_dataset_logic(imgs, lbls, ids)

    # 3. Verify Model
    demo_model_architecture()

    # 4. Verify Full Pipeline Integration
    demo_full_pipeline()

    print("\nAll demonstrations completed successfully.")
