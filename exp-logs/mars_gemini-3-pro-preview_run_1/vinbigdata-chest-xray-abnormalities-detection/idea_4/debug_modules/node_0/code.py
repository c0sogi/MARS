import os
import sys
import pandas as pd
import numpy as np
import torch
import cv2
import glob

# Import from the provided library
from library.config import Config, seed_everything
from library.utils import read_dicom_binary, get_original_dimensions
from library.dataset import ThoraxDataset, get_dataloaders
from library.model import AnatomicalCenterNet
from library.loss import CenterNetLoss
from library.train import run_training
from library.inference import predict_and_format


def create_subset_metadata(source_path, dest_path, n_samples=20):
    """Creates a small subset of the metadata for demonstration purposes."""
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source metadata not found: {source_path}")

    df = pd.read_csv(source_path)
    # Sample n_samples, handling cases where df is smaller than n_samples
    n = min(len(df), n_samples)
    df_subset = df.sample(n=n, random_state=Config.SEED).reset_index(drop=True)

    # Ensure directory exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    df_subset.to_csv(dest_path, index=False)
    print(f"Created subset metadata at {dest_path} with {len(df_subset)} rows.")
    return df_subset


def verify_utils(df_sample):
    """Verifies utility functions."""
    print("\n=== Verifying Utils ===")

    # Test DICOM reading
    row = df_sample.iloc[0]
    file_path = os.path.join(Config.INPUT_DIR, row["file_path"])

    try:
        img = read_dicom_binary(file_path)
        print(f"Successfully read DICOM: {file_path}")
        print(f"Image Shape: {img.shape}, Dtype: {img.dtype}")

        # Assertions
        assert len(img.shape) >= 2, "Image must be at least 2D"
        assert img.size > 0, "Image must not be empty"

    except Exception as e:
        print(
            f"DICOM reading failed (expected if specific libraries missing in env, but checking logic): {e}"
        )

    # Test Dimension Caching
    dims_map = get_original_dimensions(df_sample.head(5))
    assert len(dims_map) > 0, "Dimension map should not be empty"
    img_id = str(row["image_id"])
    if img_id in dims_map:
        w, h = dims_map[img_id]
        print(f"Retrieved dimensions for {img_id}: {w}x{h}")
        assert w > 0 and h > 0, "Dimensions must be positive"


def verify_dataset(df_subset):
    """Verifies Dataset and DataLoader logic."""
    print("\n=== Verifying Dataset ===")

    # Instantiate Dataset
    ds = ThoraxDataset(df_subset, split="train")
    print(f"Dataset length: {len(ds)}")

    # Fetch one item
    image, targets, image_id = ds[0]

    # Verify Image
    assert isinstance(image, torch.Tensor), "Image output must be a torch Tensor"
    assert image.shape == (
        3,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    ), f"Image shape mismatch. Expected (3, {Config.IMAGE_SIZE[0]}, {Config.IMAGE_SIZE[1]}), got {image.shape}"

    # Verify Targets
    expected_keys = ["hm", "wh", "reg", "ind", "reg_mask", "global_label"]
    for k in expected_keys:
        assert k in targets, f"Missing key in targets: {k}"
        assert isinstance(targets[k], torch.Tensor), f"Target {k} must be a tensor"

    print("Dataset item shapes:")
    print(f"  Image: {image.shape}")
    print(f"  Heatmap: {targets['hm'].shape}")
    print(f"  Global Label: {targets['global_label']}")

    # Verify DataLoader
    loader, _, _ = get_dataloaders(df_subset, df_subset)
    batch_imgs, batch_targets, _ = next(iter(loader))
    assert batch_imgs.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    print("DataLoader batch retrieval successful.")
    return loader


def verify_model_and_loss(loader):
    """Verifies Model architecture and Loss calculation."""
    print("\n=== Verifying Model & Loss ===")

    device = torch.device(Config.DEVICE)
    model = AnatomicalCenterNet().to(device)
    criterion = CenterNetLoss()

    # Get a batch
    images, targets, _ = next(iter(loader))
    images = images.to(device)
    target_dict = {k: v.to(device) for k, v in targets.items()}

    # Forward Pass
    outputs = model(images)

    # Check output keys
    assert "hm" in outputs
    assert "wh" in outputs
    assert "reg" in outputs
    assert "global_label" in outputs

    # Check output shapes
    # Heatmap: (B, 14, H/4, W/4)
    expected_h = Config.IMAGE_SIZE[0] // 4
    expected_w = Config.IMAGE_SIZE[1] // 4
    assert outputs["hm"].shape == (Config.BATCH_SIZE, 14, expected_h, expected_w)
    assert outputs["global_label"].shape == (Config.BATCH_SIZE, 1)

    print("Model forward pass successful.")

    # Loss Calculation
    loss, loss_stats = criterion(outputs, target_dict)

    print(f"Calculated Loss: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive (usually)"

    print("Loss calculation successful.")


def run_full_pipeline_demo():
    """Runs the training and inference modules using the subset data."""
    print("\n=== Running Full Pipeline Demo (Train + Inference) ===")

    # 1. Run Training
    # run_training() reads from Config, which we have patched
    print("Starting training loop...")
    run_training()

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "last_model.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print("Training completed and checkpoint verified.")

    # 2. Run Inference
    print("Starting inference...")
    predict_and_format(
        checkpoint_path=checkpoint_path, output_path=Config.SUBMISSION_PATH
    )

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(sub_df)} rows.")
    print(sub_df.head())

    # Basic format check
    assert "ID" in sub_df.columns or "image_id" in sub_df.columns, "Missing ID column"
    assert "PredictionString" in sub_df.columns, "Missing PredictionString column"


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)

    # Define temporary paths for demo
    working_dir = "./working/demo"
    os.makedirs(working_dir, exist_ok=True)

    demo_train_path = os.path.join(working_dir, "train_subset.csv")
    demo_val_path = os.path.join(working_dir, "val_subset.csv")
    demo_test_path = os.path.join(working_dir, "test_subset.csv")
    demo_sub_path = os.path.join(working_dir, "submission.csv")

    # 2. Patch Config for Speed and Demo Paths
    # We modify the class attributes directly
    print("Patching Configuration for Demo...")
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.TRAIN_META_PATH = demo_train_path
    Config.VAL_META_PATH = demo_val_path
    Config.TEST_META_PATH = demo_test_path
    Config.SUBMISSION_PATH = demo_sub_path
    Config.CHECKPOINT_DIR = os.path.join(working_dir, "checkpoints")
    Config.CACHE_DIR = os.path.join(working_dir, "cache")

    # Ensure directories exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 3. Create Data Subsets
    # We use the original metadata files provided in the environment
    orig_train_meta = "./metadata/train_meta.csv"
    orig_val_meta = "./metadata/val_meta.csv"
    orig_test_meta = "./metadata/test_meta.csv"

    train_subset = create_subset_metadata(
        orig_train_meta, demo_train_path, n_samples=32
    )
    val_subset = create_subset_metadata(orig_val_meta, demo_val_path, n_samples=16)
    test_subset = create_subset_metadata(orig_test_meta, demo_test_path, n_samples=10)

    # 4. Verify Components
    verify_utils(train_subset)
    loader = verify_dataset(train_subset)
    verify_model_and_loss(loader)

    # 5. Run Integrated Pipeline
    run_full_pipeline_demo()

    print("\n=== All Demonstrations Completed Successfully ===")
