import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    load_dicom,
    load_case_data,
    weighted_loss_metric,
)
from library.dataset import RSNADataset, get_dataloaders, get_transforms
from library.model import CervicalFractureModel
from library.trainer import run_training

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== RSNA Cervical Spine Fracture Detection Pipeline Demo ===\n")

    # ------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Purposes
    # ------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Override Config defaults to run quickly on a small subset
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.SEQ_LEN = 16  # Reduced from 64 to save memory/time
    Config.IMAGE_SIZE = 128  # Reduced from 256 to speed up processing
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in this simple script
    Config.DEBUG_SAMPLE_SIZE = 4  # Tiny subset for debugging
    Config.CACHE_DIR = "./working/demo_cache"  # Separate cache for this run

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print(
        "    Configuration updated: EPOCHS=1, BATCH_SIZE=2, SEQ_LEN=16, IMAGE_SIZE=128"
    )

    # ------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # ------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Load metadata to get valid paths
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    sample_row = train_df.iloc[0]
    sample_study_id = sample_row["StudyInstanceUID"]
    sample_rel_path = sample_row["image_path"]
    sample_full_path = os.path.join(Config.INPUT_DIR, sample_rel_path)

    # Test A: load_dicom
    # Find a single .dcm file in the directory
    dcm_files = [f for f in os.listdir(sample_full_path) if f.endswith(".dcm")]
    if dcm_files:
        dcm_path = os.path.join(sample_full_path, dcm_files[0])
        img = load_dicom(dcm_path, size=Config.IMAGE_SIZE)

        assert isinstance(img, np.ndarray), "load_dicom should return a numpy array"
        assert img.shape == (
            Config.IMAGE_SIZE,
            Config.IMAGE_SIZE,
        ), f"Expected shape ({Config.IMAGE_SIZE}, {Config.IMAGE_SIZE}), got {img.shape}"
        assert img.dtype == np.uint8, "Image dtype should be uint8"
        print(f"    load_dicom passed. Output shape: {img.shape}")
    else:
        print("    Skipping load_dicom check (no .dcm files found in sample dir).")

    # Test B: load_case_data (2.5D Volume Generation)
    # This will create a cache file in ./working/demo_cache
    volume = load_case_data(sample_study_id, sample_full_path, load_cached_data=False)

    assert volume.shape == (
        Config.SEQ_LEN,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
        3,
    ), f"Volume shape mismatch. Expected {(Config.SEQ_LEN, Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3)}, got {volume.shape}"
    print(f"    load_case_data passed. Volume shape: {volume.shape}")

    # Test C: weighted_loss_metric
    # Create dummy predictions and targets (Batch=2, Classes=8)
    y_pred = np.array([[0.1] * 8, [0.9] * 8])
    y_true = np.array([[0] * 8, [1] * 8])
    loss_val = weighted_loss_metric(y_pred, y_true)

    assert isinstance(loss_val, float), "Metric should return a float"
    assert loss_val > 0, "Loss should be positive"
    print(f"    weighted_loss_metric passed. Loss: {loss_val:.4f}")

    # ------------------------------------------------------------------------
    # 3. Verify Dataset and DataLoader
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Dataset and DataLoaders...")

    # Use the get_dataloaders function in debug mode
    # This internally handles caching for the debug subset
    train_loader, val_loader = get_dataloaders(debug=True)

    assert len(train_loader) > 0, "Train loader is empty"

    # Fetch one batch
    batch = next(iter(train_loader))
    images = batch["images"]
    positions = batch["positions"]
    targets = batch["targets"]

    # Check shapes
    # Images: (Batch, Seq_Len, C, H, W) -> (2, 16, 3, 128, 128)
    expected_img_shape = (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    )
    assert (
        images.shape == expected_img_shape
    ), f"Batch images shape mismatch. Expected {expected_img_shape}, got {images.shape}"

    # Positions: (Batch, Seq_Len, 1)
    assert positions.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        1,
    ), f"Positions shape mismatch. Got {positions.shape}"

    # Targets: (Batch, 8) -> 7 vertebrae + 1 patient_overall
    assert targets.shape == (
        Config.BATCH_SIZE,
        8,
    ), f"Targets shape mismatch. Got {targets.shape}"

    print(f"    DataLoader passed. Batch shapes verified.")

    # ------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # ------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    device = torch.device("cpu")  # Use CPU for simple architecture check
    model = CervicalFractureModel(
        pretrained=False
    )  # Skip downloading weights for speed
    model.to(device)
    model.eval()

    with torch.no_grad():
        # Forward pass with the batch fetched earlier
        logits = model(images.to(device), positions.to(device))

    # Output should be (Batch, 7)
    # Note: The model outputs 7 logits (C1-C7). Patient overall is derived in loss/metric.
    assert logits.shape == (
        Config.BATCH_SIZE,
        7,
    ), f"Model output shape mismatch. Expected ({Config.BATCH_SIZE}, 7), got {logits.shape}"

    print(f"    Model forward pass passed. Output logits shape: {logits.shape}")

    # ------------------------------------------------------------------------
    # 5. Run Full Training Pipeline (Debug Mode)
    # ------------------------------------------------------------------------
    print("\n[5] Executing Training Pipeline (Debug Mode)...")

    # This runs the trainer.run_training function
    # It will use the Config overrides we set earlier (1 Epoch, small subset)
    try:
        run_training(debug=True)
        print("    run_training executed successfully.")
    except Exception as e:
        print(f"    run_training failed with error: {e}")
        raise e

    # Check if artifacts were saved
    expected_model_path = "working/best_model.pth"
    if os.path.exists(expected_model_path):
        print(f"    Artifact check passed: {expected_model_path} exists.")
    else:
        # It's possible validation didn't improve in 1 epoch with random weights,
        # so best_model might not be saved if logic dictates strictly < best_loss.
        # However, trainer.py initializes best_val_loss = infinity, so the first val should save.
        raise AssertionError(f"Artifact check failed: {expected_model_path} not found.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
