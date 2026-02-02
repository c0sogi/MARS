import os
import sys
import pandas as pd
import torch
import numpy as np
import warnings

# Add current directory to path to ensure library imports work
sys.path.append(".")

# Import library components
from library.config import Config, seed_everything
from library.dataset import CervicalSpineDataset, get_slice_cache, get_bbox_cache
from library.model import CervicalFractureNet
from library.loss import TriLevelFractureLoss
from library.engine import fit, generate_submission
from torch.utils.data import DataLoader


def main():
    # --- 1. Setup & Configuration ---
    print("[Demo] Setting up configuration...")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set seed for reproducibility
    seed_everything(42)

    # Override Config parameters for a fast demo run
    Config.DEBUG = True
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.SEQ_LEN = 16  # Reduced from 96 to speed up I/O and compute
    Config.NUM_WORKERS = 0  # Use main process to avoid multiprocessing overhead in demo
    Config.IMAGE_SIZE = 256  # Reduced image size for speed

    # Define a specific output directory for this demo
    Config.OUTPUT_DIR = os.path.join(Config.WORKING_DIR, "demo_execution")
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Update paths in Config to point to demo outputs
    Config.CACHE_DIR = Config.OUTPUT_DIR
    Config.SUBMISSION_PATH = os.path.join(Config.OUTPUT_DIR, "submission.csv")

    # --- 2. Data Preparation ---
    print("[Demo] Preparing data...")

    # Load metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Subsample data: 4 train samples, 2 val samples, 2 test studies
    # This ensures we have enough for at least one batch of size 2
    train_meta_demo = train_meta.iloc[:4].copy()
    val_meta_demo = val_meta.iloc[:2].copy()
    test_meta_demo = test_meta.iloc[:2].copy()

    # Generate Caches (Slices and Bounding Boxes)
    # We pass the combined metadata so the cache covers all needed studies
    combined_meta = pd.concat(
        [train_meta_demo, val_meta_demo, test_meta_demo], ignore_index=True
    )

    # Force re-computation or local caching by setting load_cached_data=False initially
    # or relying on the new CACHE_DIR being empty.
    slice_cache = get_slice_cache(combined_meta, load_cached_data=True)
    bbox_cache = get_bbox_cache(Config.BOUNDING_BOX_PATH, load_cached_data=True)

    # Initialize Datasets
    train_ds = CervicalSpineDataset(
        metadata_df=train_meta_demo,
        study_to_slices=slice_cache,
        study_to_bboxes=bbox_cache,
        is_train=True,
        seq_len=Config.SEQ_LEN,
    )

    val_ds = CervicalSpineDataset(
        metadata_df=val_meta_demo,
        study_to_slices=slice_cache,
        study_to_bboxes=bbox_cache,
        is_train=True,  # True to get targets for validation evaluation
        seq_len=Config.SEQ_LEN,
    )

    # Initialize DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # --- 3. Data Verification ---
    print("[Demo] Verifying data loader...")
    images, targets = next(iter(train_loader))

    # Expected Image Shape: (Batch, Seq, Channels, Height, Width)
    # Channels = 3 (from Config.IN_CHANNELS / Dataset logic)
    expected_shape = (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    )

    if images.shape != expected_shape:
        raise AssertionError(
            f"Image batch shape mismatch. Expected {expected_shape}, got {images.shape}"
        )

    # Expected Targets
    if targets["study_labels"].shape != (Config.BATCH_SIZE, 8):
        raise AssertionError(
            f"Study labels shape mismatch. Expected {(Config.BATCH_SIZE, 8)}, got {targets['study_labels'].shape}"
        )

    print(f"   Batch Shape Verified: {images.shape}")

    # --- 4. Model Initialization ---
    print("[Demo] Initializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model (pretrained=False for offline execution safety and speed)
    model = CervicalFractureNet(pretrained=False).to(device)

    # Verify Forward Pass
    print("[Demo] Verifying forward pass...")
    with torch.no_grad():
        dummy_input = images.to(device)
        outputs = model(dummy_input)

    # Check outputs
    if outputs["study_logits"].shape != (Config.BATCH_SIZE, 8):
        raise AssertionError("Output study_logits shape mismatch.")
    if outputs["slice_logits"].shape != (Config.BATCH_SIZE, Config.SEQ_LEN):
        raise AssertionError("Output slice_logits shape mismatch.")

    print("   Forward pass successful.")

    # --- 5. Training Loop ---
    print("[Demo] Starting training...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # No scheduler for this short demo

    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        device=device,
        epochs=Config.EPOCHS,
        patience=1,
        save_dir=Config.OUTPUT_DIR,
    )

    # --- 6. Inference & Submission ---
    print("[Demo] Generating submission...")

    # To use the existing generate_submission function which reads from Config.TEST_METADATA_PATH,
    # we temporarily point Config to a file containing our subsampled test data.
    temp_test_meta_path = os.path.join(Config.OUTPUT_DIR, "test_metadata.csv")
    test_meta_demo.to_csv(temp_test_meta_path, index=False)
    Config.TEST_METADATA_PATH = temp_test_meta_path

    generate_submission(model, device)

    # --- 7. Submission Verification ---
    print("[Demo] Verifying submission file...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    required_cols = {"row_id", "fractured"}
    if not required_cols.issubset(sub_df.columns):
        raise AssertionError(
            f"Submission missing required columns. Found {sub_df.columns}"
        )

    # Check row count
    # 2 studies * 8 targets = 16 rows
    expected_rows = len(test_meta_demo) * 8
    if len(sub_df) != expected_rows:
        raise AssertionError(
            f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"
        )

    print(f"   Submission verified. {len(sub_df)} rows generated.")
    print("[Demo] Execution completed successfully.")


if __name__ == "__main__":
    main()
