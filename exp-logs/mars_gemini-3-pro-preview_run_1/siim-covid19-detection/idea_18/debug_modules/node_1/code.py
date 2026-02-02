import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders, get_test_dataloader
from library.model import StochasticDepthResNet34UNet
from library.engine import train_one_epoch, validate, inference


def run_demo():
    print("==== Starting End-to-End Pipeline Demo ====")

    # 1. Setup & Configuration Overrides
    # We use a specific working directory for this demo to avoid conflicts
    demo_dir = "./working/demo_pipeline"
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config to speed up execution
    Config.WORKING_DIR = demo_dir
    Config.OUTPUT_DIR = demo_dir
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 2  # Reduce workers for small data to avoid overhead
    Config.IMG_SIZE = (512, 512)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Create Mini-Datasets (Subsetting)
    # We read the full metadata but only save the first few rows to new CSVs
    # This ensures the heavy DICOM reading only happens for a few images.
    print("\n[Step 1] Creating mini-datasets for rapid testing...")

    train_full = pd.read_csv("./metadata/train.csv")
    val_full = pd.read_csv("./metadata/val.csv")
    test_full = pd.read_csv("./metadata/test.csv")

    # Take 8 samples for train (2 batches), 4 for val, 4 for test
    train_mini = train_full.head(8).copy()
    val_mini = val_full.head(4).copy()
    test_mini = test_full.head(4).copy()

    mini_train_path = os.path.join(demo_dir, "train_mini.csv")
    mini_val_path = os.path.join(demo_dir, "val_mini.csv")
    mini_test_path = os.path.join(demo_dir, "test_mini.csv")

    train_mini.to_csv(mini_train_path, index=False)
    val_mini.to_csv(mini_val_path, index=False)
    test_mini.to_csv(mini_test_path, index=False)

    # Point Config to these new mini metadata files
    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path
    Config.TEST_METADATA_PATH = mini_test_path

    # 3. Data Loading & Verification
    print("\n[Step 2] Initializing DataLoaders...")
    # load_cached_data=False forces the system to process our new mini CSVs
    # instead of loading old .npy files from previous runs
    train_loader, val_loader = get_dataloaders(load_cached_data=False)

    # Verify Train Loader
    try:
        images, labels, masks = next(iter(train_loader))
        print(
            f"  Batch Shapes -> Images: {images.shape}, Labels: {labels.shape}, Masks: {masks.shape}"
        )

        # Assertions
        assert images.shape == (
            Config.BATCH_SIZE,
            3,
            512,
            512,
        ), "Image batch shape mismatch"
        assert labels.shape == (Config.BATCH_SIZE,), "Label batch shape mismatch"
        assert masks.shape == (
            Config.BATCH_SIZE,
            1,
            512,
            512,
        ), "Mask batch shape mismatch"
        assert images.dtype == torch.float32, "Images should be float32"
        assert labels.dtype == torch.long, "Labels should be long"

        print("  Data Loading Verified: OK")
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # 4. Model Initialization
    print("\n[Step 3] Initializing Model...")
    device = Config.DEVICE
    print(f"  Device: {device}")

    model = StochasticDepthResNet34UNet()
    model.to(device)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_input = images.to(device)
        cls_logits, seg_logits = model(dummy_input)

    print(f"  Output Shapes -> Cls: {cls_logits.shape}, Seg: {seg_logits.shape}")

    assert cls_logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_STUDY_CLASSES,
    ), "Classification output shape mismatch"
    assert seg_logits.shape == (
        Config.BATCH_SIZE,
        1,
        512,
        512,
    ), "Segmentation output shape mismatch"
    print("  Model Architecture Verified: OK")

    # 5. Training Loop
    print("\n[Step 4] Running Training Loop (1 Epoch)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run training
    train_loss = train_one_epoch(model, train_loader, optimizer, None, device, epoch=1)

    print(f"  Training Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss returned NaN!"
    print("  Training Loop Verified: OK")

    # 6. Validation Loop
    print("\n[Step 5] Running Validation Loop...")
    val_loss, val_map = validate(model, val_loader, device)

    print(f"  Val Loss: {val_loss:.4f}, mAP: {val_map:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0.0 <= val_map <= 1.0, "mAP score out of range [0, 1]"
    print("  Validation Logic Verified: OK")

    # 7. Inference & Submission
    print("\n[Step 6] Running Inference on Test Set...")
    test_loader = get_test_dataloader(load_cached_data=False)

    # Run inference
    inference(model, test_loader, device)

    submission_path = os.path.join(Config.OUTPUT_DIR, "submission.csv")
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    # Verify Submission Content
    sub_df = pd.read_csv(submission_path)
    print(f"  Submission generated with {len(sub_df)} rows.")
    print(f"  First few rows:\n{sub_df.head()}")

    # Check expected row count: Each test image generates 1 study row and 1 image row
    expected_rows = len(test_mini) * 2
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows in submission, found {len(sub_df)}"

    # Check columns
    assert (
        "id" in sub_df.columns and "PredictionString" in sub_df.columns
    ), "Submission columns missing"

    print("  Inference & Submission Verified: OK")

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
