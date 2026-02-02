import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.dicom_utils import read_dicom_robust, map_slice_ids
from library.network import GroupedEfficientNetV2
from library.data_loader import get_dataloaders, BraTSDataset
from library.trainer import run_training, set_seed


def setup_demo_environment():
    """
    Sets up a temporary environment for the demo.
    Creates a subset of the metadata to ensure the demo runs quickly.
    Modifies the global Config to point to these subsets.
    """
    # 1. Define paths
    demo_dir = os.path.join("working", "demo_execution")
    cache_dir = os.path.join(demo_dir, "cache")

    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(cache_dir, exist_ok=True)

    # 2. Create Metadata Subsets
    # Load original metadata
    orig_train = pd.read_csv("./metadata/train.csv")
    orig_val = pd.read_csv("./metadata/val.csv")
    orig_test = pd.read_csv("./metadata/test.csv")

    # Take small subsets (e.g., 5-10 samples)
    # Ensure we have enough for a batch
    subset_train = orig_train.head(8).copy()
    subset_val = orig_val.head(4).copy()
    subset_test = orig_test.head(4).copy()

    # Save subsets
    train_path = os.path.join(demo_dir, "train_demo.csv")
    val_path = os.path.join(demo_dir, "val_demo.csv")
    test_path = os.path.join(demo_dir, "test_demo.csv")

    subset_train.to_csv(train_path, index=False)
    subset_val.to_csv(val_path, index=False)
    subset_test.to_csv(test_path, index=False)

    print(f"Created demo metadata subsets in {demo_dir}")

    # 3. Override Config
    print("Overriding Config for demo execution...")
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = cache_dir

    Config.TRAIN_METADATA = train_path
    Config.VAL_METADATA = val_path
    Config.TEST_METADATA = test_path

    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "best_model_demo.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission_demo.csv")

    # Reduce compute load
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.PATIENCE = 1

    # Ensure reproducibility
    set_seed(Config.SEED)


def verify_dicom_utils():
    """
    Verifies the functionality of dicom_utils.py
    """
    print("\n--- Verifying DICOM Utils ---")

    # Get a valid path from the config (which now points to our demo subset)
    df = pd.read_csv(Config.TRAIN_METADATA)
    row = df.iloc[0]
    flair_dir = os.path.join(Config.INPUT_DIR, row["path_FLAIR"])

    # 1. Test map_slice_ids
    slice_map = map_slice_ids(flair_dir)
    if not slice_map:
        raise AssertionError(f"Failed to map slices in {flair_dir}")

    first_sid = sorted(slice_map.keys())[0]
    file_path = slice_map[first_sid]
    print(f"Mapped slice ID {first_sid} to {os.path.basename(file_path)}")

    # 2. Test read_dicom_robust
    # Test loading with resize
    target_size = (128, 128)
    img = read_dicom_robust(file_path, target_size=target_size)

    if not isinstance(img, np.ndarray):
        raise AssertionError("read_dicom_robust did not return a numpy array")

    if img.shape != target_size:
        raise AssertionError(
            f"Image shape mismatch. Expected {target_size}, got {img.shape}"
        )

    if img.dtype != np.float32:
        raise AssertionError(f"Image dtype mismatch. Expected float32, got {img.dtype}")

    print("DICOM reading and resizing successful.")


def verify_network():
    """
    Verifies the GroupedEfficientNetV2 model architecture.
    """
    print("\n--- Verifying Network Architecture ---")

    device = torch.device("cpu")  # Use CPU for quick structural check
    model = GroupedEfficientNetV2().to(device)
    model.eval()

    # Create dummy input: (Batch, Channels, H, W)
    # Channels = 12 (4 modalities * 3 slices)
    batch_size = 2
    dummy_input = torch.randn(batch_size, 12, 224, 224).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    # Expected output: (Batch, 1) (Logits for binary classification)
    if output.shape != (batch_size, 1):
        raise AssertionError(
            f"Network output shape mismatch. Expected {(batch_size, 1)}, got {output.shape}"
        )

    print("Network forward pass successful. Output shape verified.")


def verify_data_pipeline():
    """
    Verifies the DataLoader and Dataset classes.
    """
    print("\n--- Verifying Data Pipeline ---")

    # This will trigger the FidelityPreprocessing cache generation on the subset
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # Fetch one batch
    images, labels = next(iter(train_loader))

    # Check shapes
    expected_channels = 12
    expected_hw = 224

    if images.shape[1] != expected_channels:
        raise AssertionError(
            f"Batch channel count mismatch. Expected {expected_channels}, got {images.shape[1]}"
        )

    if images.shape[2] != expected_hw or images.shape[3] != expected_hw:
        raise AssertionError(
            f"Batch spatial dim mismatch. Expected {expected_hw}x{expected_hw}, got {images.shape[2:]}"
        )

    if labels.shape[0] != Config.BATCH_SIZE:
        raise AssertionError(
            f"Batch size mismatch. Expected {Config.BATCH_SIZE}, got {labels.shape[0]}"
        )

    print("Data Pipeline verified. Batch shapes are correct.")


def verify_full_training_loop():
    """
    Runs the full training loop (1 epoch) using the subset data.
    """
    print("\n--- Verifying Full Training Loop ---")

    # run_training uses the global Config, which we have overridden
    run_training()

    # Verify artifacts
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise AssertionError("Model checkpoint was not saved.")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError("Submission file was not generated.")

    # Verify submission format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    required_cols = ["BraTS21ID", "MGMT_value"]
    if not all(col in sub_df.columns for col in required_cols):
        raise AssertionError(f"Submission file missing columns. Found {sub_df.columns}")

    if len(sub_df) == 0:
        raise AssertionError("Submission file is empty.")

    print(
        f"Full training loop completed successfully. Submission generated at {Config.SUBMISSION_PATH}"
    )


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    try:
        # 1. Setup
        setup_demo_environment()

        # 2. Verify Components
        verify_dicom_utils()
        verify_network()
        verify_data_pipeline()

        # 3. Verify Integration
        verify_full_training_loop()

        print("\nAll demonstrations and verifications passed successfully.")

    except Exception as e:
        print(f"\nFAILED: {e}")
        sys.exit(1)
