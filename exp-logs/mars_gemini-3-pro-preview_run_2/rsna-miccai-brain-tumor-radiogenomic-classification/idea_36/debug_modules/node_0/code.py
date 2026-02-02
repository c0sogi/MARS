import os
import shutil
import pandas as pd
import numpy as np
import torch
import sys

# Import library modules
# We import config first to monkey-patch it for the demo
import library.config as config
import library.dicom_utils as dicom_utils
import library.data_factory as data_factory
import library.model_factory as model_factory
import library.engine as engine


def run_demo():
    print("=== Starting Glioblastoma Subtype Classification Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration & Setup for Demo
    # -------------------------------------------------------------------------
    # Define a specific directory for this demo execution to avoid conflicts
    demo_dir = "./working/demo_execution"
    os.makedirs(demo_dir, exist_ok=True)

    # Create a cache directory inside the demo dir
    demo_cache_dir = os.path.join(demo_dir, "cache")
    os.makedirs(demo_cache_dir, exist_ok=True)

    print(f"Working directory: {demo_dir}")

    # Monkey-patch library.config to use the demo directory and reduced parameters
    # This ensures the library functions use our demo settings without modifying the files
    config.WORKING_DIR = demo_cache_dir  # Data loader saves .npy cache here
    config.MODEL_SAVE_PATH = os.path.join(demo_dir, "best_model_demo.pth")
    config.SUBMISSION_PATH = os.path.join(demo_dir, "submission_demo.csv")

    # Update cache file paths in config to point to the new demo cache dir
    config.TRAIN_CACHE_DATA = os.path.join(demo_cache_dir, "train_data.npy")
    config.TRAIN_CACHE_LABELS = os.path.join(demo_cache_dir, "train_labels.npy")
    config.VAL_CACHE_DATA = os.path.join(demo_cache_dir, "val_data.npy")
    config.VAL_CACHE_LABELS = os.path.join(demo_cache_dir, "val_labels.npy")
    config.TEST_CACHE_DATA = os.path.join(demo_cache_dir, "test_data.npy")
    config.TEST_CACHE_IDS = os.path.join(demo_cache_dir, "test_ids.npy")

    # Reduce training parameters for speed
    config.NUM_EPOCHS = 2
    config.BATCH_SIZE = 4
    config.NUM_WORKERS = 2  # Reduce workers for small batch

    # -------------------------------------------------------------------------
    # 2. Create Demo Metadata (Subset)
    # -------------------------------------------------------------------------
    print("\n--- Creating Demo Metadata Subsets ---")

    # Load original metadata
    orig_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    orig_val = pd.read_csv(config.VAL_METADATA_PATH)
    orig_test = pd.read_csv(config.TEST_METADATA_PATH)

    # Sample subsets (Train: 10, Val: 5, Test: 5)
    # We ensure we pick subjects that actually have files (metadata generation checked this, but good to be safe)
    demo_train = orig_train.head(10).copy()
    demo_val = orig_val.head(5).copy()
    demo_test = orig_test.head(5).copy()

    # Save demo metadata
    demo_train_path = os.path.join(demo_dir, "train_demo.csv")
    demo_val_path = os.path.join(demo_dir, "val_demo.csv")
    demo_test_path = os.path.join(demo_dir, "test_demo.csv")

    demo_train.to_csv(demo_train_path, index=False)
    demo_val.to_csv(demo_val_path, index=False)
    demo_test.to_csv(demo_test_path, index=False)

    # Update config to point to demo metadata
    config.TRAIN_METADATA_PATH = demo_train_path
    config.VAL_METADATA_PATH = demo_val_path
    config.TEST_METADATA_PATH = demo_test_path

    print(f"Demo Train samples: {len(demo_train)}")
    print(f"Demo Val samples: {len(demo_val)}")
    print(f"Demo Test samples: {len(demo_test)}")

    # -------------------------------------------------------------------------
    # 3. Verify DICOM Utilities
    # -------------------------------------------------------------------------
    print("\n--- Verifying DICOM Utilities ---")

    # Pick a sample FLAIR path from the first training subject
    sample_row = demo_train.iloc[0]
    flair_dir = os.path.join(config.INPUT_DIR, sample_row["path_FLAIR"])

    # Find a .dcm file
    flair_files = [f for f in os.listdir(flair_dir) if f.endswith(".dcm")]
    if flair_files:
        sample_dcm_path = os.path.join(flair_dir, flair_files[0])
        print(f"Testing read on: {sample_dcm_path}")

        # Test read_dicom_raw
        img_raw = dicom_utils.read_dicom_raw(sample_dcm_path)
        print(f"Raw image shape: {img_raw.shape}, dtype: {img_raw.dtype}")

        # Test get_image_plane (resize)
        img_resized = dicom_utils.get_image_plane(img_raw)
        print(f"Resized image shape: {img_resized.shape}")

        assert img_resized.shape == (config.IMG_SIZE, config.IMG_SIZE), "Resize failed"
    else:
        print("Warning: No DICOM files found in sample directory to test utils.")

    # -------------------------------------------------------------------------
    # 4. Verify Data Factory (Pipeline & Loader)
    # -------------------------------------------------------------------------
    print("\n--- Verifying Data Factory ---")

    # This will trigger processing and caching for our demo subset
    # We set load_cached_data=False to force processing the demo subset
    train_loader, val_loader, test_loader = data_factory.get_dataloaders(
        load_cached_data=False
    )

    # Check Train Loader
    inputs, targets = next(iter(train_loader))
    print(f"Train Batch - Inputs: {inputs.shape}, Targets: {targets.shape}")

    # Expected: (Batch, Channels, H, W) -> (4, 12, 256, 256)
    expected_channels = (
        config.NUM_MODALITIES * config.NUM_SLICES_PER_MODALITY
    )  # 4 * 3 = 12
    assert inputs.shape == (
        config.BATCH_SIZE,
        expected_channels,
        config.IMG_SIZE,
        config.IMG_SIZE,
    )
    assert targets.shape == (config.BATCH_SIZE,)
    assert inputs.dtype == torch.float32

    print("Data Pipeline verified successfully.")

    # -------------------------------------------------------------------------
    # 5. Verify Model Factory
    # -------------------------------------------------------------------------
    print("\n--- Verifying Model Factory ---")

    model = model_factory.get_model()
    model.eval()  # Set to eval for deterministic check

    # Move sample input to device
    inputs = inputs.to(config.DEVICE)

    with torch.no_grad():
        outputs = model(inputs)

    print(f"Model Output Shape: {outputs.shape}")

    # Expected: (Batch, 1) - Logits
    assert outputs.shape == (config.BATCH_SIZE, 1)
    print("Model architecture verified successfully.")

    # -------------------------------------------------------------------------
    # 6. Run Full Training & Inference Engine
    # -------------------------------------------------------------------------
    print("\n--- Running Training Engine (Demo) ---")

    # We already loaded data, but run_training calls get_dataloaders internally.
    # Since we set config paths and cache paths, it will pick up the cached demo data
    # we just generated in step 4 (load_cached_data=True is default in run_training).

    engine.run_training(load_cached_data=True)

    # -------------------------------------------------------------------------
    # 7. Validate Submission
    # -------------------------------------------------------------------------
    print("\n--- Validating Submission ---")

    if os.path.exists(config.SUBMISSION_PATH):
        df_sub = pd.read_csv(config.SUBMISSION_PATH)
        print("Submission file contents (head):")
        print(df_sub.head())

        # Check columns
        assert "BraTS21ID" in df_sub.columns
        assert "MGMT_value" in df_sub.columns

        # Check row count (should match demo test set size)
        assert len(df_sub) == len(demo_test)

        # Check value range (probabilities)
        assert df_sub["MGMT_value"].min() >= 0.0
        assert df_sub["MGMT_value"].max() <= 1.0

        print("Submission format verified.")
    else:
        raise FileNotFoundError(
            f"Submission file not found at {config.SUBMISSION_PATH}"
        )

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        run_demo()
    except Exception as e:
        print(f"\n!!! Demo Failed: {e} !!!")
        import traceback

        traceback.print_exc()
        sys.exit(1)
