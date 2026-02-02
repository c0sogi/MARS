import os
import pandas as pd
import numpy as np
import torch
import sys

# Import necessary components from the provided library
from library.config import (
    INPUT_DIR,
    TRAIN_METADATA_PATH,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    IMG_SIZE,
    WORKING_DIR,
    TRAIN_CACHE_PATH,
    TRAIN_LABEL_CACHE_PATH,
)
from library.dicom_utils import read_dicom_robust, process_image
from library.dataset import load_dataset
from library.model import AsymmetricEfficientNet
from library.trainer import train_model, set_seed
from library.inference import generate_submission


def run_demo():
    print("=== Starting Library Demo ===\n")

    # Ensure reproducibility
    set_seed(42)

    # -------------------------------------------------------------------------
    # 1. Verify DICOM Utilities
    # -------------------------------------------------------------------------
    print("--- 1. Verifying DICOM Utilities ---")

    # Load metadata to find a valid file path
    df_train = pd.read_csv(TRAIN_METADATA_PATH)
    sample_row = df_train.iloc[0]

    # Construct a path to a real DICOM file (FLAIR modality)
    # Metadata paths are relative, e.g., "train/00000/FLAIR"
    flair_dir = os.path.join(INPUT_DIR, sample_row["path_FLAIR"])

    # Find the first DICOM file in the directory
    if os.path.exists(flair_dir):
        files = [f for f in os.listdir(flair_dir) if f.endswith(".dcm")]
        if files:
            sample_file_path = os.path.join(flair_dir, files[0])
            print(f"Testing with file: {sample_file_path}")

            # Test read_dicom_robust
            img_raw = read_dicom_robust(sample_file_path)

            # Check raw output
            if np.sum(img_raw) == 0:
                print(
                    "Warning: Image is empty (all zeros). This might be a blank slice or read error."
                )
            else:
                print(
                    f"Raw image read successfully. Shape: {img_raw.shape}, Dtype: {img_raw.dtype}"
                )

            # Test process_image
            img_proc = process_image(img_raw)

            # Assertions for processed image
            assert img_proc.shape == (
                IMG_SIZE,
                IMG_SIZE,
            ), f"Processed image shape mismatch. Expected ({IMG_SIZE}, {IMG_SIZE}), got {img_proc.shape}"
            assert (
                img_proc.dtype == np.float32
            ), f"Processed image dtype mismatch. Expected float32, got {img_proc.dtype}"

            print("DICOM processing verification passed.")
        else:
            print("No DICOM files found in sample directory to test.")
    else:
        print(f"Sample directory {flair_dir} does not exist.")

    print("")

    # -------------------------------------------------------------------------
    # 2. Verify Dataset Loading (Subset)
    # -------------------------------------------------------------------------
    print("--- 2. Verifying Dataset Loading ---")

    # Define a small subset size for debugging
    subset_size = 4

    # Clean up any existing cache for this demo to ensure we test generation logic
    # (In a real run, we would keep the cache, but here we want to prove generation works)
    if os.path.exists(TRAIN_CACHE_PATH):
        try:
            # Only remove if it doesn't match our subset size logic to avoid confusion
            # The library handles mismatch, but let's be clean.
            data = np.load(TRAIN_CACHE_PATH)
            if len(data) != subset_size:
                os.remove(TRAIN_CACHE_PATH)
                if os.path.exists(TRAIN_LABEL_CACHE_PATH):
                    os.remove(TRAIN_LABEL_CACHE_PATH)
        except Exception:
            pass

    print(f"Loading dataset with debug_max_samples={subset_size}...")
    dataset = load_dataset(
        metadata_path=TRAIN_METADATA_PATH,
        cache_path_data=TRAIN_CACHE_PATH,
        cache_path_labels=TRAIN_LABEL_CACHE_PATH,
        load_cached_data=True,
        transform=None,
        debug_max_samples=subset_size,
    )

    # Assertions
    assert (
        len(dataset) == subset_size
    ), f"Dataset length mismatch. Expected {subset_size}, got {len(dataset)}"

    # Check item structure
    x, y = dataset[0]
    print(f"Sample Tensor Shape: {x.shape}")
    print(f"Sample Label: {y}")

    # Expected shape: (12, 224, 224) -> 12 channels (4 groups * 3 slices)
    assert x.shape == (
        12,
        IMG_SIZE,
        IMG_SIZE,
    ), f"Input tensor shape mismatch. Expected (12, {IMG_SIZE}, {IMG_SIZE}), got {x.shape}"
    assert isinstance(y, torch.Tensor), "Label should be a torch Tensor"

    print("Dataset verification passed.")
    print("")

    # -------------------------------------------------------------------------
    # 3. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("--- 3. Verifying Model Architecture ---")

    model = AsymmetricEfficientNet()
    model.eval()

    # Create a dummy input batch: (Batch_Size=2, Channels=12, H=224, W=224)
    dummy_input = torch.randn(2, 12, IMG_SIZE, IMG_SIZE)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"

    print("Model verification passed.")
    print("")

    # -------------------------------------------------------------------------
    # 4. Run Training Loop (Demo)
    # -------------------------------------------------------------------------
    print("--- 4. Running Training Loop (Demo) ---")

    # Run training on a very small subset (e.g., 10 samples) to ensure it finishes quickly
    # The config specifies 15 epochs, but with 10 samples, this will be instant.
    train_subset_size = 10

    print(f"Invoking train_model with debug_max_samples={train_subset_size}...")
    train_model(debug_max_samples=train_subset_size)

    # Verify artifacts
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Success: Best model saved to {MODEL_SAVE_PATH}")
    else:
        raise AssertionError(
            f"Training failed to produce model file at {MODEL_SAVE_PATH}"
        )

    print("Training demo passed.")
    print("")

    # -------------------------------------------------------------------------
    # 5. Run Inference & Submission (Demo)
    # -------------------------------------------------------------------------
    print("--- 5. Running Inference Generation (Demo) ---")

    test_subset_size = 5

    print(f"Invoking generate_submission with debug_max_samples={test_subset_size}...")
    generate_submission(debug_max_samples=test_subset_size, load_cached_data=False)

    # Verify submission file
    if os.path.exists(SUBMISSION_PATH):
        print(f"Success: Submission file created at {SUBMISSION_PATH}")

        # Check content format
        df_sub = pd.read_csv(SUBMISSION_PATH)
        print("Submission Head:")
        print(df_sub.head())

        assert (
            len(df_sub) == test_subset_size
        ), f"Submission length mismatch. Expected {test_subset_size}, got {len(df_sub)}"
        assert (
            "BraTS21ID" in df_sub.columns and "MGMT_value" in df_sub.columns
        ), "Submission columns mismatch."
        assert df_sub["MGMT_value"].dtype == float, "MGMT_value should be float."

    else:
        raise AssertionError(
            f"Inference failed to produce submission file at {SUBMISSION_PATH}"
        )

    print("Inference demo passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
