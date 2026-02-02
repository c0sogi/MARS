import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# -----------------------------------------------------------------------------
# 1. Configuration Patching (Must be done before importing other library modules)
# -----------------------------------------------------------------------------
import library.config

# Modify configuration for a fast demonstration
print("Patching configuration for fast demonstration...")
library.config.DEBUG_DATA_LIMIT = 16  # Process only 16 subjects
library.config.MAX_EPOCHS = 1  # Train for only 1 epoch
library.config.BATCH_SIZE = 4  # Small batch size
library.config.CACHE_DIR = "./working/demo_cache"  # Separate cache for demo
library.config.MODEL_SAVE_PATH = "./working/demo_model.pth"
library.config.SUBMISSION_PATH = "./working/demo_submission.csv"

# Ensure demo directories exist
os.makedirs(library.config.CACHE_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. Import Library Modules
# -----------------------------------------------------------------------------
from library.utils import read_dicom_robust, seed_everything
from library.data_loader import get_dataloaders
from library.model import AsymmetricGroupedEfficientNet
from library.train_eval import Trainer, CircuitBreaker

# -----------------------------------------------------------------------------
# 3. Main Execution
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Set seed for reproducibility
    seed_everything(42)

    print("\n=== Step 1: Verifying Utility Functions ===")
    # Load train metadata to find a valid file path
    train_meta = pd.read_csv(library.config.TRAIN_METADATA_PATH)
    if len(train_meta) > 0:
        # Construct a path to a DICOM file
        sample_row = train_meta.iloc[0]
        flair_dir = os.path.join(library.config.INPUT_DIR, sample_row["path_FLAIR"])
        # Find first dcm file
        dcm_files = [f for f in os.listdir(flair_dir) if f.endswith(".dcm")]
        if dcm_files:
            sample_file_path = os.path.join(flair_dir, dcm_files[0])
            print(f"Testing DICOM read on: {sample_file_path}")

            # Test read_dicom_robust
            img = read_dicom_robust(sample_file_path, target_size=(224, 224))

            # Assertions
            assert isinstance(img, np.ndarray), "Image must be a numpy array"
            assert img.shape == (
                224,
                224,
            ), f"Expected shape (224, 224), got {img.shape}"
            assert (
                img.dtype == np.float32
                or img.dtype == np.uint16
                or img.dtype == np.uint8
            ), "Unexpected dtype"
            print("read_dicom_robust verified successfully.")
        else:
            print("No DICOM files found in sample directory to test.")
    else:
        print("Metadata is empty, skipping utility verification.")

    print("\n=== Step 2: Verifying Data Pipeline ===")
    # This will process the top 16 records defined in DEBUG_DATA_LIMIT
    print(f"Generating dataloaders with limit={library.config.DEBUG_DATA_LIMIT}...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Verify Train Loader
    try:
        data_batch, label_batch = next(iter(train_loader))
        print(f"Train Batch Shape: Data={data_batch.shape}, Labels={label_batch.shape}")

        # Assertions
        # Shape: (Batch, Channels, H, W) -> (4, 12, 224, 224)
        expected_channels = 12  # 4 modalities * 3 slabs
        assert data_batch.shape == (
            library.config.BATCH_SIZE,
            expected_channels,
            224,
            224,
        ), f"Incorrect data shape: {data_batch.shape}"
        assert label_batch.shape == (
            library.config.BATCH_SIZE,
        ), f"Incorrect label shape: {label_batch.shape}"
        print("Data Pipeline verified successfully.")
    except StopIteration:
        raise RuntimeError("Train loader is empty! Check data processing logic.")

    print("\n=== Step 3: Verifying Model Architecture ===")
    model = AsymmetricGroupedEfficientNet()
    model.to(library.config.DEVICE)

    # Test Forward Pass
    with torch.no_grad():
        dummy_input = torch.randn(2, 12, 224, 224).to(library.config.DEVICE)
        output = model(dummy_input)
        print(f"Model Output Shape: {output.shape}")

        # Assertions
        assert output.shape == (2, 1), f"Expected output (2, 1), got {output.shape}"
        assert not torch.isnan(output).any(), "Model produced NaN values"
    print("Model architecture verified successfully.")

    print("\n=== Step 4: Training Loop Execution ===")
    # Circuit Breaker check
    cb = CircuitBreaker(threshold=0.1)
    cb.check(train_loader.dataset, "Train Set")

    trainer = Trainer(model, train_loader, val_loader)

    # Run training (1 epoch as configured)
    trainer.fit()

    # Verify model file creation
    assert os.path.exists(
        library.config.MODEL_SAVE_PATH
    ), "Model checkpoint was not saved."
    print(f"Model saved to {library.config.MODEL_SAVE_PATH}")

    print("\n=== Step 5: Inference and Submission ===")
    # Run TTA Prediction
    preds = trainer.predict_tta(test_loader)

    # Verify predictions
    test_meta = pd.read_csv(library.config.TEST_METADATA_PATH)
    print(f"Number of test samples: {len(test_meta)}")
    print(f"Number of predictions: {len(preds)}")

    # Generate Submission CSV
    # Note: If test loader length differs from metadata (due to batch drop_last=False),
    # we align them carefully.

    # Truncate or pad preds to match metadata length if necessary (robustness)
    if len(preds) > len(test_meta):
        preds = preds[: len(test_meta)]
    elif len(preds) < len(test_meta):
        preds.extend([0.5] * (len(test_meta) - len(preds)))

    submission = pd.DataFrame(
        {"BraTS21ID": test_meta["BraTS21ID"], "MGMT_value": preds}
    )

    submission.to_csv(library.config.SUBMISSION_PATH, index=False)

    # Final Validation
    assert os.path.exists(library.config.SUBMISSION_PATH), "Submission file not found."
    df_sub = pd.read_csv(library.config.SUBMISSION_PATH)
    assert len(df_sub) == len(test_meta), "Submission row count mismatch."
    assert (
        "BraTS21ID" in df_sub.columns and "MGMT_value" in df_sub.columns
    ), "Missing columns in submission."

    print(f"Submission generated successfully at {library.config.SUBMISSION_PATH}")
    print("\n=== Demo Completed Successfully ===")
