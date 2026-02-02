import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import cv2

# Import from the provided library files
from library.config import Config
from library.utils import (
    set_seed,
    load_dicom,
    normalize_pixels,
    resize_image,
    select_middle_indices,
)
from library.dataset import MGMTDataset
from library.model import MGMTNet
from library.train import run_training
from library.predict import generate_submission


def test_utils():
    """
    Validates utility functions in library/utils.py
    """
    print("\n[1/5] Testing Utility Functions...")

    # 1. Test select_middle_indices
    # Case A: More files than slices
    indices = select_middle_indices(num_files=10, num_slices=3)
    assert indices == [3, 4, 5], f"Expected [3, 4, 5], got {indices}"
    # Case B: Fewer files than slices
    indices = select_middle_indices(num_files=2, num_slices=3)
    assert indices == [0, 1], f"Expected [0, 1], got {indices}"
    print("  - select_middle_indices: OK")

    # 2. Test resize_image
    dummy_img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    resized = resize_image(dummy_img, size=64)
    assert resized.shape == (64, 64), f"Expected (64, 64), got {resized.shape}"
    print("  - resize_image: OK")

    # 3. Test normalize_pixels
    dummy_img = np.array([0, 127, 255], dtype=np.float32)
    normalized = normalize_pixels(dummy_img)
    assert np.isclose(normalized.min(), 0.0), "Min value should be 0.0"
    assert np.isclose(normalized.max(), 1.0), "Max value should be 1.0"
    print("  - normalize_pixels: OK")


def test_dataset():
    """
    Validates Dataset class in library/dataset.py
    """
    print("\n[2/5] Testing MGMTDataset...")

    # Load actual metadata to create a valid subset
    full_train_df = pd.read_parquet(Config.TRAIN_METADATA)

    # Create a tiny subset dataframe for testing
    # We use a unique split_name to avoid conflict with existing caches
    subset_df = full_train_df.head(4).copy()

    # Initialize Dataset (this triggers data loading/processing)
    # We set load_cached_data=False to force the processing pipeline to run
    dataset = MGMTDataset(subset_df, split_name="demo_test", load_cached_data=False)

    # Verify length
    assert len(dataset) == 4, f"Expected 4 samples, got {len(dataset)}"

    # Verify item structure
    img_tensor, target_tensor = dataset[0]

    # Check Tensor Shape: (Channels, H, W)
    # Channels = 4 modalities * 3 slices = 12
    expected_channels = len(Config.MODALITIES) * Config.NUM_SLICES
    assert img_tensor.shape == (
        expected_channels,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Expected shape ({expected_channels}, {Config.IMAGE_SIZE}, {Config.IMAGE_SIZE}), got {img_tensor.shape}"

    # Check Target
    assert isinstance(target_tensor, torch.Tensor), "Target should be a tensor"
    assert target_tensor.ndim == 0, "Target should be a scalar tensor"

    print(f"  - Dataset loaded {len(dataset)} samples successfully.")
    print(f"  - Sample shape: {img_tensor.shape}")
    print("  - MGMTDataset: OK")

    return dataset


def test_model():
    """
    Validates Model architecture in library/model.py
    """
    print("\n[3/5] Testing MGMTNet Architecture...")

    # Initialize model
    model = MGMTNet(pretrained=False)  # No need to download weights for shape check
    model.eval()

    # Create dummy input batch
    batch_size = 2
    channels = len(Config.MODALITIES) * Config.NUM_SLICES
    size = Config.IMAGE_SIZE
    dummy_input = torch.randn(batch_size, channels, size, size)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape: (Batch_Size, Num_Classes) -> (2, 1)
    assert output.shape == (
        batch_size,
        1,
    ), f"Expected output shape (2, 1), got {output.shape}"

    print("  - Model forward pass successful.")
    print("  - MGMTNet: OK")


def test_training_pipeline():
    """
    Validates the training loop in library/train.py
    """
    print("\n[4/5] Testing Training Pipeline...")

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 8  # Very small dataset
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 1  # Only 1 epoch
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small test

    # Define a temporary model path for this test
    demo_model_path = os.path.join(Config.WORKING_DIR, "demo_model.pth")

    # Run training
    best_auc = run_training(
        debug=True,
        epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        save_path=demo_model_path,
        device_name="cpu",  # Use CPU for simple logic verification to avoid CUDA init overhead if any
    )

    # Verify model file creation
    assert os.path.exists(demo_model_path), "Model checkpoint was not saved."
    print(f"  - Training completed. Best AUC: {best_auc}")
    print("  - run_training: OK")

    return demo_model_path


def test_prediction_pipeline(model_path):
    """
    Validates the prediction generation in library/predict.py
    """
    print("\n[5/5] Testing Prediction Pipeline...")

    demo_submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Run prediction
    generate_submission(
        model_path=model_path,
        output_path=demo_submission_path,
        batch_size=Config.BATCH_SIZE,
        num_workers=0,
        device_name="cpu",
    )

    # Verify submission file
    assert os.path.exists(demo_submission_path), "Submission file was not created."

    # Verify content format
    df = pd.read_csv(demo_submission_path)
    assert "BraTS21ID" in df.columns, "Missing BraTS21ID column"
    assert "MGMT_value" in df.columns, "Missing MGMT_value column"
    assert len(df) > 0, "Submission file is empty"

    # Check ID format (should be int as per sample submission)
    assert pd.api.types.is_integer_dtype(df["BraTS21ID"]), "BraTS21ID should be integer"

    print(f"  - Submission generated with {len(df)} rows.")
    print("  - generate_submission: OK")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # Setup working directory for demo
    Config.WORKING_DIR = "./working/demo"
    Config.initialize()

    print("=" * 40)
    print(" STARTING LIBRARY VERIFICATION")
    print("=" * 40)

    try:
        # 1. Utils
        test_utils()

        # 2. Dataset
        test_dataset()

        # 3. Model
        test_model()

        # 4. Training
        model_path = test_training_pipeline()

        # 5. Prediction
        test_prediction_pipeline(model_path)

        print("\n" + "=" * 40)
        print(" ALL TESTS PASSED SUCCESSFULLY")
        print("=" * 40)

    except AssertionError as e:
        print(f"\n[FAIL] Assertion Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
