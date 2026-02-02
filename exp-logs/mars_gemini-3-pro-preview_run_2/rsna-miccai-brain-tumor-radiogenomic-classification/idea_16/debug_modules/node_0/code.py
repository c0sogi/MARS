import os
import sys
import pandas as pd
import torch
import numpy as np
from functools import partial

# -----------------------------------------------------------------------------
# Suppress TQDM and Warnings for cleaner output
# -----------------------------------------------------------------------------
import tqdm


# Monkey-patch tqdm to disable progress bars from the library
def noop_tqdm(*args, **kwargs):
    if args:
        return args[0]
    return object()


# We need to patch the class constructor to default disable=True
# or simply replace the tqdm module's tqdm class if possible.
# Since the library imports tqdm, we'll try to set the disable flag globally if possible,
# but monkey-patching the constructor is more robust for already imported modules.
original_tqdm_init = tqdm.tqdm.__init__
tqdm.tqdm.__init__ = partial(original_tqdm_init, disable=True)

# -----------------------------------------------------------------------------
# Imports from Provided Library
# -----------------------------------------------------------------------------
from library.utils import seed_everything
from library.data import BraTSDataset, get_dataloader
from library.model import get_model
from library.train import run_training
from library.predict import predict_and_submit
from library import config


def test_data_pipeline():
    print("\n=== Testing Data Pipeline ===")

    # Load metadata
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    print(f"Loaded training metadata: {len(df_train)} records")

    # 1. Test Dataset Instantiation
    # Use a small subset to speed up ROI cache generation if it's not already there
    df_subset = df_train.head(10).copy()
    dataset = BraTSDataset(df_subset, phase="train", load_cached_roi=False)

    # 2. Test Single Item Retrieval
    image_tensor, target, brats_id = dataset[0]

    # Expected shape: (12, 224, 224) -> 4 modalities * 3 slices
    print(f"Single item shape: {image_tensor.shape}")
    assert image_tensor.shape == (
        12,
        224,
        224,
    ), f"Expected image shape (12, 224, 224), got {image_tensor.shape}"
    assert isinstance(target, torch.Tensor), "Target should be a tensor"

    # 3. Test DataLoader
    loader = get_dataloader(df_subset, phase="train", batch_size=4, num_workers=0)
    batch_images, batch_targets, batch_ids = next(iter(loader))

    print(f"Batch shape: {batch_images.shape}")
    assert batch_images.shape == (
        4,
        12,
        224,
        224,
    ), f"Expected batch shape (4, 12, 224, 224), got {batch_images.shape}"
    assert batch_targets.shape == (
        4,
    ), f"Expected target shape (4,), got {batch_targets.shape}"

    print("Data pipeline verified successfully.")


def test_model_architecture():
    print("\n=== Testing Model Architecture ===")

    # 1. Instantiate Model
    model = get_model()
    model.eval()

    # 2. Create Dummy Input
    # Shape: (Batch, Channels, Height, Width)
    dummy_input = torch.randn(2, 12, 224, 224)

    # 3. Forward Pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model output shape: {output.shape}")

    # Expected output: (Batch, Num_Classes) -> (2, 1)
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"

    print("Model architecture verified successfully.")


def test_training_loop():
    print("\n=== Testing Training Loop ===")

    # Run training in debug mode
    # debug=True restricts data to 32 samples
    # num_epochs=1 ensures quick completion
    print("Starting training run (Debug Mode)...")
    run_training(num_epochs=1, debug=True)

    # Verify artifact generation
    assert os.path.exists(
        config.BEST_MODEL_PATH
    ), f"Best model not found at {config.BEST_MODEL_PATH}"

    print("Training loop completed and model saved.")


def test_inference_pipeline():
    print("\n=== Testing Inference Pipeline ===")

    # Ensure test metadata exists
    assert os.path.exists(config.TEST_METADATA_PATH), "Test metadata missing"

    # Run inference
    # This uses the model saved in the previous step
    predict_and_submit()

    # Verify submission file
    assert os.path.exists(
        config.SUBMISSION_PATH
    ), f"Submission file not found at {config.SUBMISSION_PATH}"

    # Verify submission content
    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print("Submission head:")
    print(df_sub.head())

    required_cols = ["BraTS21ID", "MGMT_value"]
    assert all(
        col in df_sub.columns for col in required_cols
    ), f"Submission missing columns. Found: {df_sub.columns}"

    assert len(df_sub) > 0, "Submission file is empty"

    print("Inference pipeline verified successfully.")


if __name__ == "__main__":
    # Set seed for reproducibility
    seed_everything(42)

    try:
        # Execute verification steps
        test_data_pipeline()
        test_model_architecture()
        test_training_loop()
        test_inference_pipeline()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\nVerification Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # Print traceback for debugging
        import traceback

        traceback.print_exc()
        sys.exit(1)
