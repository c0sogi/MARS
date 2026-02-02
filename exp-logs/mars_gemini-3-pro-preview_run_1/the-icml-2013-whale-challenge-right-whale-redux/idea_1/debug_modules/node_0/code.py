import os
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Import library components
from library.config import Config
from library.utils import set_seed, save_submission
from library.dataset import get_datasets, WhaleDataset
from library.model import BiGRUClassifier
from library.trainer import run_training


def demo_utils():
    print("\n=== Demo: Utils ===")

    # 1. Test set_seed
    print("Testing set_seed...")
    set_seed(42)
    r1 = torch.rand(1).item()
    set_seed(42)
    r2 = torch.rand(1).item()
    assert r1 == r2, "set_seed failed: Random numbers are not reproducible."
    print("set_seed passed.")

    # 2. Test save_submission
    print("Testing save_submission...")
    dummy_preds = [0.1, 0.9, 0.3]
    dummy_ids = ["clip_a.aif", "clip_b.aif", "clip_c.aif"]
    temp_sub_path = os.path.join(Config.WORKING_DIR, "test_submission.csv")

    save_submission(dummy_preds, dummy_ids, output_path=temp_sub_path)

    assert os.path.exists(temp_sub_path), "Submission file was not created."
    df = pd.read_csv(temp_sub_path)
    assert list(df.columns) == [
        "clip",
        "probability",
    ], "Submission columns are incorrect."
    assert len(df) == 3, "Submission length is incorrect."
    assert df.iloc[0]["clip"] == "clip_a.aif", "Submission content mismatch."
    print("save_submission passed.")


def demo_dataset():
    print("\n=== Demo: Dataset ===")

    # Use a small limit for speed
    limit = 20
    print(f"Loading datasets with limit={limit}...")

    # Force reload to test processing logic, not just cache loading
    # Note: We modify Config to ensure cache paths don't interfere with main run if needed,
    # but here we just pass load_cached_data=False
    train_ds, val_ds, test_ds = get_datasets(load_cached_data=False, limit=limit)

    # 1. Verify Lengths
    assert (
        len(train_ds) == limit
    ), f"Train dataset length mismatch: {len(train_ds)} vs {limit}"
    assert (
        len(val_ds) == limit
    ), f"Val dataset length mismatch: {len(val_ds)} vs {limit}"
    assert (
        len(test_ds) == limit
    ), f"Test dataset length mismatch: {len(test_ds)} vs {limit}"
    print("Dataset lengths verified.")

    # 2. Verify Data Shapes and Types
    # Get one sample from training
    x, y = train_ds[0]

    print(f"Sample Input Shape: {x.shape}")
    print(f"Sample Target: {y}")

    # Expected shape: (1, N_MELS, Time)
    # Time dimension depends on HOP_LENGTH and TARGET_LENGTH
    # TARGET_LENGTH = 4000, HOP_LENGTH = 20 -> approx 201 frames
    expected_freq = Config.N_MELS

    assert x.dim() == 3, "Input tensor should be 3-dimensional (C, F, T)."
    assert x.shape[0] == 1, "Channel dimension should be 1."
    assert (
        x.shape[1] == expected_freq
    ), f"Frequency dimension mismatch. Expected {expected_freq}, got {x.shape[1]}"
    assert isinstance(y, torch.Tensor), "Target should be a tensor."

    # Get one sample from test
    x_test, clip_id = test_ds[0]
    assert isinstance(clip_id, str), "Test dataset should return clip ID as string."

    print("Dataset shapes and types verified.")
    return x.shape  # Return shape for model test


def demo_model(input_shape):
    print("\n=== Demo: Model ===")

    device = torch.device("cpu")  # Use CPU for simple demo
    model = BiGRUClassifier().to(device)
    model.eval()

    # Create dummy batch
    batch_size = 4
    # input_shape is (1, F, T), we need (Batch, 1, F, T)
    dummy_input = torch.randn(batch_size, *input_shape).to(device)

    print(f"Forward pass with input shape: {dummy_input.shape}")

    with torch.no_grad():
        logits = model(dummy_input)

    print(f"Output shape: {logits.shape}")

    assert logits.dim() == 2, "Output should be 2D (Batch, 1)."
    assert logits.shape[0] == batch_size, "Batch dimension mismatch."
    assert (
        logits.shape[1] == 1
    ), "Output dimension mismatch (should be 1 for binary class)."

    print("Model forward pass verified.")


def demo_training_pipeline():
    print("\n=== Demo: Training Pipeline ===")

    # Modify Config for speed
    print("Adjusting Config for fast execution...")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.PATIENCE = 1

    # Define a temporary cache location to avoid overwriting real training artifacts if they exist
    # However, the library uses hardcoded paths in Config. We will just proceed.
    # Since we use limit=50, the cache files generated will be small and specific to this run
    # if we force cache regeneration.

    limit = 50
    print(f"Starting run_training with limit={limit}...")

    try:
        # load_cached_data=False ensures we process the subset and don't try to load a full dataset cache
        run_training(load_cached_data=False, limit=limit)
    except Exception as e:
        raise AssertionError(f"Training pipeline crashed: {e}")

    # Verify output
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    df = pd.read_csv(submission_path)
    assert len(df) == limit, f"Submission should have {limit} rows, found {len(df)}"
    print("Training pipeline completed and submission generated successfully.")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print("Starting Library Demonstration...")

    try:
        # 1. Utils
        demo_utils()

        # 2. Dataset
        # We capture the shape from the dataset to ensure the model test uses correct dimensions
        sample_shape = demo_dataset()

        # 3. Model
        demo_model(sample_shape)

        # 4. Full Training Loop
        demo_training_pipeline()

        print("\nAll demonstrations passed successfully!")

    except AssertionError as e:
        print(f"\nVerification Failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
