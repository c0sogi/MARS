import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library files
from library.utils import load_and_process_data, set_seed
from library.model import MSICNN
from library.train import run_training


def verify_data_processing():
    """
    Verifies that data can be loaded, processed, and cached correctly.
    Uses a custom cache directory to avoid interfering with the default training cache.
    """
    print("\n=== Verifying Data Processing ===")

    # Define a temporary directory for this demo
    demo_cache_dir = "./working/demo_verification_cache"

    # Clean up previous runs if they exist
    if os.path.exists(demo_cache_dir):
        shutil.rmtree(demo_cache_dir)

    # Trigger data processing
    # We set load_cached_data=False to force processing from raw files
    print(f"Processing data into {demo_cache_dir}...")
    data = load_and_process_data(load_cached_data=False, base_dir=demo_cache_dir)

    # 1. Verify Keys
    expected_keys = [
        "X_train",
        "y_train",
        "angle_train",
        "ids_train",
        "X_test",
        "angle_test",
        "ids_test",
    ]
    for key in expected_keys:
        assert key in data, f"Missing key in processed data: {key}"

    # 2. Verify Shapes
    # Images should be (N, 3, 75, 75)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")

    assert X_train.ndim == 4, "X_train should be 4-dimensional"
    assert X_train.shape[1] == 3, "X_train should have 3 channels (Band1, Band2, Avg)"
    assert X_train.shape[2:] == (75, 75), "Image spatial dimensions should be 75x75"
    assert len(X_train) == len(y_train), "Mismatch between X_train and y_train length"

    # 3. Verify Content Sanity
    # Check that angles contain NaNs (as per dataset description, before imputation)
    angle_train = data["angle_train"]
    nan_count = np.isnan(angle_train).sum()
    print(f"Number of missing incidence angles in train: {nan_count}")
    assert nan_count > 0, "Expected some NaN values in raw training angles"

    print("Data processing verification passed.")


def verify_model_logic():
    """
    Verifies the MSICNN model architecture, forward pass, and output shapes.
    """
    print("\n=== Verifying Model Logic ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing model on device: {device}")

    model = MSICNN().to(device)

    # Create dummy input batch
    batch_size = 4
    channels = 3
    height = 75
    width = 75

    dummy_images = torch.randn(batch_size, channels, height, width).to(device)
    dummy_angles = torch.randn(batch_size).to(device)

    # 1. Test Training Forward Pass
    # The model uses Multi-Sample Dropout in training, returning 5 predictions per sample
    model.train()
    output_train = model(dummy_images, dummy_angles)

    print(f"Training output shape: {output_train.shape}")
    # Expected: (Batch_Size, 5, 1)
    assert output_train.shape == (
        batch_size,
        5,
        1,
    ), f"Expected training output shape (B, 5, 1), got {output_train.shape}"

    # 2. Test Evaluation Forward Pass
    # In eval mode, dropout is disabled, returning a single prediction per sample
    model.eval()
    with torch.no_grad():
        output_eval = model(dummy_images, dummy_angles)

    print(f"Evaluation output shape: {output_eval.shape}")
    # Expected: (Batch_Size, 1)
    assert output_eval.shape == (
        batch_size,
        1,
    ), f"Expected eval output shape (B, 1), got {output_eval.shape}"

    print("Model logic verification passed.")


def verify_full_pipeline():
    """
    Runs the full training loop for a single epoch to verify integration.
    This uses the library.train.run_training function.
    """
    print("\n=== Verifying Full Training Pipeline ===")

    # Run training with minimal epochs for speed
    # This will use the default cache directory (./working/idea_64) defined in library.utils
    print("Starting short training run (1 epoch per fold)...")

    try:
        run_training(epochs=1, batch_size=32, patience=1, load_cached_data=True)
    except Exception as e:
        raise RuntimeError(f"Training pipeline failed with error: {e}")

    # Verify Submission File
    submission_path = "./submission/submission.csv"
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file loaded. Rows: {len(df_sub)}")

    # Check format
    assert (
        "id" in df_sub.columns and "is_iceberg" in df_sub.columns
    ), "Submission file missing required columns"

    # Check probability range
    probs = df_sub["is_iceberg"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    # Check ID count (should match test set size 321)
    assert len(df_sub) == 321, f"Expected 321 predictions, got {len(df_sub)}"

    print("Full pipeline verification passed.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # Execute verification steps
    verify_data_processing()
    verify_model_logic()
    verify_full_pipeline()

    print("\nAll demonstrations completed successfully.")
