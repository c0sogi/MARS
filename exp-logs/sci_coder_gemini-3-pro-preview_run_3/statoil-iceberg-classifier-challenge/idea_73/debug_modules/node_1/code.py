import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, calculate_mad, impute_inc_angles
from library.model import ADSICNN
from library.data_loader import IcebergDataset
from library.trainer import run_training


def verify_utilities():
    """
    Verifies the correctness of utility functions:
    1. calculate_mad (Mean Absolute Deviation)
    2. impute_inc_angles (Median Imputation)
    """
    print("\n--- Verifying Utilities ---")

    # 1. Test MAD calculation
    # Data: [1, 2, 3]. Mean = 2. Deviations: |1-2|=1, |2-2|=0, |3-2|=1. Mean Dev = 2/3 ~= 0.6667
    t = torch.tensor([1.0, 2.0, 3.0])
    mad = calculate_mad(t)
    assert torch.isclose(
        mad, torch.tensor(2.0 / 3.0), atol=1e-4
    ), f"MAD calculation incorrect. Expected ~0.6667, got {mad.item()}"
    print("✓ calculate_mad verified.")

    # 2. Test Angle Imputation
    # Train has NaNs, Test has NaNs. Median of Train (ignoring NaN) should be used.
    train_angles = np.array(
        [10.0, 20.0, np.nan, 30.0]
    )  # Median of [10, 20, 30] is 20.0
    test_angles = np.array([np.nan, 100.0])

    imp_train, imp_test = impute_inc_angles(train_angles, test_angles)

    # Check Train
    expected_train = np.array([10.0, 20.0, 20.0, 30.0])
    assert np.allclose(
        imp_train, expected_train
    ), f"Train imputation failed. Got {imp_train}"

    # Check Test (should use train median = 20.0)
    expected_test = np.array([20.0, 100.0])
    assert np.allclose(
        imp_test, expected_test
    ), f"Test imputation failed. Got {imp_test}"

    print("✓ impute_inc_angles verified.")


def verify_model():
    """
    Verifies the ADSICNN model architecture:
    1. Instantiation
    2. Forward pass with dummy data
    3. Output shape correctness
    """
    print("\n--- Verifying Model Architecture ---")

    device = "cpu"  # Use CPU for simple verification
    model = ADSICNN().to(device)
    model.eval()

    # Create dummy input: Batch Size=4, Channels=3, Height=75, Width=75
    batch_size = 4
    dummy_img = torch.randn(batch_size, 3, 75, 75).to(device)
    dummy_angle = torch.randn(batch_size).to(device)

    with torch.no_grad():
        output = model(dummy_img, dummy_angle)

    # Check output shape: (Batch_Size, 1)
    expected_shape = (batch_size, 1)
    assert (
        output.shape == expected_shape
    ), f"Model output shape incorrect. Expected {expected_shape}, got {output.shape}"

    print(
        f"✓ ADSICNN instantiated and forward pass successful. Output shape: {output.shape}"
    )


def verify_dataset():
    """
    Verifies the IcebergDataset class.
    """
    print("\n--- Verifying Dataset ---")

    # Create dummy numpy arrays
    N = 10
    X = np.random.randn(N, 3, 75, 75).astype(np.float32)
    angles = np.random.randn(N).astype(np.float32)
    y = np.random.randint(0, 2, size=(N)).astype(np.float32)

    ds = IcebergDataset(X, angles, y, transform=True)

    # Check length
    assert len(ds) == N, f"Dataset length mismatch. Expected {N}, got {len(ds)}"

    # Check item retrieval
    img, ang, label = ds[0]
    assert img.shape == (3, 75, 75), f"Image tensor shape incorrect. Got {img.shape}"
    assert isinstance(ang, torch.Tensor), "Angle should be a tensor"
    assert isinstance(label, torch.Tensor), "Label should be a tensor"

    print("✓ IcebergDataset verified.")


def run_demo_pipeline():
    """
    Runs the full training pipeline using library.trainer.run_training.
    Uses a small sample size and 1 epoch to ensure speed.
    """
    print("\n--- Running Demo Training Pipeline ---")

    # Define parameters for a quick run
    # sample_size=30 ensures we only process a tiny fraction of data
    # epochs=1 ensures the training loop finishes instantly
    sample_size = 30
    epochs = 1

    print(f"Configuration: {epochs} Epoch(s), {sample_size} Samples per fold.")

    # Execute training
    # This handles data loading, preprocessing, CV splits, training, and submission generation
    run_training(load_cached_data=False, epochs=epochs, sample_size=sample_size)

    # Verify submission file generation
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if os.path.exists(submission_path):
        df = pd.read_csv(submission_path)
        print(f"✓ Submission file generated at: {submission_path}")
        print(f"  Submission shape: {df.shape}")

        # Check if shape matches sample_size (since run_training slices test data too when sample_size is set)
        assert (
            len(df) == sample_size
        ), f"Submission row count mismatch. Expected {sample_size}, got {len(df)}"

        # Check columns
        expected_cols = ["id", "is_iceberg"]
        assert (
            list(df.columns) == expected_cols
        ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df.columns)}"

        print("✓ Pipeline execution successful.")
    else:
        raise FileNotFoundError(f"Submission file not found at {submission_path}")


if __name__ == "__main__":
    # 1. Setup
    set_seed(42)

    # 2. Verify Components
    verify_utilities()
    verify_model()
    verify_dataset()

    # 3. Run Integration Test
    run_demo_pipeline()
