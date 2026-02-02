import os
import sys
import pandas as pd
import numpy as np
import torch

# Add current directory to path to ensure library imports work
sys.path.append(".")

from library.config import Config
from library.utils import (
    set_seed,
    azimuth_zenith_to_vector,
    vector_to_azimuth_zenith,
    load_sensor_geometry,
)
from library.dataset import IceCubeDataset
from library.model import SpatiotemporalPointTransformer
from library.train import run_training


def main():
    print("=== IceCube Direction Prediction Demo ===")

    # ==========================================
    # 1. Configuration for Fast Demo
    # ==========================================
    print("\n[1] Configuring environment for rapid execution...")

    # Reduce dataset sizes and training duration for demonstration purposes
    Config.TRAIN_SUBSET_SIZE = 1000  # Train on 1000 events
    Config.VAL_SUBSET_SIZE = 200  # Validate on 200 events
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_WORKERS = 2  # Reduce workers to minimize overhead

    # The full test set has 13M events. To keep this demo fast, we create a
    # temporary test metadata file containing only 50 events.
    original_test_meta = pd.read_parquet(Config.TEST_META)
    small_test_meta = original_test_meta.head(50).copy()

    temp_test_meta_path = os.path.join(Config.WORKING_DIR, "temp_test_meta.parquet")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    small_test_meta.to_parquet(temp_test_meta_path)

    # Point Config to this temporary file so the library uses it
    Config.TEST_META = temp_test_meta_path
    print(
        f"    Created temporary test metadata with {len(small_test_meta)} events at {temp_test_meta_path}"
    )

    # Set global seed for reproducibility
    set_seed(42)

    # ==========================================
    # 2. Verify Utility Functions
    # ==========================================
    print("\n[2] Verifying Utility Functions...")

    # 2.1 Geometry Loading
    geo_df = load_sensor_geometry()
    print(f"    Sensor geometry shape: {geo_df.shape}")
    if len(geo_df) != 5160:
        raise AssertionError(f"Expected 5160 sensors, found {len(geo_df)}")

    # 2.2 Coordinate Transforms
    # Test vector (0, 0, 1) -> Zenith 0 (Up)
    # Test vector (-1, 0, 0) -> Zenith pi/2, Azimuth pi (Horizontal)
    az_in = torch.tensor([0.0, np.pi])
    zen_in = torch.tensor([0.0, np.pi / 2])

    vec_out = azimuth_zenith_to_vector(az_in, zen_in)

    expected_1 = torch.tensor([0.0, 0.0, 1.0])
    expected_2 = torch.tensor([-1.0, 0.0, 0.0])

    if not torch.allclose(vec_out[0], expected_1, atol=1e-5):
        raise AssertionError(f"Transform failed for Up vector. Got {vec_out[0]}")
    if not torch.allclose(vec_out[1], expected_2, atol=1e-5):
        raise AssertionError(
            f"Transform failed for Horizontal vector. Got {vec_out[1]}"
        )

    # Test Inverse Transform
    az_rec, zen_rec = vector_to_azimuth_zenith(vec_out)

    # Zenith for (0,0,1) is 0.
    if not torch.allclose(zen_rec[0], torch.tensor(0.0), atol=1e-5):
        raise AssertionError("Inverse zenith failed.")

    print("    Coordinate transformation logic verified.")

    # ==========================================
    # 3. Verify Dataset Loading
    # ==========================================
    print("\n[3] Verifying Dataset Class...")

    # Initialize Train Dataset (triggers cache generation for the subset)
    train_dataset = IceCubeDataset(mode="train", subset_size=Config.TRAIN_SUBSET_SIZE)
    print(f"    Train dataset initialized with {len(train_dataset)} events.")

    if len(train_dataset) == 0:
        raise AssertionError("Dataset is empty.")

    # Check __getitem__ structure
    sample_x, sample_y = train_dataset[0]
    print(f"    Sample Feature Shape: {sample_x.shape}")  # Expected: (MAX_PULSES, 6)
    print(f"    Sample Target Shape: {sample_y.shape}")  # Expected: (2,)

    if sample_x.shape != (Config.MAX_PULSES, 6):
        raise AssertionError(f"Incorrect feature shape: {sample_x.shape}")
    if sample_y.shape != (2,):
        raise AssertionError(f"Incorrect target shape: {sample_y.shape}")

    # Check Data Integrity: Charge (index 4) should be non-negative (log1p)
    if (sample_x[:, 4] < 0).any():
        raise AssertionError("Found negative charge values in processed features.")

    print("    Dataset structure and loading verified.")

    # ==========================================
    # 4. Verify Model Architecture
    # ==========================================
    print("\n[4] Verifying Model Architecture...")

    model = SpatiotemporalPointTransformer()
    model.eval()

    # Create dummy batch (Batch Size=4, Sequence Length=MAX_PULSES, Channels=6)
    batch_size = 4
    dummy_input = torch.randn(batch_size, Config.MAX_PULSES, 6)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Model output shape: {output.shape}")

    if output.shape != (batch_size, 3):
        raise AssertionError(
            f"Expected output shape ({batch_size}, 3), got {output.shape}"
        )

    print("    Model forward pass verified.")

    # ==========================================
    # 5. Execute Training & Inference Pipeline
    # ==========================================
    print("\n[5] Executing Training Pipeline (Train -> Val -> Submission)...")

    # run_training uses the Config parameters we modified earlier.
    # It trains for 1 epoch, saves the best model, and generates a submission
    # using the temporary test metadata we created.
    run_training()

    # ==========================================
    # 6. Verify Submission Output
    # ==========================================
    print("\n[6] Verifying Submission File...")

    submission_path = Config.SUBMISSION_PATH
    if not os.path.exists(submission_path):
        raise AssertionError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"    Submission loaded. Shape: {df_sub.shape}")

    # Check length (should be 50 based on our temp metadata)
    if len(df_sub) != 50:
        raise AssertionError(f"Expected 50 predictions, found {len(df_sub)}")

    # Check columns
    required_cols = ["event_id", "azimuth", "zenith"]
    for col in required_cols:
        if col not in df_sub.columns:
            raise AssertionError(f"Missing column: {col}")

    # Check value ranges
    if df_sub["azimuth"].min() < 0 or df_sub["azimuth"].max() > 2 * np.pi:
        raise AssertionError("Azimuth values out of range [0, 2pi]")
    if df_sub["zenith"].min() < 0 or df_sub["zenith"].max() > np.pi:
        raise AssertionError("Zenith values out of range [0, pi]")

    print("    Submission file content verified.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
