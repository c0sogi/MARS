import os
import pandas as pd
import numpy as np
import torch
import shutil

# Import from the provided library files
from library.utils import ecef_to_lla, haversine_distance
from library.model import TemporalConvNet
from library.train import run_training
from library.inference import generate_predictions
from library.data_loader import GnssWindowedDataset


def setup_subset_metadata(input_meta_dir, output_meta_dir, sample_size=200):
    """
    Creates a small subset of metadata files to speed up the demonstration.
    """
    if os.path.exists(output_meta_dir):
        shutil.rmtree(output_meta_dir)
    os.makedirs(output_meta_dir)

    files = ["train_metadata.csv", "val_metadata.csv", "test_metadata.csv"]

    print(
        f"Creating subset metadata in {output_meta_dir} with {sample_size} samples each..."
    )

    for f in files:
        src = os.path.join(input_meta_dir, f)
        dst = os.path.join(output_meta_dir, f)

        if os.path.exists(src):
            df = pd.read_csv(src)
            # Take a subset. For train/val, we want consecutive rows to form valid windows if possible,
            # but for a quick smoke test, head() is sufficient as the Dataset class handles padding.
            # We filter for a specific drive to ensure data consistency.
            if "drive_id" in df.columns:
                drive_ids = df["drive_id"].unique()
                if len(drive_ids) > 0:
                    target_drive = drive_ids[0]
                    df_subset = df[df["drive_id"] == target_drive].head(sample_size)
                else:
                    df_subset = df.head(sample_size)
            else:
                df_subset = df.head(sample_size)

            df_subset.to_csv(dst, index=False)
            print(f"  Created {f}: {len(df_subset)} rows")
        else:
            raise FileNotFoundError(f"Source metadata file {src} not found.")


def test_utils():
    print("\n--- Testing Utils ---")
    # Test ECEF to LLA
    # Approximate ECEF for (Lat=37.4, Lon=-122.1, Alt=0)
    x, y, z = -2694322.0, -4297206.0, 3854303.0
    lat, lon, alt = ecef_to_lla(x, y, z)

    print(f"ECEF: ({x}, {y}, {z}) -> LLA: ({lat:.4f}, {lon:.4f}, {alt:.4f})")

    # Basic sanity check ranges
    assert 37.0 < lat < 38.0, f"Latitude {lat} out of expected range"
    assert -123.0 < lon < -121.0, f"Longitude {lon} out of expected range"

    # Test Haversine
    # Distance between approx San Francisco and New York
    lat1, lon1 = 37.7749, -122.4194
    lat2, lon2 = 40.7128, -74.0060
    dist = haversine_distance(lat1, lon1, lat2, lon2)
    print(f"Distance SF->NY: {dist/1000:.2f} km")

    # Approx 4130 km
    assert 4100000 < dist < 4200000, f"Distance {dist} seems incorrect"
    print("Utils verification passed.")


def test_model_architecture():
    print("\n--- Testing Model Architecture ---")
    batch_size = 4
    window_size = 32
    input_channels = 5
    hidden_dim = 64
    output_dim = 2

    model = TemporalConvNet(
        input_channels=input_channels,
        window_size=window_size,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )

    # Create dummy input: (Batch, Window, Channels)
    dummy_input = torch.randn(batch_size, window_size, input_channels)

    # Forward pass
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    assert output.shape == (
        batch_size,
        output_dim,
    ), f"Expected output shape {(batch_size, output_dim)}, got {output.shape}"
    print("Model architecture verification passed.")


def run_pipeline_demonstration():
    print("\n--- Running Training & Inference Pipeline ---")

    # Configuration optimized for speed
    config = {
        "window_size": 32,
        "batch_size": 16,
        "lr": 1e-3,
        "epochs": 2,  # Minimal epochs for demonstration
        "patience": 2,
        "hidden_dim": 64,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "seed": 42,
    }

    input_dir = "./input"
    base_metadata_dir = "./metadata"
    working_dir = "./working"
    subset_metadata_dir = os.path.join(working_dir, "metadata_subset")
    submission_dir = os.path.join(working_dir, "submission")

    # 1. Create subset metadata for speed
    setup_subset_metadata(base_metadata_dir, subset_metadata_dir, sample_size=100)

    # 2. Run Training
    print("\nInvoking library.train.run_training...")
    history = run_training(
        input_dir=input_dir,
        metadata_dir=subset_metadata_dir,
        working_dir=working_dir,
        submission_dir=submission_dir,
        config=config,
    )

    # Validation of Training Outputs
    model_path = os.path.join(working_dir, "model_weights.pth")
    assert os.path.exists(model_path), "Model weights file was not created."
    print(f"Verified: Model weights found at {model_path}")

    submission_path = os.path.join(submission_dir, "submission.csv")
    assert os.path.exists(
        submission_path
    ), "Submission file was not created during training step."
    print(f"Verified: Submission file found at {submission_path}")

    # 3. Run Inference (Explicitly)
    # Although run_training does generation, we demonstrate the separate inference function here
    print("\nInvoking library.inference.generate_predictions...")

    # We use a different output directory to verify independent execution
    inference_output_dir = os.path.join(working_dir, "inference_output")

    generate_predictions(
        input_dir=input_dir,
        metadata_dir=subset_metadata_dir,
        working_dir=working_dir,
        submission_dir=inference_output_dir,
        config=config,
    )

    inf_submission_path = os.path.join(inference_output_dir, "submission.csv")
    assert os.path.exists(inf_submission_path), "Inference submission file not created."

    # Check content of submission
    df_sub = pd.read_csv(inf_submission_path)
    print(f"Inference submission shape: {df_sub.shape}")
    expected_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    assert all(
        col in df_sub.columns for col in expected_cols
    ), "Submission missing required columns."

    # Check for NaNs (should be none if model works correctly, though input NaNs might propagate if not handled)
    # The data_loader handles NaNs via interpolation and filling.
    nan_count = df_sub[["LatitudeDegrees", "LongitudeDegrees"]].isna().sum().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaNs found in predictions.")
    else:
        print("Verified: No NaNs in predictions.")

    print("Pipeline demonstration completed successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Test Utility Functions
    test_utils()

    # 2. Test Model Definition
    test_model_architecture()

    # 3. Run Full Pipeline (Data Load -> Train -> Inference)
    run_pipeline_demonstration()
