import os
import shutil
import numpy as np
import pandas as pd
import torch
import unittest.mock
from unittest.mock import MagicMock

# Import library modules
from library.config import Config
from library.utils import set_seed, WGS84, haversine_distance
from library.dataset import GnssSequenceDataset
from library.model import SEResUNet1D
from library.loss import DeepSupervisionLoss
from library.train import train_model
from library.inference import predict_drive


def generate_synthetic_data(num_samples=1000, mode="train"):
    """Generates a synthetic DataFrame matching the preprocessed data schema."""
    np.random.seed(42)

    # Metadata
    drive_ids = ["drive_1"] * num_samples
    phone_names = ["phone_1"] * num_samples
    timestamps = np.arange(num_samples) * 1000 + 1600000000000

    # WLS Baseline (Lat/Lon)
    wls_lat = np.random.uniform(37.4, 37.5, num_samples)
    wls_lon = np.random.uniform(-122.1, -122.0, num_samples)

    # Targets (ENU offsets) - only for train/val
    target_east = np.random.randn(num_samples).astype(np.float32)
    target_north = np.random.randn(num_samples).astype(np.float32)

    # Features
    # 10 Global + 8*3 Panorama = 34 features
    feature_data = np.random.randn(num_samples, 34).astype(np.float32)
    # Ensure positive values for log1p features (index 9 is global_pr_unc_mean)
    feature_data[:, 9] = np.abs(feature_data[:, 9])

    # Feature Column Names
    feature_cols = [
        "global_cn0_mean",
        "global_cn0_std",
        "global_cn0_min",
        "global_cn0_max",
        "global_elev_mean",
        "global_elev_std",
        "global_elev_min",
        "global_elev_max",
        "global_sat_count",
        "global_pr_unc_mean",
    ]
    for i in range(8):  # 8 Azimuth bins
        feature_cols.extend(
            [f"bin_{i}_cn0_max", f"bin_{i}_elev_mean", f"bin_{i}_sat_count"]
        )

    df = pd.DataFrame(feature_data, columns=feature_cols)
    df["drive_id"] = drive_ids
    df["phone_name"] = phone_names
    df["UnixTimeMillis"] = timestamps
    df["wls_lat"] = wls_lat
    df["wls_lon"] = wls_lon

    if mode != "test":
        df["target_east"] = target_east
        df["target_north"] = target_north

    return df


def test_utils():
    print("\n=== Testing Utils ===")

    # 1. Haversine Distance
    # Distance between approx 1 deg lat (111km)
    lat1, lon1 = 0.0, 0.0
    lat2, lon2 = 1.0, 0.0
    dist = haversine_distance(lat1, lon1, lat2, lon2)
    print(f"Haversine distance (1 deg lat): {dist:.2f} meters")
    # Expected approx 111,195 meters
    assert 111000 < dist < 112000, "Haversine distance calculation is incorrect"

    # 2. WGS84 Conversions
    wgs84 = WGS84()
    lat, lon, alt = 37.42, -122.08, 30.0
    x, y, z = wgs84.geodetic_to_ecef(lat, lon, alt)
    lat_back, lon_back, alt_back = wgs84.ecef_to_geodetic(x, y, z)

    print(f"Original LLA: {lat}, {lon}, {alt}")
    print(f"Converted ECEF: {x:.2f}, {y:.2f}, {z:.2f}")
    print(f"Recovered LLA: {lat_back:.5f}, {lon_back:.5f}, {alt_back:.5f}")

    assert np.isclose(lat, lat_back, atol=1e-5), "Latitude conversion failed"
    assert np.isclose(lon, lon_back, atol=1e-5), "Longitude conversion failed"
    assert np.isclose(alt, alt_back, atol=1e-3), "Altitude conversion failed"
    print("Utils tests passed.")


def test_dataset_and_model():
    print("\n=== Testing Dataset and Model ===")

    # Generate data
    df_train = generate_synthetic_data(num_samples=300, mode="train")

    # 1. Dataset
    # Window size 256, stride 128
    dataset = GnssSequenceDataset(df_train, mode="train", window_size=256, stride=128)
    print(f"Dataset length: {len(dataset)}")

    # Get item
    features, targets, meta = dataset[0]
    print(f"Feature shape: {features.shape} (C, T)")
    print(f"Target shape: {targets.shape} (2, T)")

    assert features.shape == (
        34,
        256,
    ), f"Expected feature shape (34, 256), got {features.shape}"
    assert targets.shape == (
        2,
        256,
    ), f"Expected target shape (2, 256), got {targets.shape}"
    assert meta["mask"].shape == (
        256,
    ), f"Expected mask shape (256,), got {meta['mask'].shape}"

    # 2. Model
    model = SEResUNet1D(in_channels=34, out_channels=2)
    # Create batch
    # Cite debug_lesson_16: Increase batch size to 2 to avoid BatchNorm error with 1x1 feature map in ASPP
    batch_size = 2
    batch_features = features.unsqueeze(0).repeat(batch_size, 1, 1)  # (2, 34, 256)

    # Forward pass
    model.train()
    out_final, out_aux1, out_aux2 = model(batch_features)

    print(
        f"Model Output shapes: Final={out_final.shape}, Aux1={out_aux1.shape}, Aux2={out_aux2.shape}"
    )

    # Cite debug_lesson_17: Update assertions to match dynamic batch size
    assert out_final.shape == (batch_size, 2, 256), "Final output shape mismatch"
    assert out_aux1.shape == (batch_size, 2, 128), "Aux1 output shape mismatch"
    assert out_aux2.shape == (batch_size, 2, 64), "Aux2 output shape mismatch"

    # 3. Loss
    criterion = DeepSupervisionLoss()
    # Update targets and mask to match batch size
    batch_targets = targets.unsqueeze(0).repeat(batch_size, 1, 1)  # (2, 2, 256)
    batch_mask = meta["mask"].unsqueeze(0).repeat(batch_size, 1)  # (2, 256)

    loss = criterion([out_final, out_aux1, out_aux2], batch_targets, batch_mask)
    print(f"Calculated Loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"

    print("Dataset and Model tests passed.")


def run_training_pipeline():
    print("\n=== Running Training Pipeline (Mocked Data) ===")

    # Create synthetic dataframes
    train_df = generate_synthetic_data(num_samples=500, mode="train")
    val_df = generate_synthetic_data(num_samples=200, mode="train")  # Val has targets
    test_df = generate_synthetic_data(num_samples=100, mode="test")

    # Patch PreProcessor.process_data to return our synthetic data
    # We patch it in library.train because that's where it's instantiated
    with unittest.mock.patch("library.train.PreProcessor") as MockPreProcessor:
        # Configure the mock instance
        mock_instance = MockPreProcessor.return_value
        mock_instance.process_data.return_value = (train_df, val_df, test_df)

        # Override Config parameters for speed
        original_epochs = Config.EPOCHS
        original_batch_size = Config.BATCH_SIZE
        Config.EPOCHS = 1
        Config.BATCH_SIZE = 4
        Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

        try:
            train_model(
                debug=False
            )  # Debug=False here because we control data size via mock
        finally:
            # Restore Config
            Config.EPOCHS = original_epochs
            Config.BATCH_SIZE = original_batch_size
            Config.NUM_WORKERS = 4


def run_inference_pipeline():
    print("\n=== Running Inference Pipeline (Mocked Data) ===")

    # Create synthetic test data
    # Ensure tripId matches what inference expects (drive_id-phone_name)
    # generate_synthetic_data uses 'drive_1' and 'phone_1'
    test_df = generate_synthetic_data(num_samples=100, mode="test")

    # We need a sample submission file that matches the synthetic data
    # Create a dummy sample submission
    sample_sub_df = pd.DataFrame(
        {
            "tripId": [
                f"{row['drive_id']}-{row['phone_name']}"
                for _, row in test_df.iterrows()
            ],
            "UnixTimeMillis": test_df["UnixTimeMillis"],
            "LatitudeDegrees": 0.0,
            "LongitudeDegrees": 0.0,
        }
    )

    # Save dummy sample submission to where Config expects it, or override Config path
    # Since ./input is read-only, we must override Config.SAMPLE_SUBMISSION_PATH
    dummy_sub_path = os.path.join(Config.WORKING_DIR, "dummy_sample_submission.csv")
    sample_sub_df.to_csv(dummy_sub_path, index=False)

    original_sub_path = Config.SAMPLE_SUBMISSION_PATH
    Config.SAMPLE_SUBMISSION_PATH = dummy_sub_path
    Config.NUM_WORKERS = 0

    # Patch PreProcessor in library.inference
    with unittest.mock.patch("library.inference.PreProcessor") as MockPreProcessor:
        mock_instance = MockPreProcessor.return_value
        # inference.py expects (train, val, test), we only care about test
        mock_instance.process_data.return_value = (None, None, test_df)

        try:
            predict_drive(debug=False)
        except FileNotFoundError as e:
            print(f"Caught expected error (if model wasn't trained): {e}")
        except Exception as e:
            print(f"An error occurred during inference: {e}")
            raise e
        finally:
            Config.SAMPLE_SUBMISSION_PATH = original_sub_path
            Config.NUM_WORKERS = 4


if __name__ == "__main__":
    # Setup working directory
    # Override Config working dir to a temp location
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_DIR = os.path.join(Config.WORKING_DIR, "models")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    set_seed(42)

    # Run Demonstrations
    try:
        test_utils()
        test_dataset_and_model()
        run_training_pipeline()
        run_inference_pipeline()
        print("\nAll demonstrations completed successfully.")
    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        raise e
