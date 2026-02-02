import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings
import importlib

# Force reload of libraries to ensure changes are picked up in persistent environment
import library.config

importlib.reload(library.config)
import library.utils

importlib.reload(library.utils)
import library.model

importlib.reload(library.model)
import library.data_loader

importlib.reload(library.data_loader)
import library.trainer

importlib.reload(library.trainer)

# Import library components
from library.config import Config
from library.utils import ecef_to_lla, lla_to_enu, enu_to_lla, haversine_distance
from library.data_loader import load_and_preprocess_data
from library.model import LocalShape1DCNN
from library.trainer import train_model, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_config():
    """
    Overrides default Config parameters for a quick demonstration.
    """
    print("[Demo] Setting up configuration...")

    # Use a separate cache directory for the demo to avoid conflicts
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Reduce computational load
    Config.DEBUG = True
    Config.SAMPLE_SIZE = 2  # Use only 2 trips
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 16  # Small batch size

    print(f"  DEBUG: {Config.DEBUG}")
    print(f"  SAMPLE_SIZE: {Config.SAMPLE_SIZE}")
    print(f"  EPOCHS: {Config.EPOCHS}")
    print(f"  CACHE_DIR: {Config.CACHE_DIR}")


def test_utils():
    """
    Verifies utility functions.
    """
    print("\n[Demo] Testing utility functions...")

    # 1. Haversine Distance
    lat1, lon1 = 37.0, -122.0
    lat2, lon2 = 37.1, -122.0
    dist = haversine_distance(lat1, lon1, lat2, lon2)
    # 0.1 degree latitude is roughly 11.1 km
    print(f"  Haversine distance (0.1 deg lat): {dist:.2f} meters")
    assert 11000 < dist < 11200, "Haversine distance calculation seems off"

    # 2. Coordinate Conversions (LLA -> ENU -> LLA)
    ref_lat, ref_lon, ref_alt = 37.4, -122.1, 30.0
    target_lat, target_lon, target_alt = 37.401, -122.099, 35.0

    # Forward
    e, n, u = lla_to_enu(target_lat, target_lon, target_alt, ref_lat, ref_lon, ref_alt)

    # Backward
    lat_rec, lon_rec, alt_rec = enu_to_lla(e, n, u, ref_lat, ref_lon, ref_alt)

    print(f"  Original LLA: {target_lat}, {target_lon}, {target_alt}")
    print(f"  Recovered LLA: {lat_rec:.6f}, {lon_rec:.6f}, {alt_rec:.6f}")

    assert np.isclose(target_lat, lat_rec, atol=1e-6), "Latitude reconstruction failed"
    assert np.isclose(target_lon, lon_rec, atol=1e-6), "Longitude reconstruction failed"
    assert np.isclose(target_alt, alt_rec, atol=1e-3), "Altitude reconstruction failed"
    print("  Coordinate conversion verified.")


def test_data_loader():
    """
    Demonstrates data loading and preprocessing.
    """
    print("\n[Demo] Testing Data Loader...")

    # Load training data (subset)
    # load_cached_data=False forces processing from raw files
    dataset, metadata = load_and_preprocess_data(
        split="train", debug=Config.DEBUG, load_cached_data=False
    )

    print(f"  Dataset size: {len(dataset)}")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty! Check input data availability.")

    # Check item shape
    # Expected shape: (Window_Size, Num_Features) -> (11, 9)
    sample_X, sample_y = dataset[0]
    print(f"  Sample X shape: {sample_X.shape}")
    print(f"  Sample y shape: {sample_y.shape}")

    assert sample_X.shape == (
        Config.WINDOW_SIZE,
        Config.NUM_FEATURES,
    ), f"Expected input shape ({Config.WINDOW_SIZE}, {Config.NUM_FEATURES}), got {sample_X.shape}"
    assert sample_y.shape == (2,), f"Expected target shape (2,), got {sample_y.shape}"

    print("  Data loader verified.")
    return dataset


def test_model(dataset):
    """
    Demonstrates model initialization and forward pass.
    """
    print("\n[Demo] Testing Model...")

    # Instantiate model
    model = LocalShape1DCNN()

    # Create a batch
    batch_size = 4
    batch_X = torch.stack([dataset[i][0] for i in range(batch_size)])

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch_X)

    print(f"  Input batch shape: {batch_X.shape}")
    print(f"  Output batch shape: {output.shape}")

    assert output.shape == (
        batch_size,
        2,
    ), f"Expected output shape ({batch_size}, 2), got {output.shape}"

    print("  Model forward pass verified.")


def run_training_pipeline():
    """
    Runs the training loop and submission generation.
    """
    print("\n[Demo] Running Training Pipeline...")

    # Train model
    # This uses the parameters set in setup_demo_config
    model = train_model(
        debug=Config.DEBUG, epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE
    )

    # Check if model was saved
    model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    if os.path.exists(model_path):
        print(f"  Model saved successfully at {model_path}")
    else:
        raise FileNotFoundError("Model file was not saved.")

    # Generate submission
    print("\n[Demo] Generating Submission...")
    generate_submission(debug=Config.DEBUG)

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"  Submission generated at {submission_path}")
        print(f"  Submission rows: {len(df_sub)}")
        print("  Head:")
        print(df_sub.head())

        # Basic validation of submission format
        required_cols = [
            "tripId",
            "UnixTimeMillis",
            "LatitudeDegrees",
            "LongitudeDegrees",
        ]
        for col in required_cols:
            assert col in df_sub.columns, f"Missing column in submission: {col}"
    else:
        raise FileNotFoundError("Submission file was not generated.")


if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Setup
    setup_demo_config()

    # 2. Verify Utils
    test_utils()

    # 3. Verify Data Loading
    train_dataset = test_data_loader()

    # 4. Verify Model
    test_model(train_dataset)

    # 5. Run Training and Inference
    run_training_pipeline()

    print("\n[Demo] Demonstration completed successfully.")
