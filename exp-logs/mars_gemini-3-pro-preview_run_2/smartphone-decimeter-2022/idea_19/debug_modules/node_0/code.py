import os
import sys
import numpy as np
import pandas as pd
import torch
import shutil

# Import library modules
from library.config import config
from library.utils import (
    ecef_to_lla,
    haversine_distance,
    degrees_to_meters,
    meters_to_degrees,
)
from library.preprocessing import preprocess_dataset
from library.dataset import GNSSWindowDataset, get_dataloaders
from library.model import ECM_MLP
from library.trainer import train_model
from library.inference import generate_submission


def setup_demo_config():
    """
    Override default configuration for a fast demonstration run.
    """
    print("Setting up demo configuration...")

    # Enable debug mode to process only a small subset of trips (5 trips)
    config.DEBUG = True

    # Reduce training duration
    config.EPOCHS = 1
    config.BATCH_SIZE = 1024  # Large batch size for speed on small data

    # Use a specific working directory for this demo to avoid conflicts
    config.WORKING_DIR = "./working/demo_execution"
    config.SUBMISSION_DIR = os.path.join(config.WORKING_DIR, "submission")

    # Ensure directories exist
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Set seeds for reproducibility
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    print(f"Working Directory: {config.WORKING_DIR}")
    print(f"Debug Mode: {config.DEBUG}")


def test_utils():
    """
    Verify the correctness of utility functions.
    """
    print("\n=== Testing Utilities ===")

    # 1. ECEF to LLA
    # Test with a known point: North Pole (approx)
    # WGS84 Semi-major axis a approx 6378137.0
    # At North Pole, x=0, y=0, z approx b (semi-minor) approx 6356752.3
    x, y, z = 0, 0, 6356752.3142
    lat, lon, alt = ecef_to_lla([x], [y], [z])

    print(f"ECEF ({x}, {y}, {z}) -> LLA ({lat[0]:.4f}, {lon[0]:.4f}, {alt[0]:.4f})")

    # Expect Latitude close to 90
    assert np.isclose(lat[0], 90.0, atol=1e-4), f"Expected lat ~90, got {lat[0]}"
    assert np.isclose(lon[0], 0.0, atol=1e-4), f"Expected lon ~0, got {lon[0]}"

    # 2. Haversine Distance
    # Distance between (0, 0) and (1, 0) degrees
    # 1 degree lat is approx 111km
    d = haversine_distance(0, 0, 1, 0)
    print(f"Haversine distance (0,0) to (1,0): {d:.2f} meters")
    assert 110000 < d < 112000, f"Expected dist ~111km, got {d}"

    # 3. Degrees to Meters and Back
    lat_ref = 37.0
    d_lat = 0.001
    d_lon = 0.001

    n_m, e_m = degrees_to_meters(
        np.array([d_lat]), np.array([d_lon]), np.array([lat_ref])
    )
    rec_d_lat, rec_d_lon = meters_to_degrees(n_m, e_m, np.array([lat_ref]))

    print(
        f"Deg->Meters->Deg check: Orig({d_lat}, {d_lon}) -> Rec({rec_d_lat[0]:.6f}, {rec_d_lon[0]:.6f})"
    )
    assert np.isclose(d_lat, rec_d_lat[0]), "Latitude reconstruction failed"
    assert np.isclose(d_lon, rec_d_lon[0]), "Longitude reconstruction failed"

    print("Utilities verified successfully.")


def test_preprocessing_and_dataset():
    """
    Demonstrate data loading, preprocessing, and dataset instantiation.
    """
    print("\n=== Testing Preprocessing & Dataset ===")

    # 1. Preprocess Train Data
    # This will load metadata, process trips (sampled due to DEBUG=True), and save to cache
    print("Preprocessing training data (this may take a moment)...")
    X_train, y_train = preprocess_dataset(
        config.TRAIN_METADATA_PATH, mode="train", load_cached_data=False
    )

    print(f"Processed Train X shape: {X_train.shape}")
    print(f"Processed Train y shape: {y_train.shape}")

    assert len(X_train) == len(y_train), "Mismatch in X and y lengths"
    assert not X_train.empty, "Training data is empty"

    # 2. Instantiate Dataset
    dataset = GNSSWindowDataset(
        X_train, y_train, window_size=config.WINDOW_SIZE, mode="train"
    )

    print(f"Dataset length (valid windows): {len(dataset)}")

    # 3. Check a sample item
    if len(dataset) > 0:
        features, target = dataset[0]
        print(f"Sample feature tensor shape: {features.shape}")
        print(f"Sample target tensor shape: {target.shape}")

        # Expected input dim:
        # Trajectory: Window(15) * 8 features = 120
        # Env Context: 4
        # IMU Context: 4
        # Total = 128
        expected_dim = (
            config.WINDOW_SIZE * len(config.TRAJ_FEATURES)
            + len(config.ENV_FEATURES)
            + len(config.IMU_FEATURES)
        )
        assert (
            features.shape[0] == expected_dim
        ), f"Expected feature dim {expected_dim}, got {features.shape[0]}"
        assert target.shape[0] == 2, f"Expected target dim 2, got {target.shape[0]}"
    else:
        print(
            "Warning: Dataset is empty after windowing logic (trips might be too short for window size)."
        )

    return dataset


def test_model_forward(dataset):
    """
    Demonstrate model instantiation and a forward pass.
    """
    print("\n=== Testing Model Forward Pass ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECM_MLP().to(device)
    print(f"Model initialized on {device}")

    if len(dataset) > 0:
        # Create a small loader
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        features, targets = next(iter(loader))
        features = features.to(device)

        # Forward pass
        outputs = model(features)

        print(f"Batch input shape: {features.shape}")
        print(f"Batch output shape: {outputs.shape}")

        assert outputs.shape == (4, 2), "Output shape mismatch"
        print("Forward pass successful.")
    else:
        print("Skipping forward pass test due to empty dataset.")


def run_training_pipeline():
    """
    Run the full training pipeline (Train -> Val -> Save Model).
    """
    print("\n=== Running Training Pipeline ===")

    # train_model handles dataloading internally via get_dataloaders
    # It will use the cache generated in the previous step if available
    best_loss = train_model()

    print(f"Training finished. Best Validation Loss: {best_loss:.4f}")

    # Verify model file exists
    model_path = os.path.join(config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_path), "Model file was not saved!"
    print(f"Verified model saved at: {model_path}")

    # Verify scaler exists
    scaler_path = os.path.join(config.WORKING_DIR, "scaler.pkl")
    assert os.path.exists(scaler_path), "Scaler file was not saved!"
    print(f"Verified scaler saved at: {scaler_path}")


def run_inference_pipeline():
    """
    Run the inference pipeline to generate a submission file.
    """
    print("\n=== Running Inference Pipeline ===")

    # generate_submission handles test data loading, model loading, and file writing
    generate_submission()

    # Verify submission file
    if os.path.exists(config.SUBMISSION_PATH):
        df_sub = pd.read_csv(config.SUBMISSION_PATH)
        print(f"Submission generated at {config.SUBMISSION_PATH}")
        print(f"Submission shape: {df_sub.shape}")
        print("Head:")
        print(df_sub.head())

        required_cols = [
            "tripId",
            "UnixTimeMillis",
            "LatitudeDegrees",
            "LongitudeDegrees",
        ]
        for col in required_cols:
            assert col in df_sub.columns, f"Missing column {col} in submission"
    else:
        raise FileNotFoundError("Submission file not generated.")


if __name__ == "__main__":
    # 1. Setup
    setup_demo_config()

    # 2. Utils Verification
    test_utils()

    # 3. Data Processing & Dataset Verification
    # We run this explicitly to verify shapes before handing off to the trainer
    train_ds = test_preprocessing_and_dataset()

    # 4. Model Verification
    test_model_forward(train_ds)

    # 5. Full Training Loop
    # Note: This re-loads data using get_dataloaders, utilizing the cache created in step 3
    run_training_pipeline()

    # 6. Inference & Submission
    run_inference_pipeline()

    print("\n=== Demo Completed Successfully ===")
