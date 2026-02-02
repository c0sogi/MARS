import sys
import os
import numpy as np
import pandas as pd
import torch

# Ensure the current directory is in the python path to allow imports from library
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import (
    lla_to_ecef,
    ecef_to_lla,
    ecef_to_enu,
    enu_to_ecef,
    haversine_distance,
)
from library.model import RelativeTrajectoryCNN
from library.trainer import run_training, generate_submission
from library.data_processor import load_data, get_dataset
from library.dataset import get_dataloader


def test_utils():
    """
    Verify correctness of coordinate conversion utilities.
    """
    print("\n[1/5] Testing Coordinate Utilities...")

    # Test 1: LLA -> ECEF -> LLA Roundtrip
    lat, lon, alt = 37.4219999, -122.0840575, 30.0
    x, y, z = lla_to_ecef(lat, lon, alt)
    lat_rec, lon_rec, alt_rec = ecef_to_lla(x, y, z)

    assert np.isclose(lat, lat_rec, atol=1e-5), f"Lat mismatch: {lat} vs {lat_rec}"
    assert np.isclose(lon, lon_rec, atol=1e-5), f"Lon mismatch: {lon} vs {lon_rec}"
    assert np.isclose(alt, alt_rec, atol=1e-3), f"Alt mismatch: {alt} vs {alt_rec}"
    print("  - LLA <-> ECEF roundtrip passed.")

    # Test 2: ECEF -> ENU -> ECEF Roundtrip
    # Reference point (e.g., WLS position)
    ref_lat, ref_lon, ref_alt = 37.422, -122.084, 0.0
    # Target point (e.g., GT position)
    tgt_lat, tgt_lon, tgt_alt = 37.4221, -122.0841, 10.0
    tgt_x, tgt_y, tgt_z = lla_to_ecef(tgt_lat, tgt_lon, tgt_alt)

    e, n, u = ecef_to_enu(tgt_x, tgt_y, tgt_z, ref_lat, ref_lon, ref_alt)
    rec_x, rec_y, rec_z = enu_to_ecef(e, n, u, ref_lat, ref_lon, ref_alt)

    assert np.allclose(
        [tgt_x, tgt_y, tgt_z], [rec_x, rec_y, rec_z], atol=1e-3
    ), "ENU roundtrip failed"
    print("  - ECEF <-> ENU roundtrip passed.")

    # Test 3: Haversine Distance
    # Distance between two points approx 111km apart (1 degree lat)
    d = haversine_distance(0, 0, 1, 0)
    # 1 degree latitude is approximately 111km
    assert 111000 < d < 111400, f"Haversine calculation suspicious: {d}"
    print("  - Haversine distance passed.")


def test_data_loading_and_processing():
    """
    Verify data loading, feature engineering, and windowing pipeline.
    """
    print("\n[2/5] Testing Data Pipeline...")

    # Configure for speed
    Config.DEBUG_SAMPLE_SIZE = 100
    Config.WINDOW_SIZE = 11

    # Load training data
    # load_cached_data=False forces processing from scratch to verify logic
    print("  - Loading and processing training data (debug_size=100)...")
    X, y, meta = load_data(
        mode="train", load_cached_data=False, debug_size=Config.DEBUG_SAMPLE_SIZE
    )

    # Verify shapes
    print(f"    X shape: {X.shape}")
    print(f"    y shape: {y.shape}")

    assert (
        len(X) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} samples, got {len(X)}"
    assert X.ndim == 3, "X should be 3D (N, Window, Features)"
    assert y.ndim == 2, "y should be 2D (N, OutputDim)"
    assert (
        X.shape[1] == Config.WINDOW_SIZE
    ), f"Window size mismatch: {X.shape[1]} vs {Config.WINDOW_SIZE}"
    assert (
        X.shape[2] == Config.NUM_INPUT_FEATURES
    ), f"Feature count mismatch: {X.shape[2]} vs {Config.NUM_INPUT_FEATURES}"

    # Verify metadata alignment
    assert len(meta) == len(X), "Metadata length mismatch"
    assert "tripId" in meta.columns, "Metadata missing tripId"

    # Test DataLoader creation
    print("  - Creating DataLoader...")
    loader = get_dataloader(X, y, batch_size=16, shuffle=False)
    batch_X, batch_y = next(iter(loader))

    assert batch_X.shape == (
        16,
        Config.WINDOW_SIZE,
        Config.NUM_INPUT_FEATURES,
    ), "Batch X shape mismatch"
    assert batch_y.shape == (16, 2), "Batch y shape mismatch"
    print("  - Data pipeline verification passed.")


def test_model_architecture():
    """
    Verify model instantiation and forward pass.
    """
    print("\n[3/5] Testing Model Architecture...")

    model = RelativeTrajectoryCNN(
        input_channels=Config.NUM_INPUT_FEATURES,
        hidden_channels=Config.CNN_HIDDEN_CHANNELS,
        kernel_size=Config.CNN_KERNEL_SIZE,
        fc_dim=Config.FC_HIDDEN_DIM,
        dropout=Config.DROPOUT_RATE,
    )

    # Move to configured device
    model.to(Config.DEVICE)
    model.eval()

    # Create dummy input
    batch_size = 4
    dummy_input = torch.randn(
        batch_size, Config.WINDOW_SIZE, Config.NUM_INPUT_FEATURES
    ).to(Config.DEVICE)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"  - Input shape: {dummy_input.shape}")
    print(f"  - Output shape: {output.shape}")

    assert output.shape == (
        batch_size,
        2,
    ), "Model output shape mismatch (expected [Batch, 2])"
    print("  - Model architecture verification passed.")


def test_training_loop():
    """
    Run a short training loop to verify the trainer.
    """
    print("\n[4/5] Testing Training Loop...")

    # Configure for a very quick run
    Config.DEBUG_SAMPLE_SIZE = 200
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 32

    # Run training
    # This will process data (since size changed/cache invalid for this size), train, and save model
    run_training(
        debug_size=Config.DEBUG_SAMPLE_SIZE,
        epochs=Config.NUM_EPOCHS,
        load_cached_data=False,
    )

    # Check if model artifact exists
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file was not created at {Config.MODEL_PATH}")

    print("  - Training loop completed and model saved.")


def test_inference_and_submission():
    """
    Run inference using the trained model and generate submission file.
    """
    print("\n[5/5] Testing Inference and Submission...")

    # Ensure model exists (should be created by previous step)
    assert os.path.exists(Config.MODEL_PATH), "Model path does not exist for inference"

    # Generate submission for a subset of test data
    debug_test_size = 50
    generate_submission(debug_size=debug_test_size, load_cached_data=False)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  - Submission shape: {df_sub.shape}")

    assert (
        len(df_sub) == debug_test_size
    ), f"Submission size mismatch: {len(df_sub)} vs {debug_test_size}"
    required_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    for col in required_cols:
        assert col in df_sub.columns, f"Missing column in submission: {col}"

    # Basic sanity check on coordinates (valid range)
    lat_valid = df_sub["LatitudeDegrees"].between(-90, 90).all()
    lon_valid = df_sub["LongitudeDegrees"].between(-180, 180).all()

    if not lat_valid or not lon_valid:
        raise ValueError(
            "Submission contains invalid coordinates out of Lat/Lon bounds."
        )

    print("  - Inference and submission verification passed.")


if __name__ == "__main__":
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)

    print("Starting End-to-End Demonstration...")

    try:
        test_utils()
        test_data_loading_and_processing()
        test_model_architecture()
        test_training_loop()
        test_inference_and_submission()
        print("\nSUCCESS: All pipeline components verified.")
    except Exception as e:
        print(f"\nFAILURE: An error occurred during demonstration: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
