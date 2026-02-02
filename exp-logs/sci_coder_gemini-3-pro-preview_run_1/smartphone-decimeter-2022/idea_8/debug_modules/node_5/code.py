import os
import shutil
import sys
import numpy as np
import pandas as pd
import torch

# 1. Patch Configuration for Speed and Demo Purposes
from library.config import Config

# Modify Config for the demo run
Config.DEBUG = True
Config.EPOCHS = 1
Config.BATCH_SIZE = 4
Config.NUM_WORKERS = (
    0  # Use 0 for simple debugging/demo to avoid multiprocessing overhead
)
Config.CACHE_DIR = "./working/demo_cache"
Config.SUBMISSION_DIR = "./working/demo_submission"

# Ensure directories exist
os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

# Clean up stale weights to prevent loading incompatible pre-existing files
if os.path.exists("./working/model_weights.pth"):
    os.remove("./working/model_weights.pth")

# Import Library Modules
from library.utils import (
    llh_to_ecef,
    ecef_to_llh,
    llh_to_enu,
    enu_to_llh,
    haversine_distance,
)
from library.data_loader import (
    get_train_val_loaders,
    get_test_loader,
    GnssSequenceDataset,
)
from library.model import CascadedResUNet
from library.trainer import train_model
from library.inference import generate_submission


def test_utils():
    print("\n--- Testing Utils ---")

    # Test 1: LLH <-> ECEF Round Trip
    lat, lon, alt = 37.42, -122.08, 30.0
    x, y, z = llh_to_ecef(lat, lon, alt)
    lat_out, lon_out, alt_out = ecef_to_llh(x, y, z)

    print(f"Original LLH: {lat}, {lon}, {alt}")
    print(f"Recovered LLH: {lat_out:.6f}, {lon_out:.6f}, {alt_out:.6f}")

    assert np.isclose(lat, lat_out, atol=1e-5), "Latitude mismatch in ECEF conversion"
    assert np.isclose(lon, lon_out, atol=1e-5), "Longitude mismatch in ECEF conversion"
    assert np.isclose(alt, alt_out, atol=1e-3), "Altitude mismatch in ECEF conversion"
    print("LLH <-> ECEF conversion passed.")

    # Test 2: LLH <-> ENU Round Trip
    ref_lat, ref_lon, ref_alt = 37.40, -122.10, 20.0
    e, n, u = llh_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)
    lat_Rec, lon_Rec, alt_Rec = enu_to_llh(e, n, u, ref_lat, ref_lon, ref_alt)

    assert np.isclose(lat, lat_Rec, atol=1e-5), "Latitude mismatch in ENU conversion"
    assert np.isclose(lon, lon_Rec, atol=1e-5), "Longitude mismatch in ENU conversion"
    print("LLH <-> ENU conversion passed.")

    # Test 3: Haversine
    # Distance between (0,0) and (0,1) degree roughly 111km
    d = haversine_distance(0, 0, 0, 1)
    print(f"Haversine distance (0,0) to (0,1): {d:.2f} m")
    assert 111000 < d < 111500, "Haversine distance calculation seems off"
    print("Haversine distance passed.")


def test_data_loader():
    print("\n--- Testing Data Loader ---")

    # Get loaders (this will trigger processing of a few drives due to DEBUG=True)
    # We force load_cached_data=False to ensure the processing logic runs and populates the cache
    train_loader, val_loader = get_train_val_loaders(load_cached_data=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    if len(train_loader) == 0:
        print(
            "Warning: Train loader is empty. This might happen if DEBUG subset is too small or data missing."
        )
        return None

    # Fetch one batch
    features, targets, wls, meta = next(iter(train_loader))

    print(f"Features shape: {features.shape} (Batch, Channels, SeqLen)")
    print(f"Targets shape: {targets.shape} (Batch, 2, SeqLen)")
    print(f"WLS shape: {wls.shape} (Batch, SeqLen, 3)")
    print(f"Meta shape: {meta.shape} (Batch, SeqLen)")

    # Assertions
    assert (
        features.shape[1] == Config.INPUT_CHANNELS
    ), f"Expected {Config.INPUT_CHANNELS} channels, got {features.shape[1]}"
    assert (
        features.shape[2] == Config.SEQUENCE_LENGTH
    ), f"Expected sequence length {Config.SEQUENCE_LENGTH}, got {features.shape[2]}"
    assert targets.shape[1] == 2, "Targets should have 2 channels (East, North)"

    print("Data Loader checks passed.")
    return features  # Return for model testing


def test_model(sample_input):
    print("\n--- Testing Model ---")

    model = CascadedResUNet()
    # Move to CPU for test
    model.to("cpu")

    # Forward pass
    out1, final_out = model(sample_input)

    print(f"Stage 1 Output shape: {out1.shape}")
    print(f"Final Output shape: {final_out.shape}")

    assert out1.shape == (
        sample_input.shape[0],
        2,
        sample_input.shape[2],
    ), "Stage 1 output shape mismatch"
    assert final_out.shape == (
        sample_input.shape[0],
        2,
        sample_input.shape[2],
    ), "Final output shape mismatch"

    print("Model forward pass passed.")


def test_training():
    print("\n--- Testing Training Loop ---")
    # This runs the training loop for 1 epoch on the debug subset
    # It saves weights to ./working/model_weights.pth
    try:
        # We use cached data here since test_data_loader already processed it
        train_model(load_cached_data=True)
        print("Training loop executed successfully.")
    except Exception as e:
        print(f"Training loop failed: {e}")
        raise e

    assert os.path.exists(
        "./working/model_weights.pth"
    ), "Model weights file not created."


def test_inference():
    print("\n--- Testing Inference ---")
    # This generates submission.csv using the trained weights
    try:
        # Force processing of test drives (load_cached_data=False) to verify test pipeline
        generate_submission(load_cached_data=False)
        print("Inference executed successfully.")
    except Exception as e:
        print(f"Inference failed: {e}")
        raise e

    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(sub_path), "Submission file not created."

    df = pd.read_csv(sub_path)
    print(f"Submission rows: {len(df)}")
    print(df.head())

    required_cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    for col in required_cols:
        assert col in df.columns, f"Missing column {col} in submission"


if __name__ == "__main__":
    # Ensure clean state for demo cache
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR)

    # Run Tests
    test_utils()

    # Get sample input from loader test to use in model test
    sample_features = test_data_loader()

    if sample_features is not None:
        test_model(sample_features)
        test_training()
        test_inference()
    else:
        print("Skipping Model/Train/Inference tests due to data loader failure.")
