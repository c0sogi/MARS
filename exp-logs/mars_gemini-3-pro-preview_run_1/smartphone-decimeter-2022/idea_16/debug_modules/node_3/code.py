import os
import shutil
import numpy as np
import pandas as pd
import torch

# Import from the provided library
from library.config import Config
from library.utils import WGS84Utils, set_seed
from library.data_loader import load_dataset, GNSSSequenceDataset
from library.model import ResUNet1D
from library.train import train_model, generate_submission


def run_demo():
    print("--- Starting Demo ---")

    # 1. Configuration Overrides for Speed and Demo
    print("Configuring for demo run...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Small sample for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead/issues in demo script
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seed
    set_seed(Config.SEED)

    # 2. Verify WGS84Utils Logic
    print("\n--- Verifying WGS84Utils ---")
    # Test point: Googleplex (approx)
    lat, lon, alt = 37.422, -122.084, 10.0

    # Geodetic -> ECEF
    x, y, z = WGS84Utils.geodetic_to_ecef(lat, lon, alt)
    print(f"Geodetic ({lat}, {lon}, {alt}) -> ECEF ({x:.2f}, {y:.2f}, {z:.2f})")

    # ECEF -> Geodetic
    lat_rec, lon_rec, alt_rec = WGS84Utils.ecef_to_geodetic(x, y, z)
    print(f"ECEF -> Geodetic ({lat_rec:.6f}, {lon_rec:.6f}, {alt_rec:.2f})")

    # Check consistency
    assert np.isclose(lat, lat_rec, atol=1e-5), "Latitude reconstruction failed"
    assert np.isclose(lon, lon_rec, atol=1e-5), "Longitude reconstruction failed"
    assert np.isclose(alt, alt_rec, atol=1e-3), "Altitude reconstruction failed"

    # ECEF -> ENU -> ECEF
    # Use the same point as origin for simplicity, so ENU should be (0,0,0)
    e, n, u = WGS84Utils.ecef_to_enu(x, y, z, lat, lon, alt)
    print(f"ECEF -> ENU (Origin=Self) -> ({e:.2f}, {n:.2f}, {u:.2f})")
    assert np.allclose(
        [e, n, u], [0, 0, 0], atol=1e-3
    ), "ENU conversion at origin failed"

    # Offset point
    x2, y2, z2 = x + 100, y + 100, z + 100
    e2, n2, u2 = WGS84Utils.ecef_to_enu(x2, y2, z2, lat, lon, alt)
    x2_rec, y2_rec, z2_rec = WGS84Utils.enu_to_ecef(e2, n2, u2, lat, lon, alt)

    assert np.allclose(
        [x2, y2, z2], [x2_rec, y2_rec, z2_rec], atol=1e-3
    ), "ENU -> ECEF reconstruction failed"
    print("WGS84Utils verification passed.")

    # 3. Demonstrate Data Loading
    print("\n--- Demonstrating Data Loading ---")
    # We force load_cached_data=False to ensure processing logic runs
    train_df = load_dataset("train", load_cached_data=False, debug=True)

    if train_df.empty:
        print("Train DataFrame is empty. Check input data availability.")
    else:
        print(f"Train DataFrame shape: {train_df.shape}")
        print("Columns:", train_df.columns.tolist())

        # Check for features
        required_feats = Config.FEATURE_COLS
        assert all(
            col in train_df.columns for col in required_feats
        ), "Missing feature columns in processed data"

        # Check for targets
        assert "dNorth_meters" in train_df.columns, "Missing target dNorth_meters"
        assert "dEast_meters" in train_df.columns, "Missing target dEast_meters"

    # 4. Demonstrate Dataset Class
    print("\n--- Demonstrating GNSSSequenceDataset ---")
    if not train_df.empty:
        ds = GNSSSequenceDataset(train_df, mode="train")
        print(f"Dataset length (sequences): {len(ds)}")

        item = ds[0]
        features = item["features"]
        targets = item["targets"]
        mask = item["mask"]

        print(f"Feature shape: {features.shape} (C, L)")
        print(f"Target shape: {targets.shape} (C, L)")
        print(f"Mask shape: {mask.shape} (L)")

        assert (
            features.shape[0] == Config.IN_CHANNELS
        ), f"Expected {Config.IN_CHANNELS} channels, got {features.shape[0]}"
        assert (
            features.shape[1] == Config.MAX_SEQUENCE_LENGTH
        ), f"Expected length {Config.MAX_SEQUENCE_LENGTH}, got {features.shape[1]}"
        assert mask.sum() > 0, "Mask should have some valid values"

    # 5. Demonstrate Model
    print("\n--- Demonstrating Model Architecture ---")
    model = ResUNet1D()
    # Create dummy input: Batch=2, Channels=IN_CHANNELS, Length=MAX_SEQUENCE_LENGTH
    dummy_input = torch.randn(2, Config.IN_CHANNELS, Config.MAX_SEQUENCE_LENGTH)

    # Forward pass
    outputs = model(dummy_input)

    print(f"Number of output heads: {len(outputs)}")
    # Check Head 0 (Full Resolution)
    head0 = outputs[0]
    print(f"Head 0 output shape: {head0.shape}")

    assert head0.shape == (
        2,
        Config.OUT_CHANNELS,
        Config.MAX_SEQUENCE_LENGTH,
    ), f"Output shape mismatch. Expected (2, {Config.OUT_CHANNELS}, {Config.MAX_SEQUENCE_LENGTH}), got {head0.shape}"

    # Check Deep Supervision Heads
    if len(outputs) > 1:
        head1 = outputs[1]
        print(f"Head 1 output shape: {head1.shape}")
        expected_len_1 = Config.MAX_SEQUENCE_LENGTH // 2
        assert (
            head1.shape[2] == expected_len_1
        ), f"Head 1 length mismatch. Expected {expected_len_1}, got {head1.shape[2]}"

    # 6. Run Training Loop
    print("\n--- Running Training Loop (Demo) ---")
    # This will use the train_df loaded above (via cache or reprocessing)
    train_model(load_cached_data=True)

    # Verify model file created
    if os.path.exists(Config.MODEL_PATH):
        print(f"Model checkpoint saved at {Config.MODEL_PATH}")
    else:
        raise FileNotFoundError("Model checkpoint was not created during training.")

    # 7. Run Inference / Submission Generation
    print("\n--- Running Inference (Demo) ---")
    generate_submission(load_cached_data=False)

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file created at {Config.SUBMISSION_PATH}")
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission shape: {sub_df.shape}")
        print(sub_df.head())

        # Basic check
        required_cols = [
            "tripId",
            "UnixTimeMillis",
            "LatitudeDegrees",
            "LongitudeDegrees",
        ]
        assert all(
            c in sub_df.columns for c in required_cols
        ), "Submission missing required columns"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n--- Demo Complete ---")


if __name__ == "__main__":
    run_demo()
