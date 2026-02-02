import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import library modules
import library.config as config
import library.utils as utils
import library.data_loader as data_loader
import library.preprocessing as preprocessing
import library.model as model_lib
import library.trainer as trainer


def main():
    print("==================================================")
    print("LIBRARY USAGE DEMONSTRATION")
    print("==================================================")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed
    # ---------------------------------------------------------
    print("\n[1] Overriding Configuration for Demo Speed...")

    # We need to patch the variables in the modules where they were imported
    # to ensure the changes take effect.

    # Config module itself
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 3  # Use only 3 trips
    config.NUM_EPOCHS = 1
    config.BATCH_SIZE = 32
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Patch data_loader module
    data_loader.DEBUG = True
    data_loader.DEBUG_SAMPLE_SIZE = 3

    # Patch trainer module
    trainer.NUM_EPOCHS = 1
    trainer.BATCH_SIZE = 32
    trainer.NUM_WORKERS = 0

    # Patch preprocessing module (uses window size)
    # We keep window size as is, but ensure it's consistent
    print(f"Debug Mode: {config.DEBUG}")
    print(f"Sample Size: {config.DEBUG_SAMPLE_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")

    # ---------------------------------------------------------
    # 2. Utils Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test ECEF to LLA
    # Approximate ECEF for (Lat: 37.4, Lon: -122.1, Alt: 0) - Googleplex area
    # Calculated via external tool or approximation logic
    x, y, z = -2694000.0, -4297000.0, 3854000.0
    lat, lon, alt = utils.ecef_to_lla(x, y, z)
    print(f"ECEF to LLA: ({x}, {y}, {z}) -> ({lat:.4f}, {lon:.4f}, {alt:.4f})")

    # Basic sanity check: Lat should be around 37, Lon around -122
    assert 30 < lat < 45, "Latitude calculation seems off"
    assert -130 < lon < -110, "Longitude calculation seems off"

    # Test Haversine
    d = utils.haversine_distance(37.0, -122.0, 37.001, -122.0)
    print(f"Haversine Distance (0.001 deg lat change): {d:.4f} meters")
    # 1 degree lat is approx 111km -> 0.001 deg is approx 111m
    assert 100 < d < 120, "Haversine distance calculation seems off"

    # Test Degrees/Meters Conversion
    d_lat_m, d_lon_m = utils.degrees_to_meters(0.001, 0.001, 37.0)
    print(
        f"Deg to Meters (0.001, 0.001 at 37N): Lat={d_lat_m:.2f}m, Lon={d_lon_m:.2f}m"
    )

    d_lat_deg, d_lon_deg = utils.meters_to_degrees(d_lat_m, d_lon_m, 37.0)
    print(f"Meters to Deg (Round Trip): Lat={d_lat_deg:.6f}, Lon={d_lon_deg:.6f}")

    assert np.isclose(d_lat_deg, 0.001), "Lat round trip conversion failed"
    assert np.isclose(d_lon_deg, 0.001), "Lon round trip conversion failed"
    print("Utils verification passed.")

    # ---------------------------------------------------------
    # 3. Data Loading
    # ---------------------------------------------------------
    print("\n[3] Loading Data (Debug Mode)...")

    # We force load_cached_data=False to demonstrate processing logic
    # and ensure we use the sampled subset
    train_df = data_loader.get_dataset("train", load_cached_data=False)

    print(f"Loaded Train DataFrame Shape: {train_df.shape}")
    print(f"Columns: {list(train_df.columns[:5])} ...")

    assert not train_df.empty, "Train dataframe is empty!"
    assert "wls_lat" in train_df.columns, "wls_lat column missing"
    assert "res_lat_m" in train_df.columns, "Target column res_lat_m missing"

    # ---------------------------------------------------------
    # 4. Preprocessing
    # ---------------------------------------------------------
    print("\n[4] Preprocessing and Dataset Creation...")

    scaler = preprocessing.GNSSScaler()
    scaler.fit(train_df)
    print("Scaler fitted.")

    # Create Dataset
    dataset = preprocessing.GNSSSequenceDataset(
        train_df, scaler, window_size=config.WINDOW_SIZE, is_test=False
    )

    print(f"Dataset Length: {len(dataset)}")

    if len(dataset) > 0:
        traj, ctx, target = dataset[0]
        print(f"Sample Tensor Shapes:")
        print(
            f"  Trajectory: {traj.shape} (Expected: [{config.TRAJECTORY_INPUT_DIM}, {config.WINDOW_SIZE}])"
        )
        print(f"  Context:    {ctx.shape} (Expected: [{config.CONTEXT_INPUT_DIM}])")
        print(f"  Target:     {target.shape} (Expected: [{config.OUTPUT_DIM}])")

        assert traj.shape == (
            config.TRAJECTORY_INPUT_DIM,
            config.WINDOW_SIZE,
        ), "Trajectory shape mismatch"
        assert ctx.shape == (config.CONTEXT_INPUT_DIM,), "Context shape mismatch"
        assert target.shape == (config.OUTPUT_DIM,), "Target shape mismatch"
    else:
        print(
            "Warning: Dataset is empty (might happen if trips are shorter than window size)"
        )

    # ---------------------------------------------------------
    # 5. Model Initialization
    # ---------------------------------------------------------
    print("\n[5] Model Initialization...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_lib.GeometryConditionedCNN().to(device)
    print(model)

    # Forward pass check
    if len(dataset) > 0:
        dummy_traj = traj.unsqueeze(0).to(device)  # Batch size 1
        dummy_ctx = ctx.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(dummy_traj, dummy_ctx)

        print(f"Model Output Shape: {output.shape}")
        assert output.shape == (1, config.OUTPUT_DIM), "Model output shape mismatch"

    # ---------------------------------------------------------
    # 6. Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n[6] Running Training Loop...")

    # Force reprocessing to ensure debug sampling is applied
    # Note: In a real run, we might want to clean up the cache file first to be sure
    cache_file = os.path.join(config.WORKING_DIR, "train_data.parquet")
    if os.path.exists(cache_file):
        os.remove(cache_file)

    # Run training
    trained_model, fitted_scaler = trainer.train_model(load_cached_data=False)

    print("Training complete.")
    assert os.path.exists(
        os.path.join(config.WORKING_DIR, "best_model.pth")
    ), "Model checkpoint not found"
    assert os.path.exists(
        os.path.join(config.WORKING_DIR, "scaler.json")
    ), "Scaler file not found"

    # ---------------------------------------------------------
    # 7. Inference Demonstration
    # ---------------------------------------------------------
    print("\n[7] Running Inference...")

    # Clean test cache to ensure debug sampling
    test_cache = os.path.join(config.WORKING_DIR, "test_data.parquet")
    if os.path.exists(test_cache):
        os.remove(test_cache)

    trainer.predict_and_submit(trained_model, fitted_scaler, load_cached_data=False)

    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    if os.path.exists(submission_path):
        sub_df = pd.read_csv(submission_path)
        print(f"Submission generated at {submission_path}")
        print(sub_df.head())

        # Verify submission format
        required_cols = [
            "tripId",
            "UnixTimeMillis",
            "LatitudeDegrees",
            "LongitudeDegrees",
        ]
        assert all(
            col in sub_df.columns for col in required_cols
        ), "Submission missing required columns"
        assert not sub_df.empty, "Submission dataframe is empty"
    else:
        print("Error: Submission file was not created.")

    print("\n==================================================")
    print("DEMONSTRATION COMPLETE")
    print("==================================================")


if __name__ == "__main__":
    main()
