import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.utils import cartesian_to_wgs84, wgs84_to_cartesian, haversine_distance
from library.feature_engineering import process_dataset
from library.dataset import GNSSSequenceDataset, gnss_collate_fn
from library.model import StratifiedResUNet1D
from library.trainer import train_model


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Configuration Setup
    # Override Config for a fast demonstration run
    print("\n[1] Configuring Environment...")
    Config.DEBUG = True
    Config.DEBUG_DRIVE_COUNT = 2  # Process only 2 drives
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 2  # Small batch size
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "demo_cache")  # Separate cache

    # Ensure directories exist
    Config.setup()
    set_seed(Config.SEED)

    # 2. Verify Utility Functions
    print("\n[2] Verifying Utility Functions...")
    # Test coordinate conversion logic
    ref_lat, ref_lon = 37.4, -122.1
    # Move approx 111km North (1 degree lat) and 88km East (1 degree lon at this lat)
    # WGS84 approx: 1 deg lat ~ 111km, 1 deg lon ~ 111 * cos(lat) km
    test_lat, test_lon = 38.4, -121.1

    # Forward: WGS84 -> Cartesian
    east, north = wgs84_to_cartesian(test_lat, test_lon, ref_lat, ref_lon)

    # Inverse: Cartesian -> WGS84
    rec_lat, rec_lon = cartesian_to_wgs84(east, north, ref_lat, ref_lon)

    # Check reconstruction accuracy
    assert np.isclose(
        test_lat, rec_lat, atol=1e-5
    ), f"Lat reconstruction failed: {test_lat} vs {rec_lat}"
    assert np.isclose(
        test_lon, rec_lon, atol=1e-5
    ), f"Lon reconstruction failed: {test_lon} vs {rec_lon}"

    # Check distance calculation
    dist = haversine_distance(ref_lat, ref_lon, test_lat, test_lon)
    calc_dist = np.sqrt(east**2 + north**2)
    # Haversine (Great Circle) vs Cartesian (Flat Earth approx) will differ slightly over large distances,
    # but should be reasonably correlated.
    print(f"  Reference Point: ({ref_lat}, {ref_lon})")
    print(f"  Target Point:    ({test_lat}, {test_lon})")
    print(f"  Cartesian Offset: E={east:.2f}m, N={north:.2f}m")
    print(f"  Haversine Dist: {dist:.2f}m, Cartesian Dist: {calc_dist:.2f}m")
    print("  Utility functions verified.")

    # 3. Feature Engineering & Data Processing
    print("\n[3] Processing Data (Feature Engineering)...")
    # We force reprocessing to demonstrate the pipeline logic, ignoring any existing cache
    # This uses the metadata files located in ./metadata

    # Process Train Split
    try:
        train_df = process_dataset(
            Config.TRAIN_METADATA_PATH, load_cached_data=False, split_name="train"
        )
        print(f"  Processed Train DataFrame Shape: {train_df.shape}")

        # Basic validation of processed data
        expected_cols = [
            "UnixTimeMillis",
            "WlsLat",
            "WlsLon",
            "drive_id",
            "phone_name",
            "dEast",
            "dNorth",
        ]
        for col in expected_cols:
            assert (
                col in train_df.columns
            ), f"Missing column {col} in processed dataframe"

        # Check if features were generated (e.g., S1_Cn0DbHz_mean)
        feature_cols = [c for c in train_df.columns if c.startswith("S1_")]
        assert len(feature_cols) > 0, "No stratified features generated."
        print(f"  Generated {len(feature_cols)} stratified features.")

    except Exception as e:
        print(f"  Error during data processing: {e}")
        raise

    # 4. Dataset and DataLoader
    print("\n[4] Initializing Dataset and DataLoader...")
    # Initialize dataset (it will load the cache we just created or re-process if needed)
    train_dataset = GNSSSequenceDataset(
        split="train", load_cached_data=True, debug=True
    )

    print(f"  Dataset size (sequences): {len(train_dataset)}")

    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=gnss_collate_fn,
    )

    # Fetch one batch
    features, targets, mask, metadata_list = next(iter(train_loader))

    print(f"  Batch Features Shape: {features.shape} (Batch, Channels, Length)")
    print(f"  Batch Targets Shape:  {targets.shape}  (Batch, 2, Length)")
    print(f"  Batch Mask Shape:     {mask.shape}     (Batch, Length)")

    # Assertions
    assert (
        features.shape[1] == Config.IN_CHANNELS
    ), f"Expected {Config.IN_CHANNELS} channels, got {features.shape[1]}"
    assert (
        targets.shape[1] == 2
    ), f"Expected 2 target channels (dEast, dNorth), got {targets.shape[1]}"
    assert (
        features.shape[2] == targets.shape[2]
    ), "Feature length mismatch with Target length"
    assert len(metadata_list) == Config.BATCH_SIZE, "Metadata list length mismatch"

    # 5. Model Initialization and Forward Pass
    print("\n[5] Model Initialization and Forward Pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = StratifiedResUNet1D().to(device)

    # Move batch to device
    features = features.to(device)

    # Forward pass
    outputs = model(features)

    # Verify outputs
    assert "main" in outputs, "Model output missing 'main' head"
    main_out = outputs["main"]
    print(f"  Main Output Shape: {main_out.shape}")

    assert main_out.shape == (
        Config.BATCH_SIZE,
        2,
        features.shape[2],
    ), f"Output shape mismatch. Expected {(Config.BATCH_SIZE, 2, features.shape[2])}, got {main_out.shape}"

    # Check Aux heads
    for stride in Config.DEEP_SUPERVISION_STRIDES:
        key = f"aux_{stride}"
        if key in outputs:
            aux_out = outputs[key]
            print(f"  Aux Head {stride} Output Shape: {aux_out.shape}")
            # Aux output length depends on padding/pooling, roughly Length // stride
            # Just checking existence and channel count
            assert aux_out.shape[1] == 2, f"Aux head {stride} has wrong channel count"

    # 6. Training Loop Simulation
    print("\n[6] Running Training Loop (Simulation)...")
    # We call the library's train_model function which encapsulates the full loop
    # Note: We set load_cached_data=True to use the data processed in step 3

    try:
        best_model_path = train_model(load_cached_data=True)
        print(f"  Training complete. Best model saved at: {best_model_path}")
        assert os.path.exists(best_model_path), "Best model file was not created."
    except Exception as e:
        print(f"  Training failed: {e}")
        raise

    # 7. Inference Verification
    print("\n[7] Verifying Inference Logic...")
    # Simulate inference on the batch we fetched earlier
    model.eval()
    with torch.no_grad():
        outputs = model(features)
        preds = outputs["main"].cpu().numpy()  # (B, 2, L)

    # Take first sample in batch
    sample_idx = 0
    pred_offsets = preds[sample_idx].transpose(1, 0)  # (L, 2) -> dEast, dNorth

    # Get WLS baseline from metadata
    meta = metadata_list[sample_idx]
    wls_pos = meta["wls_pos"].numpy()  # (L, 2) -> Lat, Lon

    # Ensure lengths match (mask handles padding in real inference)
    valid_len = meta["timestamps"].shape[0]
    pred_offsets = pred_offsets[:valid_len]
    wls_pos = wls_pos[:valid_len]

    # Convert back to WGS84
    pred_lat, pred_lon = cartesian_to_wgs84(
        pred_offsets[:, 0],  # East
        pred_offsets[:, 1],  # North
        wls_pos[:, 0],  # Ref Lat
        wls_pos[:, 1],  # Ref Lon
    )

    print(f"  Sample 0 Sequence Length: {valid_len}")
    print(f"  WLS Position (t=0): ({wls_pos[0,0]:.6f}, {wls_pos[0,1]:.6f})")
    print(
        f"  Predicted Offset (t=0): E={pred_offsets[0,0]:.2f}, N={pred_offsets[0,1]:.2f}"
    )
    print(f"  Final Prediction (t=0): ({pred_lat[0]:.6f}, {pred_lon[0]:.6f})")

    # Sanity check: If offset is 0, prediction should equal WLS
    lat_zero, lon_zero = cartesian_to_wgs84(0, 0, wls_pos[0, 0], wls_pos[0, 1])
    assert np.isclose(
        lat_zero, wls_pos[0, 0]
    ), "Cartesian conversion identity check failed for Lat"
    assert np.isclose(
        lon_zero, wls_pos[0, 1]
    ), "Cartesian conversion identity check failed for Lon"

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
