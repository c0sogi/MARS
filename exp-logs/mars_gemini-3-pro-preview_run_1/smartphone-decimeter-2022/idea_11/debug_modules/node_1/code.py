import os
import numpy as np
import pandas as pd
import torch
import sys

# Import from the provided library
import library.utils as utils
import library.data as data
import library.model as model
import library.train as train
import library.inference as inference


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Seed
    utils.set_seed(42)
    WORKING_DIR = "./working"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # 2. Demonstrate Utils
    print("\n--- Testing Utils ---")
    # Test coordinate conversions with a known point (Googleplex approx)
    lat, lon, alt = 37.4219999, -122.0840575, 10.0
    print(f"Original Geodetic: Lat={lat}, Lon={lon}, Alt={alt}")

    # Geodetic to ECEF
    x, y, z = utils.geodetic_to_ecef(lat, lon, alt)
    print(f"ECEF: X={x:.2f}, Y={y:.2f}, Z={z:.2f}")

    # ECEF back to Geodetic
    lat_r, lon_r, alt_r = utils.ecef_to_geodetic(x, y, z)
    print(f"Reconstructed: Lat={lat_r}, Lon={lon_r}, Alt={alt_r}")

    # Verify round-trip accuracy
    assert np.isclose(lat, lat_r, atol=1e-5), "Latitude mismatch"
    assert np.isclose(lon, lon_r, atol=1e-5), "Longitude mismatch"
    assert np.isclose(alt, alt_r, atol=1e-3), "Altitude mismatch"
    print("Coordinate conversion round-trip successful.")

    # ENU Conversion relative to self (should be 0,0,0)
    e, n, u = utils.geodetic_to_enu(lat, lon, alt, lat0=lat, lon0=lon, alt0=alt)
    print(f"ENU relative to self: E={e:.2f}, N={n:.2f}, U={u:.2f}")
    assert np.isclose(e, 0, atol=1e-3), "East offset not zero"
    assert np.isclose(n, 0, atol=1e-3), "North offset not zero"
    assert np.isclose(u, 0, atol=1e-3), "Up offset not zero"
    print("ENU conversion verified.")

    # 3. Demonstrate Data Loading
    print("\n--- Testing Data Loading ---")
    # We use the existing metadata files but limit max_drives to 1 for speed
    train_meta_path = "./metadata/train_metadata.csv"

    input_channels = 30  # Default fallback

    if not os.path.exists(train_meta_path):
        print("Metadata not found. Skipping real data loading.")
    else:
        print("Loading training data (max_drives=1)...")
        # This uses the caching mechanism in library.data.
        # It processes raw GNSS logs into features.
        train_dataset = data.load_data(
            train_meta_path, split="train", max_drives=1, load_cached_data=True
        )

        if len(train_dataset) > 0:
            print(f"Loaded {len(train_dataset)} trips.")
            # Get first sample
            x, y, meta = train_dataset[0]
            print(f"Sample 0 Feature Shape: {x.shape}")  # (SeqLen, Features)
            print(f"Sample 0 Target Shape: {y.shape}")  # (SeqLen, 2)
            print(f"Sample 0 Meta keys: {list(meta.keys())}")

            # Verify shapes match sequence length
            assert (
                x.shape[0] == y.shape[0]
            ), "Feature and Target sequence lengths mismatch"
            assert y.shape[1] == 2, "Target should have 2 dimensions (East, North)"
            print("Data shapes verified.")

            input_channels = x.shape[1]
        else:
            print("No training data loaded (dataset empty).")

    # 4. Demonstrate Model
    print("\n--- Testing Model ---")
    # Instantiate model
    print(f"Instantiating ResUNet1D with {input_channels} input channels.")
    net = model.ResUNet1D(in_channels=input_channels, out_channels=2, base_channels=16)

    # Create dummy input batch (Batch=2, SeqLen=100, Features=input_channels)
    # The model expects (Batch, SeqLen, Features) and permutes internally
    dummy_input = torch.randn(2, 100, input_channels)
    print(f"Dummy Input Shape: {dummy_input.shape}")

    output = net(dummy_input)
    print(f"Output Shape: {output.shape}")

    assert output.shape == (2, 100, 2), "Output shape mismatch"
    print("Forward pass successful.")

    # 5. Demonstrate Training
    print("\n--- Testing Training Loop ---")
    # We will train for 1 epoch with max_drives=1 to ensure speed.
    # train_model handles data loading internally.
    val_meta_path = "./metadata/val_metadata.csv"

    if os.path.exists(train_meta_path) and os.path.exists(val_meta_path):
        print("Running training simulation...")
        trainer = train.train_model(
            epochs=1,
            batch_size=1,
            learning_rate=1e-3,
            patience=1,
            max_drives=1,  # Limit data to 1 drive for speed
            load_cached_data=True,
        )

        # Check if model saved
        if os.path.exists(train.MODEL_SAVE_PATH):
            print(f"Model saved to {train.MODEL_SAVE_PATH}")
        else:
            print("Model was not saved (possibly due to loss logic or empty data).")
    else:
        print("Metadata missing, skipping training.")

    # 6. Demonstrate Inference
    print("\n--- Testing Inference ---")
    # Create a mini test metadata file to limit inference time
    orig_test_meta_path = "./metadata/test_metadata.csv"
    mini_test_meta_path = os.path.join(WORKING_DIR, "mini_test_meta.csv")

    if os.path.exists(orig_test_meta_path):
        df_test = pd.read_csv(orig_test_meta_path)
        if not df_test.empty:
            # Pick one trip
            one_trip_id = df_test["tripId"].iloc[0]
            df_mini = df_test[df_test["tripId"] == one_trip_id].copy()
            df_mini.to_csv(mini_test_meta_path, index=False)
            print(
                f"Created mini test metadata with {len(df_mini)} rows at {mini_test_meta_path}"
            )

            # Monkeypatch the constant in library.inference to use our mini file
            inference.TEST_META = mini_test_meta_path

            # Ensure model exists for inference (if training didn't run/save, create a dummy one)
            if not os.path.exists(inference.MODEL_PATH):
                print("Creating dummy model file for inference...")
                dummy_model = model.ResUNet1D(
                    in_channels=input_channels, out_channels=2, base_channels=16
                )
                torch.save(dummy_model.state_dict(), inference.MODEL_PATH)

            print("Running inference pipeline...")
            # This runs load_data on the mini test meta, loads the model, predicts, and saves CSV
            inference.generate_submission(load_cached_data=True)

            # Check submission
            submission_path = os.path.join(inference.SUBMISSION_DIR, "submission.csv")
            if os.path.exists(submission_path):
                sub_df = pd.read_csv(submission_path)
                print(f"Submission generated with {len(sub_df)} rows.")
                print(sub_df.head())
            else:
                print("Submission file not found.")
        else:
            print("Test metadata is empty.")
    else:
        print("Original test metadata not found.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
