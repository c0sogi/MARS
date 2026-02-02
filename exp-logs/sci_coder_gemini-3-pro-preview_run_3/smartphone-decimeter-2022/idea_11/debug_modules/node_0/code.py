import pandas as pd
import numpy as np
import os
import warnings

# Import library modules
import library.config
import library.features
import library.model
import library.utils
import library.data_loader

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("Starting Physics Ensemble Model Demonstration...")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("\n[1] Overriding Configuration for Speed...")

    # Override LightGBM parameters for quick training execution
    library.model.LGBM_PARAMS["n_estimators"] = 10
    library.model.LGBM_PARAMS["min_child_samples"] = 5
    library.model.LGBM_PARAMS["learning_rate"] = 0.1

    # Override Cross-Validation settings to reduce folds
    library.model.N_FOLDS = 2
    library.model.EARLY_STOPPING_ROUNDS = 5
    library.model.VERBOSE_EVAL = 1

    print(f"   N_FOLDS set to: {library.model.N_FOLDS}")
    print(f"   n_estimators set to: {library.model.LGBM_PARAMS['n_estimators']}")

    # -------------------------------------------------------------------------
    # 2. Create Subset Metadata
    # -------------------------------------------------------------------------
    print("\n[2] Creating Subset Metadata...")

    # We create small metadata files pointing to the first few rows of the real metadata.
    # This ensures we only process a small chunk of the raw data (one drive likely).

    temp_dir = "./working"
    os.makedirs(temp_dir, exist_ok=True)

    temp_train_meta_path = os.path.join(temp_dir, "temp_train_meta.csv")
    temp_val_meta_path = os.path.join(temp_dir, "temp_val_meta.csv")
    temp_test_meta_path = os.path.join(temp_dir, "temp_test_meta.csv")

    def create_subset(src, dst, n=100):
        if os.path.exists(src):
            df = pd.read_csv(src)
            # Filter for a single drive if possible to ensure continuity and minimize I/O
            if "drive_id" in df.columns:
                first_drive = df["drive_id"].iloc[0]
                df = df[df["drive_id"] == first_drive]

            subset = df.head(n)
            subset.to_csv(dst, index=False)
            print(f"   Created {dst} with {len(subset)} rows from {src}")
            return True
        else:
            print(f"   Source {src} not found.")
            return False

    # Generate subsets
    has_train = create_subset(
        library.config.TRAIN_METADATA_PATH, temp_train_meta_path, 200
    )
    has_val = create_subset(library.config.VAL_METADATA_PATH, temp_val_meta_path, 100)
    has_test = create_subset(
        library.config.TEST_METADATA_PATH, temp_test_meta_path, 100
    )

    if not (has_train and has_val and has_test):
        raise FileNotFoundError("Could not create subset metadata. Check input files.")

    # -------------------------------------------------------------------------
    # 3. Feature Generation
    # -------------------------------------------------------------------------
    print("\n[3] Generating Features (Processing Raw Logs)...")

    # We call generate_dataset with load_cached_data=False to force processing
    # of our new temp metadata files. This reads the raw GNSS/IMU files referenced
    # in the metadata and computes physics-based features.

    try:
        print("   Processing Train Subset...")
        train_df = library.features.generate_dataset(
            temp_train_meta_path, "train", load_cached_data=False
        )

        print("   Processing Val Subset...")
        val_df = library.features.generate_dataset(
            temp_val_meta_path, "val", load_cached_data=False
        )

        print("   Processing Test Subset...")
        test_df = library.features.generate_dataset(
            temp_test_meta_path, "test", load_cached_data=False
        )

    except Exception as e:
        print(f"   Error during feature generation: {e}")
        return

    print(f"   Train Features Shape: {train_df.shape}")
    print(f"   Val Features Shape: {val_df.shape}")
    print(f"   Test Features Shape: {test_df.shape}")

    # Verify features exist
    expected_cols = library.config.FEATURE_COLS
    assert all(
        col in train_df.columns for col in expected_cols
    ), "Missing features in Train dataframe"

    # Verify targets exist in train/val
    for target in library.config.TARGET_COLS:
        assert target in train_df.columns, f"Target {target} missing in Train dataframe"
        assert target in val_df.columns, f"Target {target} missing in Val dataframe"

    # -------------------------------------------------------------------------
    # 4. Model Training
    # -------------------------------------------------------------------------
    print("\n[4] Training Physics Ensemble Model...")

    model = library.model.PhysicsEnsembleModel()

    # Train the model using the subset data
    # This will use the overridden N_FOLDS=2 and n_estimators=10
    model.train(train_df, val_df)

    # Verify models were created correctly
    assert (
        len(model.models_east) == library.model.N_FOLDS
    ), f"Expected {library.model.N_FOLDS} east models, got {len(model.models_east)}"
    assert (
        len(model.models_north) == library.model.N_FOLDS
    ), f"Expected {library.model.N_FOLDS} north models, got {len(model.models_north)}"
    print("   Training successful.")

    # -------------------------------------------------------------------------
    # 5. Prediction
    # -------------------------------------------------------------------------
    print("\n[5] Generating Predictions...")

    pred_east, pred_north = model.predict(test_df)

    print(f"   Predictions shape: East {pred_east.shape}, North {pred_north.shape}")
    assert pred_east.shape[0] == len(test_df), "Prediction count mismatch"
    assert not np.isnan(pred_east).any(), "NaNs found in predictions"

    # -------------------------------------------------------------------------
    # 6. Coordinate Reconstruction
    # -------------------------------------------------------------------------
    print("\n[6] Reconstructing WGS84 Coordinates...")

    # Convert the predicted ENU offsets back to Latitude/Longitude
    pred_lat, pred_lon = model.reconstruct_coords(test_df, pred_east, pred_north)

    print(f"   Reconstructed Lat/Lon sample: {pred_lat[0]:.5f}, {pred_lon[0]:.5f}")

    # Basic validity check (Lat between -90, 90, Lon between -180, 180)
    assert np.all((pred_lat >= -90) & (pred_lat <= 90)), "Invalid Latitude generated"
    assert np.all((pred_lon >= -180) & (pred_lon <= 180)), "Invalid Longitude generated"

    # -------------------------------------------------------------------------
    # 7. Utility Verification
    # -------------------------------------------------------------------------
    print("\n[7] Verifying Utilities...")

    # Test Haversine Distance
    # Distance between (0,0) and (1,0) degrees is approx 111km
    dist = library.utils.haversine(0, 0, 1, 0)
    print(f"   Haversine(0,0 -> 1,0): {dist:.2f} meters")
    assert 110000 < dist < 112000, "Haversine calculation incorrect"

    # Test ECEF to ENU Conversion
    # Origin at (0,0,0) ECEF is center of earth.
    # We test relative offset from a point on the surface.
    # Point A: Lat 0, Lon 0, Alt 0
    # Point B: Lat 0, Lon 0, Alt 10
    # ENU of B relative to A should be (0, 0, 10)
    xa, ya, za = library.utils.wgs84_to_ecef(0, 0, 0)
    xb, yb, zb = library.utils.wgs84_to_ecef(0, 0, 10)
    e, n, u = library.utils.ecef_to_enu(xb, yb, zb, 0, 0, 0)
    print(f"   ENU Test (Vertical 10m): East={e:.2f}, North={n:.2f}, Up={u:.2f}")

    assert np.isclose(e, 0, atol=1e-3), "ENU East calculation incorrect"
    assert np.isclose(n, 0, atol=1e-3), "ENU North calculation incorrect"
    assert np.isclose(u, 10, atol=1e-3), "ENU Up calculation incorrect"

    print("\nDone. All demonstrations and checks passed.")


if __name__ == "__main__":
    run_demo()
