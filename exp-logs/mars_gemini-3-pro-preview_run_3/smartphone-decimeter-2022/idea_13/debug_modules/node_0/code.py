import os
import sys
import numpy as np
import pandas as pd
import warnings
import shutil

# Import from the provided library files
import library.config
import library.utils
import library.data_loader
import library.feature_engineering
import library.model


def main():
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    print(">>> 1. Setup and Configuration")
    warnings.filterwarnings("ignore")

    # Set random seeds for reproducibility
    np.random.seed(42)

    # Define working directories
    WORKING_DIR = "./working"
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Patch library configuration for speed optimization in this demo
    print("Patching configuration for fast execution...")
    library.config.N_FOLDS = 2  # Reduce folds
    library.config.LGB_PARAMS["n_estimators"] = 10  # Reduce boosting rounds
    library.config.LGB_PARAMS["num_leaves"] = 8  # Reduce tree complexity
    library.config.OUTPUT_DIR = os.path.join(WORKING_DIR, "demo_output")

    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n>>> 2. Verifying Utility Functions")

    # Test Haversine Distance
    # Distance between same point should be 0
    d_zero = library.config.haversine_distance(0, 0, 0, 0)
    assert d_zero == 0.0, "Haversine distance for same point should be 0."

    # Distance for 1 degree latitude at equator is roughly 111 km
    d_deg = library.config.haversine_distance(0, 0, 1, 0)
    assert (
        110000 < d_deg < 112000
    ), f"Haversine distance for 1 deg lat unexpected: {d_deg}"
    print("Haversine distance check passed.")

    # Test Coordinate Transformations (Round Trip)
    lat_orig, lon_orig, alt_orig = 37.42, -122.08, 30.0
    x, y, z = library.utils.lla_to_ecef(lat_orig, lon_orig, alt_orig)
    lat_new, lon_new, alt_new = library.utils.ecef_to_lla(x, y, z)

    assert np.isclose(lat_orig, lat_new, atol=1e-5), "Latitude round-trip failed."
    assert np.isclose(lon_orig, lon_new, atol=1e-5), "Longitude round-trip failed."
    assert np.isclose(alt_orig, alt_new, atol=1e-3), "Altitude round-trip failed."
    print("Coordinate transformation round-trip check passed.")

    # 3. Data Loading and Feature Engineering
    # -------------------------------------------------------------------------
    print("\n>>> 3. Data Loading and Feature Engineering")

    # Create a mini metadata file to process only a small subset of data
    full_meta_path = "./metadata/train_metadata.csv"
    if not os.path.exists(full_meta_path):
        raise FileNotFoundError(f"Metadata file not found at {full_meta_path}")

    df_meta = pd.read_csv(full_meta_path)

    # Select a single drive to maintain time-series continuity
    sample_drive_id = df_meta["drive_id"].unique()[0]
    print(f"Sampling data from drive: {sample_drive_id}")

    # Take top 100 rows from this drive
    mini_meta = df_meta[df_meta["drive_id"] == sample_drive_id].head(100).copy()
    mini_meta_path = os.path.join(WORKING_DIR, "mini_train_meta.csv")
    mini_meta.to_csv(mini_meta_path, index=False)

    # Load dataset using the library function
    # load_cached_data=False ensures we run the raw data processing logic
    print("Processing raw GNSS data (calculating residuals, aggregating sectors)...")
    train_df = library.data_loader.get_dataset(
        metadata_path=mini_meta_path, load_cached_data=False, split="train"
    )

    print(f"Processed dataframe shape: {train_df.shape}")

    # Verify that targets were computed
    assert "target_E" in train_df.columns, "Target East (target_E) missing."
    assert "target_N" in train_df.columns, "Target North (target_N) missing."

    # Verify that features were generated
    # We expect columns like 'pr_residual_mean_s0' (sector 0) or 'global_pr_residual_mean'
    feature_cols = [c for c in train_df.columns if "pr_residual" in c]
    assert (
        len(feature_cols) > 0
    ), "Feature engineering failed to generate pseudorange residual features."
    print(f"Generated {len(feature_cols)} residual-related feature columns.")

    # Verify WLS coordinates are present (needed for reconstruction)
    assert "wls_lat" in train_df.columns, "WLS Latitude missing."
    assert "wls_lon" in train_df.columns, "WLS Longitude missing."

    # 4. Model Training
    # -------------------------------------------------------------------------
    print("\n>>> 4. Model Training")

    # Train the model using the mini dataset
    # This uses GroupKFold internally based on 'drive_id'
    models_E, models_N, feature_names = library.model.train_model(train_df)

    # Verify model output
    assert (
        len(models_E) == library.config.N_FOLDS
    ), f"Expected {library.config.N_FOLDS} East models."
    assert (
        len(models_N) == library.config.N_FOLDS
    ), f"Expected {library.config.N_FOLDS} North models."
    print("Training completed successfully.")

    # 5. Inference and Reconstruction
    # -------------------------------------------------------------------------
    print("\n>>> 5. Inference and Reconstruction")

    # Generate predictions on the same training data (acting as test data for demonstration)
    pred_E = library.model.predict(models_E, train_df, feature_names)
    pred_N = library.model.predict(models_N, train_df, feature_names)

    assert len(pred_E) == len(train_df), "Prediction size mismatch."

    print(f"Mean Predicted East Residual: {np.mean(pred_E):.4f} meters")
    print(f"Mean Predicted North Residual: {np.mean(pred_N):.4f} meters")

    # Reconstruct Latitude/Longitude from predicted ENU residuals
    # New Lat = WLS_Lat + dN / 111320
    # New Lon = WLS_Lon + dE / (111320 * cos(Lat))

    wls_lat_rad = np.radians(train_df["wls_lat"])
    pred_lat = train_df["wls_lat"] + (pred_N / 111320.0)
    pred_lon = train_df["wls_lon"] + (pred_E / (111320.0 * np.cos(wls_lat_rad)))

    # Verify reconstruction validity
    assert not pred_lat.isnull().any(), "NaN values in predicted Latitude."
    assert not pred_lon.isnull().any(), "NaN values in predicted Longitude."

    # Compare with WLS baseline (sanity check)
    # The model should adjust the position, so diff should not be exactly zero,
    # but should be small (within reasonable GPS error range, e.g., < 100m)
    diff_lat_m = (pred_lat - train_df["wls_lat"]) * 111320.0
    print(
        f"Average correction applied to Latitude: {np.mean(np.abs(diff_lat_m)):.4f} meters"
    )

    print("\n>>> Demo Execution Completed Successfully.")


if __name__ == "__main__":
    main()
