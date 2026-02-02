import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
# Note: We assume the library files are in the python path.
from library.config import Config
import library.utils as utils
import library.data_processing as dp
import library.dataset as ds
import library.model as model_lib
import library.trainer as trainer
import library.inference as inference


def main():
    print("=== Starting Demonstration Script ===\n")

    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Override
    # -------------------------------------------------------------------------
    print("1. Configuring Environment...")

    # Define a temporary directory for this run's outputs
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR)

    # Override Config paths and parameters for speed
    Config.WORKING_DIR = os.path.join(DEMO_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Point to mini metadata files we will create shortly
    Config.TRAIN_METADATA_PATH = os.path.join(DEMO_DIR, "mini_train_meta.csv")
    Config.VAL_METADATA_PATH = os.path.join(DEMO_DIR, "mini_val_meta.csv")
    Config.TEST_METADATA_PATH = os.path.join(DEMO_DIR, "mini_test_meta.csv")

    # Reduce training parameters
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.DEBUG = True

    # Set seeds
    utils.set_seed(42)
    print("   Config updated for demo execution.")

    # -------------------------------------------------------------------------
    # 2. Create Mini-Datasets (Subsetting Metadata)
    # -------------------------------------------------------------------------
    print("\n2. Creating Mini-Metadata Files...")

    # Load original metadata
    orig_train_meta = pd.read_csv("./metadata/train_metadata.csv")
    orig_val_meta = pd.read_csv("./metadata/val_metadata.csv")
    orig_test_meta = pd.read_csv("./metadata/test_metadata.csv")

    # Select one drive for training
    train_drive = orig_train_meta["drive_id"].unique()[0]
    mini_train = orig_train_meta[orig_train_meta["drive_id"] == train_drive].copy()
    mini_train.to_csv(Config.TRAIN_METADATA_PATH, index=False)
    print(
        f"   Created mini train metadata with {len(mini_train)} rows (Drive: {train_drive})."
    )

    # Select one drive for validation (different from train if possible)
    val_drives = orig_val_meta["drive_id"].unique()
    val_drive = val_drives[0] if len(val_drives) > 0 else train_drive
    mini_val = orig_val_meta[orig_val_meta["drive_id"] == val_drive].copy()
    mini_val.to_csv(Config.VAL_METADATA_PATH, index=False)
    print(
        f"   Created mini val metadata with {len(mini_val)} rows (Drive: {val_drive})."
    )

    # Select one drive for testing
    test_drive = orig_test_meta["drive_id"].unique()[0]
    mini_test = orig_test_meta[orig_test_meta["drive_id"] == test_drive].copy()
    mini_test.to_csv(Config.TEST_METADATA_PATH, index=False)
    print(
        f"   Created mini test metadata with {len(mini_test)} rows (Drive: {test_drive})."
    )

    # -------------------------------------------------------------------------
    # 3. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n3. Verifying Coordinate Conversion Utilities...")

    # Test point (approximate location in Mountain View)
    lat_orig, lon_orig, alt_orig = 37.42, -122.08, 30.0

    # 1. Geodetic -> ENU -> Geodetic (Round Trip)
    # Define a reference point slightly offset
    ref_lat, ref_lon = 37.41, -122.09

    north, east = utils.geodetic_to_enu(lat_orig, lon_orig, ref_lat, ref_lon)
    lat_rec, lon_rec = utils.enu_to_geodetic(north, east, ref_lat, ref_lon)

    print(f"   Original: ({lat_orig}, {lon_orig})")
    print(f"   Recovered: ({lat_rec:.6f}, {lon_rec:.6f})")

    assert np.isclose(lat_orig, lat_rec, atol=1e-5), "Latitude round-trip failed"
    assert np.isclose(lon_orig, lon_rec, atol=1e-5), "Longitude round-trip failed"
    print("   Coordinate conversion verification passed.")

    # -------------------------------------------------------------------------
    # 4. Data Processing Demonstration
    # -------------------------------------------------------------------------
    print("\n4. Running Data Processing (Loading & Aggregation)...")

    # Process the mini training set
    # load_cached_data=False forces processing from raw files
    train_df = dp.process_dataset(
        Config.TRAIN_METADATA_PATH, load_cached_data=False, split_name="mini_train"
    )

    print(f"   Processed DataFrame Shape: {train_df.shape}")
    print(f"   Columns: {list(train_df.columns[:5])} ...")

    # Check for essential columns
    required_cols = ["lat_res_m", "lon_res_m", "Cn0DbHz_mean"]
    for col in required_cols:
        if col not in train_df.columns:
            raise AssertionError(
                f"Expected column {col} missing from processed dataframe"
            )

    print("   Data processing successful. Residuals and features computed.")

    # -------------------------------------------------------------------------
    # 5. Dataset and DataLoader Demonstration
    # -------------------------------------------------------------------------
    print("\n5. Initializing Dataset and DataLoader...")

    # Create dataset
    # load_cached_data=False forces sequence generation
    ds_train = ds.GNSSSequenceDataset(
        train_df, split_name="mini_train", load_cached_data=False
    )

    print(f"   Number of sequences in dataset: {len(ds_train)}")

    # Inspect one item
    item = ds_train[0]
    print(f"   Sample Item Keys: {item.keys()}")
    print(f"   Feature Shape (C, L): {item['features'].shape}")
    print(f"   Target Shape (C, L): {item['targets'].shape}")

    assert (
        item["features"].shape[0] == Config.IN_CHANNELS
    ), f"Feature channels {item['features'].shape[0]} != Config {Config.IN_CHANNELS}"
    assert (
        item["targets"].shape[0] == Config.OUTPUT_CHANNELS
    ), f"Target channels {item['targets'].shape[0]} != Config {Config.OUTPUT_CHANNELS}"

    # Create DataLoader
    dl_train = torch.utils.data.DataLoader(
        ds_train, batch_size=Config.BATCH_SIZE, collate_fn=ds.collate_padded
    )

    # Fetch one batch
    batch = next(iter(dl_train))
    print(f"   Batch Keys: {batch.keys()}")
    print(f"   Batch Features Shape (B, C, L): {batch['features'].shape}")
    print(f"   Batch Targets Shape (B, C, L): {batch['targets'].shape}")
    print(f"   Batch Lengths: {batch['lengths']}")

    print("   Dataset and DataLoader verification passed.")

    # -------------------------------------------------------------------------
    # 6. Model Demonstration
    # -------------------------------------------------------------------------
    print("\n6. Initializing Model and Forward Pass...")

    device = Config.DEVICE
    model = model_lib.UNet1D().to(device)

    # Move batch to device
    inputs = batch["features"].to(device)

    # Forward pass
    outputs = model(inputs)

    print(f"   Input Shape: {inputs.shape}")
    print(f"   Output Shape: {outputs.shape}")

    assert (
        outputs.shape == batch["targets"].shape
    ), f"Output shape {outputs.shape} mismatch with targets {batch['targets'].shape}"

    print("   Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 7. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n7. Running Training Loop (1 Epoch)...")

    # We call the trainer.train_model function.
    # It will reload the data using the Config paths we overrode.
    # It returns the trained model and the feature stats used for normalization.
    trained_model, feature_stats = trainer.train_model(load_cached_data=True)

    weights_path = os.path.join(Config.WORKING_DIR, "model_weights.pth")
    if os.path.exists(weights_path):
        print(f"   Training complete. Weights saved at: {weights_path}")
    else:
        raise AssertionError("Model weights file was not created.")

    # -------------------------------------------------------------------------
    # 8. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n8. Running Inference and Generating Submission...")

    # Call inference.generate_submission
    # This uses Config.TEST_METADATA_PATH which we set to the mini test set
    inference.generate_submission(trained_model, feature_stats, load_cached_data=False)

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"   Submission generated at: {submission_path}")
        print(f"   Submission Shape: {df_sub.shape}")
        print(f"   Submission Columns: {list(df_sub.columns)}")

        # Basic validation
        expected_cols = [
            "tripId",
            "UnixTimeMillis",
            "LatitudeDegrees",
            "LongitudeDegrees",
        ]
        if not all(col in df_sub.columns for col in expected_cols):
            raise AssertionError(
                f"Submission missing required columns. Found: {df_sub.columns}"
            )
        if len(df_sub) == 0:
            raise AssertionError("Submission file is empty.")
    else:
        raise AssertionError("Submission file was not created.")

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
