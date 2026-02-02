import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, angles_to_direction, direction_to_angles
from library.geometry import (
    load_sensor_geometry,
    compute_canonical_rotation,
    apply_rotation,
)
from library.data import process_batch, IceCubeDataset
from library.model import DV_AGN
from library.loss import CosineSimilarityLoss
from library.train import train_model
from library.inference import generate_submission


def create_subset_metadata(source_dir, dest_dir, n_rows=200):
    """
    Creates subset metadata files to speed up the demo.
    Preserves valid batch_ids that exist in the input directory.
    """
    os.makedirs(dest_dir, exist_ok=True)

    files = ["train_metadata.parquet", "val_metadata.parquet", "test_metadata.parquet"]
    for f in files:
        src_path = os.path.join(source_dir, f)
        if os.path.exists(src_path):
            df = pd.read_parquet(src_path)
            # Take a small subset, ensuring we keep intact batches if possible,
            # or just the first n rows.
            # We need to make sure we don't pick a batch_id that doesn't exist
            # (though metadata comes from existing files, so it should be fine).

            # Filter for a single batch to make it very fast
            unique_batches = df["batch_id"].unique()
            if len(unique_batches) > 0:
                target_batch = unique_batches[0]
                subset = df[df["batch_id"] == target_batch].head(n_rows)
            else:
                subset = df.head(n_rows)

            subset.to_parquet(os.path.join(dest_dir, f))
            print(
                f"Created subset for {f} with {len(subset)} rows (Batch ID: {subset['batch_id'].unique()})"
            )


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print("--- 1. Configuration & Setup ---")

    # Override Config for Demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Point metadata to a temp dir where we will put subsets
    ORIGINAL_METADATA_DIR = Config.METADATA_DIR
    Config.METADATA_DIR = os.path.join(Config.WORKING_DIR, "metadata_subset")

    # Reduce training parameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Setup directories and seeds
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.setup()
    seed_everything(Config.SEED)

    # Create subset metadata
    create_subset_metadata(ORIGINAL_METADATA_DIR, Config.METADATA_DIR, n_rows=50)

    # -------------------------------------------------------------------------
    # 2. Geometry & Utils Verification
    # -------------------------------------------------------------------------
    print("\n--- 2. Geometry & Utils Verification ---")

    # Load Geometry
    sensor_map = load_sensor_geometry()
    print(f"Loaded geometry for {len(sensor_map)} sensors.")
    assert len(sensor_map) > 5000, "Sensor map should contain over 5000 sensors."

    # Verify Angle <-> Direction conversion
    az_true = torch.tensor([0.0, np.pi / 2, np.pi], dtype=torch.float32)
    ze_true = torch.tensor(
        [np.pi / 2, np.pi / 2, 0.0], dtype=torch.float32
    )  # x-axis, y-axis, z-axis

    vectors = angles_to_direction(az_true, ze_true)

    # Expected vectors: (1,0,0), (0,1,0), (0,0,1)
    expected = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float32
    )

    assert torch.allclose(vectors, expected, atol=1e-6), "angles_to_direction failed."

    az_rec, ze_rec = direction_to_angles(vectors)
    # Note: direction_to_angles output range is [0, 2pi] for az, [0, pi] for ze
    assert torch.allclose(
        az_rec, az_true, atol=1e-6
    ), "direction_to_angles (azimuth) failed."
    assert torch.allclose(
        ze_rec, ze_true, atol=1e-6
    ), "direction_to_angles (zenith) failed."
    print("Geometry and Utils verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Data Processing Verification
    # -------------------------------------------------------------------------
    print("\n--- 3. Data Processing Verification ---")

    # Load the subset metadata we created
    train_meta = pd.read_parquet(
        os.path.join(Config.METADATA_DIR, "train_metadata.parquet")
    )
    batch_id = train_meta["batch_id"].iloc[0]

    print(f"Processing batch {batch_id}...")
    X_raw, X_canon, targets = process_batch(
        batch_id,
        train_meta,
        sensor_map,
        mode="train",
        load_cached_data=False,  # Force processing
    )

    print(f"X_raw shape: {X_raw.shape}")
    print(f"X_canon shape: {X_canon.shape}")
    print(f"Targets shape: {targets.shape}")

    # Assertions
    assert X_raw.ndim == 3 and X_raw.shape[2] == 6, "X_raw should be [N, Max_Pulses, 6]"
    assert (
        X_canon.ndim == 3 and X_canon.shape[2] == 6
    ), "X_canon should be [N, Max_Pulses, 6]"
    assert targets.shape[1] == 2, "Targets should be [N, 2]"
    assert not np.isnan(X_raw).any(), "X_raw contains NaNs"

    # Test Dataset and DataLoader
    dataset = IceCubeDataset(X_raw, X_canon, targets)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    batch_X_raw, batch_X_canon, batch_y = next(iter(loader))
    assert batch_X_raw.shape[0] == 4, "DataLoader batch size mismatch"
    print("Data processing pipeline verified.")

    # -------------------------------------------------------------------------
    # 4. Model & Loss Verification
    # -------------------------------------------------------------------------
    print("\n--- 4. Model & Loss Verification ---")

    model = DV_AGN().to(Config.DEVICE)
    model.eval()

    # Move batch to device
    batch_X_raw = batch_X_raw.to(Config.DEVICE)
    batch_X_canon = batch_X_canon.to(Config.DEVICE)
    batch_y = batch_y.to(Config.DEVICE)

    # Forward pass
    with torch.no_grad():
        preds = model(batch_X_raw, batch_X_canon)

    print(f"Prediction shape: {preds.shape}")
    assert preds.shape == (4, 3), "Model output shape should be [Batch, 3]"

    # Loss calculation
    criterion = CosineSimilarityLoss()
    loss = criterion(preds, batch_y[:, 0], batch_y[:, 1])

    print(f"Calculated Loss: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"
    print("Model and Loss verified.")

    # -------------------------------------------------------------------------
    # 5. Training Pipeline Demonstration
    # -------------------------------------------------------------------------
    print("\n--- 5. Training Pipeline Demonstration ---")

    # We use the library function train_model.
    # It uses Config.METADATA_DIR which we pointed to our subsets.
    # It saves the model to Config.WORKING_DIR/model.pth

    print("Running training loop (1 epoch, subset data)...")
    train_model(
        load_cached_data=True,  # Allow caching now that we verified processing
        epochs=1,
        patience=1,
        debug=False,  # We already manually subsetted the metadata, so debug=False is fine
        save_path=os.path.join(Config.WORKING_DIR, "model.pth"),
    )

    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "model.pth")
    ), "Model file was not saved."
    print("Training pipeline completed.")

    # -------------------------------------------------------------------------
    # 6. Inference Pipeline Demonstration
    # -------------------------------------------------------------------------
    print("\n--- 6. Inference Pipeline Demonstration ---")

    # generate_submission reads test_metadata from Config.METADATA_DIR
    # and uses the model at Config.WORKING_DIR/model.pth

    print("Running inference...")
    generate_submission(load_cached_data=True)

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    df_sub = pd.read_csv(submission_path)
    print(f"Submission generated with {len(df_sub)} rows.")
    print(df_sub.head())

    assert "event_id" in df_sub.columns
    assert "azimuth" in df_sub.columns
    assert "zenith" in df_sub.columns

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
