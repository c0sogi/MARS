import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil

# Import from the provided library files
import library.config as config
from library.geometry_utils import process_geometry
from library.feature_extractor import get_atomic_features, get_global_features
from library.data_loader import get_dataloaders
from library.model import CCWDS
from library.training import run_training


def main():
    print("Starting demonstration of the CCWDS pipeline...")

    # 1. Setup and Configuration Overrides for Speed
    # We override config constants to ensure the demo runs quickly and uses a separate working directory
    config.WORKING_DIR = "./working/demo_execution"
    config.TRAIN_CACHE_PATH = os.path.join(config.WORKING_DIR, "train_data.npz")
    config.VAL_CACHE_PATH = os.path.join(config.WORKING_DIR, "val_data.npz")
    config.TEST_CACHE_PATH = os.path.join(config.WORKING_DIR, "test_data.npz")
    config.SCALERS_CACHE_PATH = os.path.join(config.WORKING_DIR, "scalers.npz")
    config.MODEL_SAVE_PATH = os.path.join(config.WORKING_DIR, "demo_model.pt")
    config.SUBMISSION_DIR = "./working/demo_submission"
    config.SUBMISSION_PATH = os.path.join(config.SUBMISSION_DIR, "demo_submission.csv")

    # Ensure directories exist
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)
    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

    # Set seeds
    config.set_seed(42)

    # 2. Geometry Processing Demo
    print("\n--- Testing Geometry Processing ---")
    # Load metadata to get a valid file path
    train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    sample_row = train_df.iloc[0]
    sample_file_path = os.path.join(config.INPUT_DIR, sample_row["file_path"])

    print(f"Processing file: {sample_file_path}")
    geo_data = process_geometry(sample_file_path)

    # Assertions for geometry data
    assert "coords" in geo_data
    assert "species" in geo_data
    assert "nn_dist" in geo_data
    assert "lattice" in geo_data
    assert geo_data["coords"].shape[1] == 3
    assert len(geo_data["species"]) == geo_data["coords"].shape[0]
    print("Geometry processing successful. Keys:", geo_data.keys())

    # 3. Feature Extraction Demo
    print("\n--- Testing Feature Extraction ---")
    atomic_feats = get_atomic_features(geo_data)
    global_feats = get_global_features(sample_row, geo_data)

    print(f"Atomic features shape: {atomic_feats.shape}")
    print(f"Global features shape: {global_feats.shape}")

    # Assertions for feature dimensions
    # Atomic: 4 (Self) + 3 (Spatial) + 1 (Dist) + 4 (NN) = 12
    assert (
        atomic_feats.shape[1] == config.ATOM_INPUT_DIM
    ), f"Expected {config.ATOM_INPUT_DIM}, got {atomic_feats.shape[1]}"
    # Global: 3 (Lat len) + 3 (Lat ang) + 1 (Vol) + 1 (Dens) + 3 (Stoich) + 1 (N_atoms) = 12
    assert (
        global_feats.shape[0] == config.GLOBAL_INPUT_DIM
    ), f"Expected {config.GLOBAL_INPUT_DIM}, got {global_feats.shape[0]}"
    print("Feature extraction successful.")

    # 4. Data Loader Demo
    print("\n--- Testing Data Loader ---")
    # Use a small subset (max_samples=50) and small batch size
    batch_size = 4
    max_samples = 50

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=0,
        load_cached_data=False,  # Force processing
        max_samples=max_samples,
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    batch_atomic, batch_indices, batch_global, batch_targets, batch_ids = batch

    print(f"Batch Atomic Features: {batch_atomic.shape}")
    print(f"Batch Indices: {batch_indices.shape}")
    print(f"Batch Global Features: {batch_global.shape}")
    print(f"Batch Targets: {batch_targets.shape}")

    # Assertions
    assert batch_global.shape[0] == batch_size
    assert batch_targets.shape == (batch_size, 2)
    assert batch_atomic.shape[0] == batch_indices.shape[0]
    # Check if indices are within range [0, batch_size-1]
    assert batch_indices.max() < batch_size
    print("Data loading successful.")

    # 5. Model Forward Pass Demo
    print("\n--- Testing Model Forward Pass ---")
    model = CCWDS()
    model.to(config.DEVICE)

    # Move batch to device
    batch_atomic = batch_atomic.to(config.DEVICE)
    batch_indices = batch_indices.to(config.DEVICE)
    batch_global = batch_global.to(config.DEVICE)

    # Forward
    outputs = model(batch_atomic, batch_indices, batch_global)

    print(f"Model Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (batch_size, 2)
    print("Model forward pass successful.")

    # 6. Full Training Pipeline Demo
    print("\n--- Testing Full Training Pipeline (Reduced) ---")
    # We run for 1 epoch on the small subset
    # Note: run_training internally calls train_model which calls get_dataloaders.
    # Since we already populated the cache in step 4 (get_dataloaders with load_cached_data=False),
    # run_training will pick up the cached data if we allow it, or we can force it.
    # Here we just run it.

    try:
        run_training(max_samples=max_samples, epochs=1)
        print("Training pipeline execution completed.")
    except Exception as e:
        print(f"Training pipeline failed: {e}")
        raise e

    # 7. Verify Submission
    print("\n--- Verifying Submission File ---")
    if os.path.exists(config.SUBMISSION_PATH):
        sub_df = pd.read_csv(config.SUBMISSION_PATH)
        print(f"Submission file found with {len(sub_df)} rows.")
        print(sub_df.head())

        # Assertions
        assert (
            len(sub_df) == max_samples
        ), f"Expected {max_samples} predictions, got {len(sub_df)}"
        assert "id" in sub_df.columns
        assert "formation_energy_ev_natom" in sub_df.columns
        assert "bandgap_energy_ev" in sub_df.columns
        # Check for non-negative values (physics constraint applied in generate_submission)
        assert (sub_df["formation_energy_ev_natom"] >= 0).all()
        assert (sub_df["bandgap_energy_ev"] >= 0).all()
        print("Submission file verification successful.")
    else:
        raise FileNotFoundError(
            f"Submission file was not created at {config.SUBMISSION_PATH}"
        )

    print("\nAll demonstrations passed successfully.")


if __name__ == "__main__":
    main()
