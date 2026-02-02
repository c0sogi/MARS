import sys
import os
import shutil
import numpy as np
import torch
import pandas as pd

# Add current directory to sys.path to ensure library can be imported
sys.path.append(".")

# Import config to override settings for the demo
import library.config as config

# ==========================================
# Configuration Overrides for Demo
# ==========================================
# Override config parameters to run a fast demonstration
config.DEBUG_MODE = True
config.DEBUG_DATA_SIZE = 50  # Use only 50 samples for speed
config.NUM_EPOCHS = 2  # Train for only 2 epochs
config.BATCH_SIZE = 16  # Smaller batch size
config.WORKING_DIR = "./working/demo_run"
config.CACHE_DIR = os.path.join(config.WORKING_DIR, "cache")
config.CHECKPOINT_DIR = os.path.join(config.WORKING_DIR, "checkpoints")
config.METADATA_DIR = os.path.join(config.WORKING_DIR, "metadata")

# Ensure necessary directories exist
os.makedirs(config.CACHE_DIR, exist_ok=True)
os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(config.METADATA_DIR, exist_ok=True)

# Copy metadata files to the new metadata directory to avoid modifying original ones
# This ensures the library uses the correct file paths
original_metadata_dir = "./metadata"
for filename in ["train_metadata.csv", "val_metadata.csv", "test_metadata.csv"]:
    src = os.path.join(original_metadata_dir, filename)
    dst = os.path.join(config.METADATA_DIR, filename)
    if os.path.exists(src):
        shutil.copy(src, dst)

# Update the paths in config module to point to the demo metadata
config.TRAIN_METADATA_PATH = os.path.join(config.METADATA_DIR, "train_metadata.csv")
config.VAL_METADATA_PATH = os.path.join(config.METADATA_DIR, "val_metadata.csv")
config.TEST_METADATA_PATH = os.path.join(config.METADATA_DIR, "test_metadata.csv")

# Import library components after configuration update
from library.utils import set_seed, compute_rmsle, StandardScaler
from library.data import CrystalGraphDataset, process_geometry
from library.model import MH_RA_CGN
from library.train import train_model


def demo_utils():
    """
    Demonstrates and verifies utility functions.
    """
    print("\n--- Demonstrating Utils ---")
    set_seed(42)

    # 1. Test compute_rmsle
    y_true = np.array([[1.0, 10.0], [2.0, 20.0]])
    y_pred = np.array([[1.1, 9.5], [1.9, 21.0]])
    rmsle = compute_rmsle(y_pred, y_true)
    print(f"Computed RMSLE: {rmsle:.4f}")

    # Manual calculation check
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    mse = np.mean((log_true - log_pred) ** 2, axis=0)
    expected_rmsle = np.mean(np.sqrt(mse))
    assert np.isclose(rmsle, expected_rmsle), "RMSLE calculation mismatch"
    print("RMSLE check passed.")

    # 2. Test StandardScaler
    data = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=torch.float32)
    scaler = StandardScaler()
    scaler.fit(data)

    transformed = scaler.transform(data)
    print(f"StandardScaler Mean: {scaler.mean}")
    print(f"StandardScaler Std: {scaler.std}")

    # Check properties: transformed mean should be approx 0
    assert torch.allclose(
        transformed.mean(dim=0), torch.zeros(2), atol=1e-6
    ), "Transformed mean not 0"

    # Check inverse transform
    inverse = scaler.inverse_transform(transformed)
    assert torch.allclose(data, inverse, atol=1e-5), "Inverse transform failed"
    print("StandardScaler inverse transform check passed.")


def demo_data():
    """
    Demonstrates and verifies data processing and dataset loading.
    """
    print("\n--- Demonstrating Data Processing ---")

    # 1. Test process_geometry on a single file
    # We read the metadata to get a valid file path
    df = pd.read_csv(config.TRAIN_METADATA_PATH)
    sample_file = df.iloc[0]["file_path"]
    print(f"Processing sample geometry: {sample_file}")

    z, edge_index, edge_attr = process_geometry(sample_file, config.CUTOFF_RADIUS)
    print(f"Node features shape: {z.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge attr shape: {edge_attr.shape}")

    # Basic assertions on graph structure
    assert z.dim() == 2 and z.shape[1] == 1, "Incorrect node feature shape"
    assert (
        edge_index.dim() == 2 and edge_index.shape[0] == 2
    ), "Incorrect edge index shape"
    assert edge_attr.dim() == 2 and edge_attr.shape[1] == 1, "Incorrect edge attr shape"
    print("Geometry processing check passed.")

    # 2. Test Dataset Loading
    # We use a temporary cache file for this demo
    print("Initializing CrystalGraphDataset (Debug Mode)...")
    temp_cache = os.path.join(config.CACHE_DIR, "demo_train_graphs.npz")
    if os.path.exists(temp_cache):
        os.remove(temp_cache)

    dataset = CrystalGraphDataset(
        config.TRAIN_METADATA_PATH,
        temp_cache,
        load_cached_data=False,  # Force processing from scratch
    )

    print(f"Dataset length: {len(dataset)}")
    # Verify we respected the DEBUG_DATA_SIZE
    assert (
        len(dataset) == config.DEBUG_DATA_SIZE
    ), f"Dataset size mismatch: expected {config.DEBUG_DATA_SIZE}, got {len(dataset)}"

    sample_data = dataset[0]
    print("Sample Data Object:", sample_data)
    assert sample_data.x is not None
    assert sample_data.edge_index is not None
    assert sample_data.y is not None
    print("Dataset loading check passed.")
    return dataset


def demo_model(dataset):
    """
    Demonstrates model instantiation and a forward pass.
    """
    print("\n--- Demonstrating Model ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MH_RA_CGN().to(device)
    print("Model initialized.")

    # Create a batch using PyG DataLoader
    from torch_geometric.loader import DataLoader

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader)).to(device)

    # Forward pass
    print("Running forward pass on a batch...")
    out = model(batch)
    print(f"Output shape: {out.shape}")

    # Verify output shape: (Batch_Size, 2 targets)
    assert out.shape == (4, 2), f"Expected output shape (4, 2), got {out.shape}"
    print("Model forward pass check passed.")


def demo_training():
    """
    Demonstrates the full training loop.
    """
    print("\n--- Demonstrating Training Loop ---")

    # Clean up any existing cache in the demo dir to ensure fresh run
    # This ensures train_model generates new cache files based on our debug config
    if os.path.exists(config.CACHE_DIR):
        shutil.rmtree(config.CACHE_DIR)
    os.makedirs(config.CACHE_DIR, exist_ok=True)

    # Run training
    # load_cached_data=False forces dataset creation (respecting DEBUG_MODE)
    model = train_model(load_cached_data=False)

    # Verify checkpoint creation
    best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Checkpoint successfully created at {best_model_path}")
    else:
        raise AssertionError("Checkpoint file was not created.")

    print("Training demonstration completed successfully.")


if __name__ == "__main__":
    try:
        demo_utils()
        dataset = demo_data()
        demo_model(dataset)
        demo_training()
        print("\nAll demonstrations passed!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # Raise to ensure non-zero exit code on failure
        raise e
