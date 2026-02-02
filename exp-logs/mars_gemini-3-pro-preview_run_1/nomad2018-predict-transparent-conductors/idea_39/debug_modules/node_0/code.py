import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import library components
from library.config import Config
from library.utils import set_seed, StandardScaler, log_transform, inverse_log_transform
from library.geometry import GeometryProcessor, process_dataset
from library.data import (
    MaterialDataset,
    collate_batch,
    get_train_val_loaders,
    get_test_loader,
)
from library.model import SCC_WDS_Net
from library.engine import Engine

# Suppress warnings
warnings.filterwarnings("ignore")


def setup_demo_environment():
    """
    Sets up a small environment in ./working/demo_execution to run the demo quickly.
    Creates mini versions of metadata files.
    """
    print("Setting up demo environment...")

    # Define demo directories
    demo_dir = "./working/demo_execution"
    demo_meta_dir = "./working/demo_metadata"
    demo_cache_dir = "./working/demo_cache"
    demo_submission_dir = "./working/demo_submission"

    os.makedirs(demo_dir, exist_ok=True)
    os.makedirs(demo_meta_dir, exist_ok=True)
    os.makedirs(demo_cache_dir, exist_ok=True)
    os.makedirs(demo_submission_dir, exist_ok=True)

    # Create mini metadata files (first 50 samples)
    # We read from the original metadata provided in the environment
    train_df = pd.read_csv(Config.TRAIN_METADATA).head(50)
    val_df = pd.read_csv(Config.VAL_METADATA).head(50)
    test_df = pd.read_csv(Config.TEST_METADATA).head(50)

    mini_train_path = os.path.join(demo_meta_dir, "train.csv")
    mini_val_path = os.path.join(demo_meta_dir, "val.csv")
    mini_test_path = os.path.join(demo_meta_dir, "test.csv")

    train_df.to_csv(mini_train_path, index=False)
    val_df.to_csv(mini_val_path, index=False)
    test_df.to_csv(mini_test_path, index=False)

    # Override Config paths to point to demo environment
    Config.METADATA_DIR = demo_meta_dir
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_submission_dir

    Config.TRAIN_METADATA = mini_train_path
    Config.VAL_METADATA = mini_val_path
    Config.TEST_METADATA = mini_test_path

    Config.TRAIN_CACHE = os.path.join(demo_dir, "train_data.npz")
    Config.VAL_CACHE = os.path.join(demo_dir, "val_data.npz")
    Config.TEST_CACHE = os.path.join(demo_dir, "test_data.npz")
    Config.SCALERS_CACHE = os.path.join(demo_dir, "scalers.npz")
    Config.MODEL_PATH = os.path.join(demo_dir, "best_model.pt")
    Config.SUBMISSION_FILE = os.path.join(demo_submission_dir, "demo_submission.csv")

    # Adjust hyperparameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.HIDDEN_DIM = 64  # Smaller model for demo
    Config.GLOBAL_HIDDEN_DIM = 32

    print("Demo environment configured.")
    Config.print_config()


def verify_utils():
    print("\n--- Verifying Utils ---")

    # Test Seed
    set_seed(42)
    r1 = np.random.rand()
    set_seed(42)
    r2 = np.random.rand()
    assert r1 == r2, "Seed setting failed for numpy"
    print("Random seed verification passed.")

    # Test StandardScaler
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)

    expected_mean = np.array([3.0, 4.0])
    expected_std = np.std(data, axis=0)

    assert np.allclose(scaler.mean, expected_mean), "Scaler mean calculation incorrect"
    assert np.allclose(
        np.mean(scaled, axis=0), 0.0, atol=1e-7
    ), "Scaled data mean is not 0"
    assert np.allclose(
        np.std(scaled, axis=0), 1.0, atol=1e-7
    ), "Scaled data std is not 1"

    inversed = scaler.inverse_transform(scaled)
    assert np.allclose(data, inversed), "Inverse transform failed"

    # Test Save/Load
    scaler_path = os.path.join(Config.WORKING_DIR, "test_scaler.npz")
    scaler.save(scaler_path)
    loaded_scaler = StandardScaler()
    loaded_scaler.load(scaler_path)
    assert np.allclose(
        scaler.mean, loaded_scaler.mean
    ), "Scaler save/load mean mismatch"
    print("StandardScaler verification passed.")

    # Test Log Transform
    vals = np.array([0.0, 1.0, 10.0])
    log_vals = log_transform(vals)
    inv_vals = inverse_log_transform(log_vals)
    assert np.allclose(vals, inv_vals), "Log transform/inverse failed"
    print("Log transform verification passed.")


def verify_geometry_processor():
    print("\n--- Verifying GeometryProcessor ---")
    processor = GeometryProcessor()

    # Pick a sample file from the training set
    sample_id = 1
    sample_file = os.path.join(Config.INPUT_DIR, f"train/{sample_id}/geometry.xyz")

    if os.path.exists(sample_file):
        print(f"Processing sample file: {sample_file}")
        atomic_feats, global_feats = processor.process_file(sample_file)

        # Check dimensions
        # Atomic features: OneHot(4) + Coords(3) + d_min(1) + Context(4) = 12
        assert (
            atomic_feats.shape[1] == Config.ATOMIC_FEATURE_DIM
        ), f"Atomic feature dim mismatch. Expected {Config.ATOMIC_FEATURE_DIM}, got {atomic_feats.shape[1]}"

        # Global features: Lengths(3) + Angles(3) + Vol(1) + Dens(1) + Stoich(3) + NumAtoms(1) = 12
        assert (
            global_feats.shape[0] == Config.GLOBAL_FEATURE_DIM
        ), f"Global feature dim mismatch. Expected {Config.GLOBAL_FEATURE_DIM}, got {global_feats.shape[0]}"

        print(f"Atomic features shape: {atomic_feats.shape}")
        print(f"Global features shape: {global_feats.shape}")
        print("GeometryProcessor verification passed.")
    else:
        print(f"Sample file {sample_file} not found. Skipping specific file check.")


def verify_data_pipeline():
    print("\n--- Verifying Data Pipeline ---")

    # We use the get_train_val_loaders function which handles processing, scaling, and caching
    # Since we set up the demo environment, this will use the mini datasets
    train_loader, val_loader = get_train_val_loaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=False  # Force processing
    )

    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")

    # Check one batch
    batch = next(iter(train_loader))

    # Verify batch keys
    expected_keys = {
        "atomic_features",
        "global_features",
        "target",
        "id",
        "batch_index",
    }
    assert all(k in batch for k in expected_keys), "Missing keys in batch"

    # Verify shapes
    # atomic_features: (Total_Atoms_In_Batch, Feature_Dim)
    # global_features: (Batch_Size, Global_Dim)
    # targets: (Batch_Size, 2)
    # batch_index: (Total_Atoms_In_Batch,)

    n_batch = batch["target"].shape[0]
    n_atoms = batch["atomic_features"].shape[0]

    assert batch["global_features"].shape == (n_batch, Config.GLOBAL_FEATURE_DIM)
    assert batch["target"].shape == (n_batch, 2)
    assert batch["batch_index"].shape == (n_atoms,)
    assert batch["atomic_features"].shape[1] == Config.ATOMIC_FEATURE_DIM

    print("Data Pipeline verification passed.")
    return train_loader, val_loader


def verify_model():
    print("\n--- Verifying Model ---")

    model = SCC_WDS_Net()
    # Move to CPU for verification
    model.to("cpu")

    # Create dummy data
    batch_size = 4
    num_atoms_per_crystal = 10
    total_atoms = batch_size * num_atoms_per_crystal

    atomic_feats = torch.randn(total_atoms, Config.ATOMIC_FEATURE_DIM)
    global_feats = torch.randn(batch_size, Config.GLOBAL_FEATURE_DIM)

    # Create batch index: [0,0...0, 1,1...1, ..., 3,3...3]
    batch_index = torch.repeat_interleave(
        torch.arange(batch_size), num_atoms_per_crystal
    )

    # Forward pass
    output = model(atomic_feats, global_feats, batch_index)

    # Check output shape: (Batch_Size, 2)
    assert output.shape == (
        batch_size,
        2,
    ), f"Model output shape mismatch. Expected {(batch_size, 2)}, got {output.shape}"

    print("Model forward pass successful.")
    print(f"Output shape: {output.shape}")


def run_full_engine_cycle():
    print("\n--- Running Full Engine Cycle ---")

    engine = Engine()

    # 1. Train
    print("Running training...")
    # We force load_cached_data=True because we just generated the cache in verify_data_pipeline
    # But wait, verify_data_pipeline returned loaders but didn't necessarily save to the exact path
    # if the internal logic didn't trigger save.
    # Actually, process_dataset saves to cache if cache_path is provided.
    # In setup_demo_environment, we set Config.TRAIN_CACHE.
    # So the data should be cached now.

    best_loss = engine.run_training(
        batch_size=Config.BATCH_SIZE,
        epochs=Config.EPOCHS,
        patience=2,  # Short patience for demo
        load_cached_data=True,
    )

    print(f"Training finished with best validation loss: {best_loss:.4f}")

    # 2. Predict
    print("Running prediction...")
    submission_df = engine.predict(
        batch_size=Config.BATCH_SIZE,
        load_cached_data=False,  # Process test data from scratch
    )

    assert not submission_df.empty, "Submission DataFrame is empty"
    assert "formation_energy_ev_natom" in submission_df.columns
    assert "bandgap_energy_ev" in submission_df.columns

    print("Engine cycle completed successfully.")


if __name__ == "__main__":
    print("Starting Demonstration Script...")

    # 1. Setup Environment (Mock Data)
    setup_demo_environment()

    # 2. Verify Utilities
    verify_utils()

    # 3. Verify Geometry Processing
    verify_geometry_processor()

    # 4. Verify Data Loading (and generate cache for training)
    verify_data_pipeline()

    # 5. Verify Model Architecture
    verify_model()

    # 6. Run Engine (Train & Inference)
    run_full_engine_cycle()

    print("\nAll demonstrations and verifications passed!")
