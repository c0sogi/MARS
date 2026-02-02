import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import library components
from library.config import Config
from library.geometry_utils import parse_xyz, calculate_atomic_features
from library.data_loader import process_data, MaterialDataset, collate_fn
from library.model import HCCRDSModel
from library.trainer import run_training, generate_submission


def setup_demo_config():
    """
    Overrides Config parameters to ensure the demo runs quickly and uses a separate working directory.
    """
    print("Setting up demo configuration...")

    # Use a specific directory for demo outputs to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update cache paths to point to the demo directory
    # This ensures we don't load full-dataset caches and force re-processing of the small subset
    Config.TRAIN_DATA_CACHE = os.path.join(Config.WORKING_DIR, "train_data.npz")
    Config.VAL_DATA_CACHE = os.path.join(Config.WORKING_DIR, "val_data.npz")
    Config.TEST_DATA_CACHE = os.path.join(Config.WORKING_DIR, "test_data.npz")
    Config.SCALERS_PATH = os.path.join(Config.WORKING_DIR, "scalers.npz")
    Config.MODEL_CHECKPOINT = os.path.join(Config.WORKING_DIR, "demo_model.pt")

    # Update submission directory
    Config.SUBMISSION_DIR = "./working/demo_submission"
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Reduce training parameters for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    print(f"Working directory set to: {Config.WORKING_DIR}")
    print(f"Num Epochs: {Config.NUM_EPOCHS}, Batch Size: {Config.BATCH_SIZE}")


def test_geometry_utils():
    """
    Demonstrates and verifies parsing of XYZ files and feature calculation.
    """
    print("\n--- Testing Geometry Utils ---")

    # Pick a sample file
    sample_id = 1
    xyz_path = os.path.join(Config.INPUT_DIR, f"train/{sample_id}/geometry.xyz")

    if not os.path.exists(xyz_path):
        print(f"Sample file {xyz_path} not found. Skipping geometry test.")
        return

    # 1. Parse XYZ
    lattice_vectors, atom_types, atom_coords = parse_xyz(xyz_path)

    print(f"Parsed {len(atom_types)} atoms from {xyz_path}")
    assert lattice_vectors.shape == (3, 3), "Lattice vectors should be 3x3"
    assert (
        len(atom_types) == atom_coords.shape[0]
    ), "Mismatch between atom types and coords"
    assert atom_coords.shape[1] == 3, "Coordinates should be 3D"

    # 2. Calculate Atomic Features
    features = calculate_atomic_features(atom_types, atom_coords, lattice_vectors)

    # Expected feature dim: 4 (one-hot) + 3 (coords) + 4 (recip) + 1 (packing) = 12
    expected_dim = 12
    print(f"Calculated atomic features shape: {features.shape}")

    assert features.shape[0] == len(atom_types), "Feature rows must match num atoms"
    assert (
        features.shape[1] == expected_dim
    ), f"Feature dim should be {expected_dim}, got {features.shape[1]}"
    assert not np.isnan(features).any(), "Features contain NaNs"

    print("Geometry utils test passed.")


def test_data_loader():
    """
    Demonstrates data loading and processing pipeline.
    """
    print("\n--- Testing Data Loader ---")

    # Process a small subset of data
    # This will trigger feature extraction, scaling, and caching in the demo directory
    sample_size = 20
    train_loader, val_loader, test_loader = process_data(
        load_cached_data=True, sample_size=sample_size
    )

    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")

    # Check one batch
    batch = next(iter(train_loader))

    atomic_feats = batch["atomic_features"]
    batch_indices = batch["batch_indices"]
    global_feats = batch["global_features"]
    targets = batch["targets"]
    ids = batch["id"]

    print(f"Batch keys: {batch.keys()}")
    print(f"Atomic features shape: {atomic_feats.shape}")
    print(f"Global features shape: {global_feats.shape}")
    print(f"Targets shape: {targets.shape}")

    # Assertions
    assert atomic_feats.ndim == 2 and atomic_feats.shape[1] == Config.ATOMIC_INPUT_DIM
    assert global_feats.ndim == 2 and global_feats.shape[1] == Config.GLOBAL_INPUT_DIM
    assert targets.ndim == 2 and targets.shape[1] == Config.OUTPUT_DIM
    assert (
        len(batch_indices) == atomic_feats.shape[0]
    ), "Batch indices must match atomic features length"

    print("Data loader test passed.")


def test_model_architecture():
    """
    Demonstrates model instantiation and a forward pass with dummy data.
    """
    print("\n--- Testing Model Architecture ---")

    model = HCCRDSModel()
    model.eval()

    # Create dummy data simulating a batch of 2 graphs
    # Graph 1: 5 atoms, Graph 2: 3 atoms. Total 8 atoms.
    n_atoms_1 = 5
    n_atoms_2 = 3
    total_atoms = n_atoms_1 + n_atoms_2
    batch_size = 2

    dummy_atomic = torch.randn(total_atoms, Config.ATOMIC_INPUT_DIM)
    dummy_global = torch.randn(batch_size, Config.GLOBAL_INPUT_DIM)

    # Batch indices: [0, 0, 0, 0, 0, 1, 1, 1]
    dummy_indices = torch.cat(
        [
            torch.zeros(n_atoms_1, dtype=torch.long),
            torch.ones(n_atoms_2, dtype=torch.long),
        ]
    )

    print("Running forward pass with dummy data...")
    with torch.no_grad():
        output = model(dummy_atomic, dummy_indices, dummy_global)

    print(f"Output shape: {output.shape}")

    assert output.shape == (
        batch_size,
        Config.OUTPUT_DIM,
    ), f"Expected output shape {(batch_size, Config.OUTPUT_DIM)}, got {output.shape}"

    print("Model architecture test passed.")


def test_training_loop():
    """
    Demonstrates the full training loop using the library's trainer module.
    """
    print("\n--- Testing Training Loop ---")

    # Run training on a small sample
    # This uses the config overrides set in setup_demo_config
    sample_size = 20
    model = run_training(sample_size=sample_size)

    assert isinstance(
        model, torch.nn.Module
    ), "run_training should return a PyTorch model"
    assert os.path.exists(Config.MODEL_CHECKPOINT), "Model checkpoint should be saved"

    print("Training loop test passed.")
    return model


def test_submission_generation(model):
    """
    Demonstrates submission file generation.
    """
    print("\n--- Testing Submission Generation ---")

    # Generate submission using the trained model
    # Note: process_data inside generate_submission will load the test cache created during run_training
    # or recompute if needed.
    generate_submission(model)

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created"

    df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission file loaded. Shape: {df.shape}")

    required_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
    for col in required_cols:
        assert col in df.columns, f"Missing column {col} in submission"

    # Since we used a sample size for training, the test set might still be full size
    # (process_data processes full test set by default unless sample_size is passed,
    # but run_training passes sample_size. However, generate_submission calls process_data
    # without sample_size, but if cache exists, it loads cache.
    # In setup_demo_config, we pointed cache to a demo dir.
    # During run_training(sample_size=20), the test set was also sliced to 20.
    # So we expect 20 rows if cache was saved and reloaded.)

    # Let's check if it's not empty
    assert len(df) > 0, "Submission dataframe is empty"

    print("Submission generation test passed.")


if __name__ == "__main__":
    # 1. Setup
    setup_demo_config()

    # 2. Geometry Utils
    test_geometry_utils()

    # 3. Data Loader
    test_data_loader()

    # 4. Model
    test_model_architecture()

    # 5. Training
    trained_model = test_training_loop()

    # 6. Submission
    test_submission_generation(trained_model)

    print("\nAll tests completed successfully.")
