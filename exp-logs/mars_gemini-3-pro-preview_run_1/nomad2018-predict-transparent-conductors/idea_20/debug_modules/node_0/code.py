import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.geometry_utils import (
    parse_xyz,
    get_cell_volume,
    get_atomic_density,
    compute_local_anisotropy,
    get_centered_coordinates,
    get_pbc_displacement,
)
from library.data_loader import get_dataloaders
from library.model import LAWDS
from library.trainer import train_step, validate_step, generate_submission

# Define demo directory globally for use in functions
demo_dir = "./working/demo_execution"


def setup_demo_environment():
    """
    Sets up a temporary environment with a subset of data for quick demonstration.
    """
    print("Setting up demo environment...")

    # Define working directories
    os.makedirs(demo_dir, exist_ok=True)

    # Load original metadata
    train_df = pd.read_csv("./metadata/train.csv")
    val_df = pd.read_csv("./metadata/val.csv")
    test_df = pd.read_csv("./metadata/test.csv")

    # Create subsets (e.g., 50 samples for train, 10 for val/test)
    # This ensures the data loader processes quickly
    mini_train = train_df.head(50).copy()
    mini_val = val_df.head(10).copy()
    mini_test = test_df.head(10).copy()

    # Save mini metadata
    mini_train_path = os.path.join(demo_dir, "mini_train.csv")
    mini_val_path = os.path.join(demo_dir, "mini_val.csv")
    mini_test_path = os.path.join(demo_dir, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # Override Config paths to point to these mini datasets
    Config.TRAIN_META_PATH = mini_train_path
    Config.VAL_META_PATH = mini_val_path
    Config.TEST_META_PATH = mini_test_path

    # Override Cache and Submission paths
    Config.CACHE_DIR = "./working/demo_cache"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Override Training Hyperparameters for speed
    Config.BATCH_SIZE = 4
    Config.MAX_EPOCHS = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print("Demo environment setup complete.")


def test_geometry_utils():
    """
    Verifies the functionality of geometry utility functions.
    """
    print("\nTesting geometry_utils...")

    # Pick a sample file from the input directory
    # We know train/1/geometry.xyz exists based on the problem description
    sample_file = "./input/train/1/geometry.xyz"
    if not os.path.exists(sample_file):
        # Fallback to finding one
        for root, dirs, files in os.walk("./input/train"):
            if "geometry.xyz" in files:
                sample_file = os.path.join(root, "geometry.xyz")
                break

    print(f"Parsing {sample_file}...")
    lattice_vectors, atom_types, coords = parse_xyz(sample_file)

    # Check shapes
    assert lattice_vectors.shape == (3, 3), "Lattice vectors shape mismatch"
    assert coords.shape[0] == len(
        atom_types
    ), "Coordinate count mismatch with atom types"
    assert coords.shape[1] == 3, "Coordinates must be 3D"

    # Test Volume
    vol = get_cell_volume(lattice_vectors)
    assert vol > 0, "Volume must be positive"
    print(f"Cell Volume: {vol:.4f}")

    # Test Density
    density = get_atomic_density(len(atom_types), vol)
    assert density > 0, "Density must be positive"

    # Test Centered Coordinates
    centered = get_centered_coordinates(coords, lattice_vectors)
    assert centered.shape == coords.shape

    # Test Local Anisotropy (this is computationally intensive part usually)
    print("Computing local anisotropy...")
    eigenvalues, nn_dists = compute_local_anisotropy(
        coords, lattice_vectors, k_neighbors=Config.K_NEIGHBORS
    )

    assert eigenvalues.shape == (len(atom_types), 3), "Eigenvalues shape mismatch"
    assert nn_dists.shape == (len(atom_types), 1), "NN dists shape mismatch"

    # Check that eigenvalues are sorted ascending as per implementation
    assert np.all(eigenvalues[:, 0] <= eigenvalues[:, 1]), "Eigenvalues not sorted"
    assert np.all(eigenvalues[:, 1] <= eigenvalues[:, 2]), "Eigenvalues not sorted"

    print("Geometry utils verification passed.")


def test_data_loading_and_model():
    """
    Verifies data loading, model instantiation, and a training step.
    """
    print("\nTesting Data Loading and Model...")

    # 1. Get Dataloaders (this will trigger processing of the mini datasets)
    # We force reload to ensure we use the mini datasets we just created
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")

    # 2. Inspect a batch
    atom_x, batch_indices, global_x, targets, ids = next(iter(train_loader))

    print(
        f"Batch Atom Features Shape: {atom_x.shape}"
    )  # [Total_Atoms_In_Batch, ATOMIC_INPUT_DIM]
    print(f"Batch Indices Shape: {batch_indices.shape}")  # [Total_Atoms_In_Batch]
    print(
        f"Batch Global Features Shape: {global_x.shape}"
    )  # [Batch_Size, GLOBAL_INPUT_DIM]
    print(f"Batch Targets Shape: {targets.shape}")  # [Batch_Size, 2]

    # Assert dimensions match Config
    assert (
        atom_x.shape[1] == Config.ATOMIC_INPUT_DIM
    ), f"Expected atomic dim {Config.ATOMIC_INPUT_DIM}, got {atom_x.shape[1]}"
    assert (
        global_x.shape[1] == Config.GLOBAL_INPUT_DIM
    ), f"Expected global dim {Config.GLOBAL_INPUT_DIM}, got {global_x.shape[1]}"
    assert (
        targets.shape[1] == Config.OUTPUT_DIM
    ), f"Expected output dim {Config.OUTPUT_DIM}, got {targets.shape[1]}"

    # 3. Instantiate Model
    device = Config.DEVICE
    model = LAWDS().to(device)
    print(f"Model instantiated on {device}")

    # 4. Run a Forward Pass
    atom_x = atom_x.to(device)
    batch_indices = batch_indices.to(device)
    global_x = global_x.to(device)

    outputs = model(atom_x, batch_indices, global_x)
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.OUTPUT_DIM,
    ), "Model output shape mismatch"
    print("Forward pass successful.")

    # 5. Run Training Loop (Short)
    print("\nRunning Training Loop (2 Epochs)...")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(Config.MAX_EPOCHS):
        train_loss = train_step(model, train_loader, optimizer, criterion, device)
        val_loss = validate_step(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        # Basic sanity check: loss should be a number
        assert not np.isnan(train_loss), "Training loss is NaN"
        assert not np.isnan(val_loss), "Validation loss is NaN"

    # 6. Generate Submission
    print("\nGenerating Submission...")
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Verify submission format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission head:")
    print(sub_df.head())
    assert list(sub_df.columns) == [
        "id",
        "formation_energy_ev_natom",
        "bandgap_energy_ev",
    ], "Incorrect submission columns"
    assert len(sub_df) == 10, "Submission should have 10 rows (from mini_test)"

    # Save model for completeness
    torch.save(model.state_dict(), os.path.join(demo_dir, "demo_model.pt"))
    print("Demo model saved.")


if __name__ == "__main__":
    # Fix seeds
    Config.set_seed(42)

    # Setup
    setup_demo_environment()

    # Verify Geometry Utils
    test_geometry_utils()

    # Verify Data Loading, Model, Training, and Submission
    test_data_loading_and_model()

    print("\nAll demonstrations completed successfully.")
