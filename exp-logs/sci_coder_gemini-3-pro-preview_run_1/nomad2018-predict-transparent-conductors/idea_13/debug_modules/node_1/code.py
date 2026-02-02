import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import (
    parse_xyz,
    get_lattice_params,
    get_cell_volume,
    cartesian_to_fractional,
    compute_pbc_distances,
    compute_local_potential,
    center_coordinates,
)
from library.data import get_dataloaders, process_dataset
from library.model import MCPDSModel
from library.train import Trainer, set_seed


def run_demo():
    print("Starting MC-PDS Library Demo...")

    # 1. Setup Configuration for Demo
    # We modify the Config class attributes directly for this run to ensure speed and isolation
    print("\n[1] Configuring for Demo Run")
    Config.EPOCHS = 2  # Reduced from 200 for speed
    Config.WORKING_DIR = "./working/demo_execution"
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "cache", "train_data.npz")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "cache", "val_data.npz")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "cache", "test_data.npz")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pt")
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Ensure directories exist
    os.makedirs(os.path.dirname(Config.TRAIN_CACHE), exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    set_seed(Config.SEED)
    Config.print_config()

    # 2. Test Utility Functions
    print("\n[2] Testing Utility Functions")
    # We use a known file from the dataset description
    sample_xyz_path = os.path.join(Config.INPUT_DIR, "train/1/geometry.xyz")

    # Fallback to test/1/geometry.xyz if train/1 doesn't exist (though it should based on metadata)
    if not os.path.exists(sample_xyz_path):
        sample_xyz_path = os.path.join(Config.INPUT_DIR, "test/1/geometry.xyz")

    if os.path.exists(sample_xyz_path):
        print(f"Parsing {sample_xyz_path}...")
        lattice, types, coords = parse_xyz(sample_xyz_path)

        print(f"  Lattice shape: {lattice.shape}")
        print(f"  Number of atoms: {len(types)}")
        print(f"  Coordinates shape: {coords.shape}")

        assert lattice.shape == (3, 3), "Lattice matrix must be 3x3"
        assert coords.shape == (len(types), 3), "Coordinates must be Nx3"

        lengths, angles = get_lattice_params(lattice)
        print(f"  Lattice lengths: {lengths}")
        print(f"  Lattice angles: {angles}")
        assert len(lengths) == 3 and len(angles) == 3

        vol = get_cell_volume(lattice)
        print(f"  Cell Volume: {vol:.4f}")
        assert vol > 0, "Volume must be positive"

        centered = center_coordinates(coords)
        assert np.allclose(centered.mean(axis=0), 0, atol=1e-5), "Centering failed"

        frac = cartesian_to_fractional(coords, lattice)
        print(f"  Fractional coords shape: {frac.shape}")

        dist_matrix = compute_pbc_distances(coords, lattice)
        print(f"  Distance matrix shape: {dist_matrix.shape}")
        assert dist_matrix.shape == (len(types), len(types))
        # Diagonal should be 0 (distance to self)
        assert np.allclose(
            np.diag(dist_matrix), 0
        ), "Diagonal of distance matrix should be 0"

        potential = compute_local_potential(dist_matrix)
        print(f"  Local potential shape: {potential.shape}")
        assert potential.shape == (len(types),)

        print("  -> Utils verification passed.")
    else:
        print(
            f"  Warning: Sample file {sample_xyz_path} not found. Skipping specific file utils test."
        )

    # 3. Test Data Loading
    print("\n[3] Testing Data Loading")
    # This will process the dataset and save to the new cache locations defined above
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    atomic_feats = batch["atomic_features"]
    global_feats = batch["global_features"]
    mask = batch["mask"]
    targets = batch["targets"]
    ids = batch["ids"]

    print(
        f"  Batch Atomic Features Shape: {atomic_feats.shape}"
    )  # (B, MaxAtoms, FeatDim)
    print(f"  Batch Global Features Shape: {global_feats.shape}")  # (B, GlobalDim)
    print(f"  Batch Mask Shape: {mask.shape}")
    print(f"  Batch Targets Shape: {targets.shape}")

    assert (
        atomic_feats.shape[2] == Config.ATOMIC_INPUT_DIM
    ), f"Expected atomic dim {Config.ATOMIC_INPUT_DIM}, got {atomic_feats.shape[2]}"
    assert (
        global_feats.shape[1] == Config.GLOBAL_INPUT_DIM
    ), f"Expected global dim {Config.GLOBAL_INPUT_DIM}, got {global_feats.shape[1]}"
    assert targets.shape[1] == 2, "Targets should have 2 columns"

    print("  -> Data Loading verification passed.")

    # 4. Test Model Architecture
    print("\n[4] Testing Model Architecture")
    model = MCPDSModel()
    model.to(Config.DEVICE)

    # Forward pass with the batch fetched earlier
    atomic_feats = atomic_feats.to(Config.DEVICE)
    global_feats = global_feats.to(Config.DEVICE)
    mask = mask.to(Config.DEVICE)

    outputs = model(atomic_feats, global_feats, mask)
    print(f"  Model Output Shape: {outputs.shape}")

    assert outputs.shape == (atomic_feats.shape[0], 2), "Model output shape mismatch"
    print("  -> Model verification passed.")

    # 5. Test Training Loop
    print("\n[5] Testing Training Loop")
    trainer = Trainer(model)

    # Run fit (using the reduced EPOCHS=2)
    trainer.fit(
        train_loader, val_loader, epochs=Config.EPOCHS, patience=Config.PATIENCE
    )

    # Check if model checkpoint was saved
    if os.path.exists(Config.MODEL_PATH):
        print(f"  -> Model checkpoint successfully saved at {Config.MODEL_PATH}")
    else:
        # It's possible validation loss didn't improve in 2 epochs if initialized well,
        # but unlikely. If it fails, we force save for the next step.
        print(
            "  -> Warning: Best model not saved by Trainer (maybe no improvement). Saving current state manually for demo."
        )
        torch.save(model.state_dict(), Config.MODEL_PATH)

    # 6. Test Submission Generation
    print("\n[6] Testing Submission Generation")
    trainer.generate_submission(test_loader, Config.SUBMISSION_PATH)

    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"  -> Submission file generated at {Config.SUBMISSION_PATH}")
        df = pd.read_csv(Config.SUBMISSION_PATH)
        print("  Submission Head:")
        print(df.head())

        # Verify submission format
        assert len(df) == 240, f"Expected 240 predictions, got {len(df)}"
        assert "id" in df.columns
        assert "formation_energy_ev_natom" in df.columns
        assert "bandgap_energy_ev" in df.columns
        print("  -> Submission format verified.")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
