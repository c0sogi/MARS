import os
import sys
import numpy as np
import torch
import pandas as pd

# Import from the provided library files
import library.config as config
from library.features import (
    parse_xyz,
    get_pbc_distances,
    get_lce_features,
    get_global_features,
    one_hot_encode_atoms,
    process_dataset,
)
from library.data import get_data_loaders, MaterialDataset, collate_fn
from library.architecture import LCEWDS
from library.engine import run_training, generate_submission, set_seed


def verify_feature_extraction():
    """
    Verifies the correctness of feature extraction functions using a sample file.
    """
    print("Verifying feature extraction logic...")

    # Pick a sample file from the training set
    sample_id = 1
    rel_path = f"train/{sample_id}/geometry.xyz"
    full_path = os.path.join(config.INPUT_DIR, rel_path)

    if not os.path.exists(full_path):
        print(f"Sample file {full_path} not found. Skipping specific file check.")
        return

    # 1. Parse XYZ
    lattice, atoms, coords = parse_xyz(rel_path)
    assert lattice.shape == (3, 3), "Lattice shape mismatch"
    assert len(atoms) == len(coords), "Atom count mismatch"
    print(f"  Parsed {len(atoms)} atoms from {rel_path}")

    # 2. PBC Distances
    dist_matrix = get_pbc_distances(coords, lattice)
    assert dist_matrix.shape == (
        len(atoms),
        len(atoms),
    ), "Distance matrix shape mismatch"
    # Diagonal should be close to 0
    assert np.allclose(np.diag(dist_matrix), 0), "Distance matrix diagonal non-zero"
    print("  PBC Distance calculation successful.")

    # 3. LCE Features
    lce_feats, nn_dists = get_lce_features(atoms, dist_matrix)
    # LCE features: [avg_mass, avg_radius, avg_neg] -> 3 dims
    assert lce_feats.shape == (len(atoms), 3), "LCE features shape mismatch"
    assert nn_dists.shape == (len(atoms), 1), "NN dists shape mismatch"
    print("  LCE Feature calculation successful.")

    # 4. Global Features
    global_feats = get_global_features(lattice, atoms)
    # Global dims: 3(lengths) + 3(angles) + 1(vol) + 1(dens) + 3(stoich) + 1(count) = 12
    assert global_feats.shape == (
        12,
    ), f"Global features shape mismatch: {global_feats.shape}"
    print("  Global Feature calculation successful.")

    # 5. One-Hot Encoding
    one_hot = one_hot_encode_atoms(atoms)
    assert one_hot.shape == (len(atoms), 4), "One-hot encoding shape mismatch"
    print("  One-Hot encoding successful.")


def verify_model_architecture():
    """
    Verifies the model forward pass with dummy tensors.
    """
    print("\nVerifying model architecture...")

    # Define dummy input dimensions based on config
    n_atoms = 50
    batch_size = 4
    atomic_dim = config.ATOMIC_FEATURE_DIM
    global_dim = config.GLOBAL_FEATURE_DIM

    # Create dummy data
    atomic_feats = torch.randn(n_atoms, atomic_dim)
    global_feats = torch.randn(batch_size, global_dim)

    # Create a dummy batch index map (assign atoms to 4 samples)
    # e.g., 10, 15, 15, 10 atoms per sample
    batch_indices = torch.cat(
        [
            torch.full((10,), 0),
            torch.full((15,), 1),
            torch.full((15,), 2),
            torch.full((10,), 3),
        ]
    ).long()

    assert batch_indices.size(0) == n_atoms

    # Instantiate model
    model = LCEWDS()
    model.eval()

    # Forward pass
    with torch.no_grad():
        output = model(atomic_feats, global_feats, batch_indices)

    # Check output shape: (batch_size, output_dim) -> (4, 2)
    assert output.shape == (
        batch_size,
        2,
    ), f"Model output shape mismatch: {output.shape}"
    print(f"  Model forward pass successful. Output shape: {output.shape}")


def run_short_training_pipeline():
    """
    Runs the training pipeline with reduced hyperparameters for demonstration speed.
    """
    print("\nRunning short training pipeline...")

    # Modify hyperparameters for speed
    config.TRAINING_PARAMS["epochs"] = 2
    config.TRAINING_PARAMS["batch_size"] = 16
    config.TRAINING_PARAMS["patience"] = 1

    # Ensure working directory exists
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Run training (this handles data loading, processing, and training loop)
    # Note: This will process the dataset if not already cached in WORKING_DIR.
    # Given the dataset size (~2000 samples), this is feasible within the time limit.
    trained_model, test_loader = run_training(load_cached_data=True)

    print("  Training pipeline finished.")

    # Generate submission
    print("Generating submission...")
    generate_submission(trained_model, test_loader)

    # Verify submission file exists
    if os.path.exists(config.SUBMISSION_PATH):
        df = pd.read_csv(config.SUBMISSION_PATH)
        print(f"  Submission file created with {len(df)} rows.")
        assert len(df) > 0, "Submission file is empty"
        assert "id" in df.columns, "Missing 'id' column"
        assert (
            "formation_energy_ev_natom" in df.columns
        ), "Missing formation energy column"
        assert "bandgap_energy_ev" in df.columns, "Missing bandgap energy column"
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    # Set seed for reproducibility
    set_seed(config.SEED)

    try:
        # 1. Verify Feature Extraction Logic
        verify_feature_extraction()

        # 2. Verify Model Architecture
        verify_model_architecture()

        # 3. Run Training and Submission Pipeline
        run_short_training_pipeline()

        print("\nAll demonstration steps completed successfully.")

    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        raise e
