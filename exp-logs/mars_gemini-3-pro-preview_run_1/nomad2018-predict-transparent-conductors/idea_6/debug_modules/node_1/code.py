import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import from the provided library
from library.config import CONFIG, PHYSICAL_CONSTANTS, ATOM_MAP
from library.geometry import parse_xyz, calculate_cell_volume, compute_local_fingerprint
from library.features import process_dataset
from library.data import get_data_loaders, CrystalDataset, collate_fn
from library.model import PADSDS
from library.engine import Engine


def run_demo():
    print("--- Starting Demo Execution ---")

    # 1. Setup Configuration for Demo
    # We override the global CONFIG to ensure the demo runs quickly.
    CONFIG["epochs"] = 1
    CONFIG["batch_size"] = 16
    CONFIG["patience"] = 1
    # Ensure we use a directory we can write to
    demo_save_dir = "./working/demo_execution"
    os.makedirs(demo_save_dir, exist_ok=True)

    print(f"Configuration: {CONFIG}")

    # 2. Demonstrate Geometry Parsing and Calculation
    print("\n--- Testing Library: Geometry ---")
    # Pick a sample file
    sample_file_path = "./input/train/1/geometry.xyz"
    if not os.path.exists(sample_file_path):
        raise FileNotFoundError(f"Sample file {sample_file_path} not found.")

    atoms, coords, lattice_vectors = parse_xyz(sample_file_path)
    print(f"Parsed {len(atoms)} atoms from {sample_file_path}")

    # Validation
    assert len(atoms) > 0, "No atoms parsed"
    assert coords.shape == (len(atoms), 3), "Coordinates shape mismatch"
    assert lattice_vectors.shape == (3, 3), "Lattice vectors shape mismatch"

    # Volume
    vol = calculate_cell_volume(lattice_vectors)
    print(f"Calculated Cell Volume: {vol:.4f}")
    assert vol > 0, "Volume must be positive"

    # Fingerprints
    # k=12 is default in config
    fingerprints = compute_local_fingerprint(
        coords, lattice_vectors, k=CONFIG["k_neighbors"]
    )
    print(f"Computed Fingerprints shape: {fingerprints.shape}")
    assert fingerprints.shape == (len(atoms), 3), "Fingerprint shape mismatch"
    # Check for non-negative distances
    assert np.all(fingerprints >= 0), "Distances in fingerprint must be non-negative"

    # 3. Demonstrate Feature Processing
    print("\n--- Testing Library: Features ---")
    # We will use a small subset of the metadata to demonstrate process_dataset without processing everything
    # However, get_data_loaders reads the files from disk directly.
    # To demonstrate process_dataset explicitly, we create a dummy dataframe.

    dummy_data = {
        "id": [1, 2],
        "file_path": ["train/1/geometry.xyz", "train/2/geometry.xyz"],
        "number_of_total_atoms": [40, 80],  # Dummy values
        "percent_atom_al": [0.25, 0.25],
        "percent_atom_ga": [0.25, 0.25],
        "percent_atom_in": [0.25, 0.25],
        "lattice_vector_1_ang": [10.0, 10.0],
        "lattice_vector_2_ang": [10.0, 10.0],
        "lattice_vector_3_ang": [10.0, 10.0],
        "lattice_angle_alpha_degree": [90.0, 90.0],
        "lattice_angle_beta_degree": [90.0, 90.0],
        "lattice_angle_gamma_degree": [90.0, 90.0],
        "formation_energy_ev_natom": [0.1, 0.2],
        "bandgap_energy_ev": [1.5, 2.0],
    }
    dummy_df = pd.DataFrame(dummy_data)

    # We use a unique cache name to avoid messing with the real training cache if possible,
    # though the library writes to ./working/idea_6. We'll let it write there.
    # Note: process_dataset prints to stdout.

    atomic_feats, global_feats, targets, ids = process_dataset(
        dummy_df, "./input", load_cached_data=False, cache_name="demo_dummy"
    )

    print(f"Processed {len(ids)} samples.")
    assert len(ids) == 2
    assert len(atomic_feats) == 2
    assert global_feats.shape == (
        2,
        12,
    )  # 3 lengths + 3 angles + vol + density + total_atoms + 3 percents
    assert targets.shape == (2, 2)

    # Check feature dimensions
    # Atomic features: 4 (one-hot) + 3 (phys) + 3 (coords) + 3 (stats) = 13
    assert (
        atomic_feats[0].shape[1] == 13
    ), f"Expected 13 atomic features, got {atomic_feats[0].shape[1]}"

    # 4. Demonstrate Data Loading
    print("\n--- Testing Library: Data ---")
    # This will load the full datasets defined in metadata/
    # It might take a moment to process/load cache.
    train_loader, val_loader, test_loader, scaler_atomic, scaler_global = (
        get_data_loaders(
            input_dir="./input",
            batch_size=CONFIG["batch_size"],
            load_cached_data=True,  # Try to load if exists to save time, else compute
        )
    )

    print(f"Train Loader batches: {len(train_loader)}")
    print(f"Val Loader batches: {len(val_loader)}")
    print(f"Test Loader batches: {len(test_loader)}")

    # Fetch one batch to verify
    batch = next(iter(train_loader))
    atomic_batch, batch_indices, global_batch, target_batch, id_list = batch

    print(f"Batch Atomic Features Shape: {atomic_batch.shape}")
    print(f"Batch Indices Shape: {batch_indices.shape}")
    print(f"Batch Global Features Shape: {global_batch.shape}")
    print(f"Batch Targets Shape: {target_batch.shape}")

    assert atomic_batch.dim() == 2
    assert atomic_batch.shape[1] == 13
    assert global_batch.shape[1] == 12
    assert target_batch.shape[1] == 2
    assert len(id_list) == global_batch.shape[0]

    # 5. Demonstrate Model
    print("\n--- Testing Library: Model ---")
    atomic_input_dim = 13
    global_input_dim = 12

    model = PADSDS(
        atomic_input_dim=atomic_input_dim,
        global_input_dim=global_input_dim,
        hidden_dim=64,  # Reduced for demo speed
        latent_dim=64,
    )
    model.to(CONFIG["device"])

    # Run forward pass with the batch fetched earlier
    atomic_batch = atomic_batch.to(CONFIG["device"])
    batch_indices = batch_indices.to(CONFIG["device"])
    global_batch = global_batch.to(CONFIG["device"])

    outputs = model(atomic_batch, batch_indices, global_batch)
    print(f"Model Output Shape: {outputs.shape}")

    assert outputs.shape == (global_batch.shape[0], 2)
    assert not torch.isnan(outputs).any(), "Model produced NaN outputs"

    # 6. Demonstrate Engine (Training Loop)
    print("\n--- Testing Library: Engine ---")

    optimizer = optim.Adam(
        model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
    )
    criterion = nn.MSELoss()

    engine = Engine(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=CONFIG["device"],
        patience=CONFIG["patience"],
        save_dir=demo_save_dir,
    )

    # Run fit (1 epoch as configured)
    print("Running training fit...")
    engine.fit(train_loader, val_loader, epochs=CONFIG["epochs"])

    # Check if model saved
    expected_model_path = os.path.join(demo_save_dir, "best_model.pt")
    if os.path.exists(expected_model_path):
        print("Best model checkpoint found.")
    else:
        print(
            "Note: Best model might not be saved if validation didn't improve (unlikely for 1st epoch)."
        )

    # 7. Demonstrate Prediction and Submission
    print("\n--- Testing Library: Submission ---")
    submission_path = "./working/demo_submission/demo_submission.csv"
    engine.generate_submission(test_loader, output_path=submission_path)

    if os.path.exists(submission_path):
        print(f"Submission file generated at {submission_path}")
        sub_df = pd.read_csv(submission_path)
        print("Submission head:")
        print(sub_df.head())
        assert list(sub_df.columns) == [
            "id",
            "formation_energy_ev_natom",
            "bandgap_energy_ev",
        ]
        assert len(sub_df) == 240  # Test set size
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
