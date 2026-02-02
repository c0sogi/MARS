import sys
import os
import shutil
import torch
import numpy as np
import pandas as pd

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.physics import (
    get_lattice_matrix,
    calculate_cell_volume,
    compute_pbc_interactions,
)
from library.data_loader import get_loaders
from library.architecture import RPA_WDS
from library.engine import train_model, generate_submission


def run_demo():
    print("Starting demonstration...")

    # 1. Configure for Demo
    print("\n[Configuration]")
    # Override Config for speed and isolation
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.CACHE_DIR = "./working/demo_cache"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    # Ensure model path is updated to new cache dir
    Config.MODEL_PATH = os.path.join(Config.CACHE_DIR, "best_model.pt")
    Config.SCALER_PATH = os.path.join(Config.CACHE_DIR, "scalers.npz")

    # Re-setup directories
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Cache Dir: {Config.CACHE_DIR}")

    # 2. Verify Physics Functions
    print("\n[Verifying Physics Module]")

    # Test Lattice Matrix (Cubic)
    a, b, c = 10.0, 10.0, 10.0
    alpha, beta, gamma = 90.0, 90.0, 90.0
    lattice = get_lattice_matrix(a, b, c, alpha, beta, gamma)
    print("Lattice Matrix (Cubic 10x10x10):\n", lattice)
    expected_lattice = np.diag([10.0, 10.0, 10.0])
    assert np.allclose(
        lattice, expected_lattice
    ), "Lattice matrix calculation incorrect for cubic system."

    # Test Volume
    vol = calculate_cell_volume(lattice)
    print(f"Volume: {vol}")
    assert np.isclose(vol, 1000.0), "Volume calculation incorrect."

    # Test PBC Interactions
    # Two atoms along x-axis: x=1 and x=9 in a box of length 10.
    # Distance should be min(|1-9|, |10 - |1-9||) = min(8, 2) = 2.
    coords = np.array([[1.0, 5.0, 5.0], [9.0, 5.0, 5.0]])
    dist_matrix = compute_pbc_interactions(coords, lattice)
    print("Distance Matrix:\n", dist_matrix)

    # Diagonal should be 0
    assert np.allclose(
        np.diag(dist_matrix), 0.0
    ), "Diagonal of distance matrix should be 0."
    # Off-diagonal should be 2.0
    assert np.isclose(
        dist_matrix[0, 1], 2.0
    ), f"PBC distance calculation failed. Expected 2.0, got {dist_matrix[0, 1]}"
    print("Physics module verification passed.")

    # 3. Data Loading
    print("\n[Loading Data]")
    # This will process data, cache it, and return loaders
    # We force re-computation to demonstrate the pipeline by using a fresh cache dir
    train_loader, val_loader, test_loader = get_loaders(batch_size=Config.BATCH_SIZE)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Inspect a batch
    atomic_x, global_x, mask, targets, ids = next(iter(train_loader))
    print(f"Atomic Input Shape: {atomic_x.shape} (Batch, Atoms, Features)")
    print(f"Global Input Shape: {global_x.shape} (Batch, Features)")
    print(f"Mask Shape: {mask.shape}")
    print(f"Targets Shape: {targets.shape}")

    # Assertions on shapes
    assert (
        atomic_x.shape[2] == Config.ATOMIC_INPUT_DIM
    ), f"Atomic feature dim mismatch. Expected {Config.ATOMIC_INPUT_DIM}, got {atomic_x.shape[2]}"
    assert (
        global_x.shape[1] == Config.GLOBAL_INPUT_DIM
    ), f"Global feature dim mismatch. Expected {Config.GLOBAL_INPUT_DIM}, got {global_x.shape[1]}"
    assert (
        targets.shape[1] == 2
    ), "Targets should have 2 columns (formation energy, bandgap)."

    # 4. Model Initialization
    print("\n[Initializing Model]")
    model = RPA_WDS(
        atomic_input_dim=Config.ATOMIC_INPUT_DIM,
        global_input_dim=Config.GLOBAL_INPUT_DIM,
        atomic_hidden_dim=Config.ATOMIC_HIDDEN_DIM,
        global_hidden_dim=Config.GLOBAL_HIDDEN_DIM,
        fusion_hidden_dim=Config.FUSION_HIDDEN_DIM,
        dropout_rate=Config.DROPOUT_RATE,
    )

    # Move to device
    device = Config.DEVICE
    model.to(device)
    print(f"Model moved to {device}")

    # Test Forward Pass
    with torch.no_grad():
        dummy_out = model(atomic_x.to(device), global_x.to(device), mask.to(device))
    print(f"Model Output Shape: {dummy_out.shape}")
    assert dummy_out.shape == (atomic_x.shape[0], 2), "Model output shape mismatch."

    # 5. Training Loop
    print("\n[Training Model]")
    # We run for a few epochs as configured
    train_model(model, train_loader, val_loader)

    # Check if model file was saved
    if os.path.exists(Config.MODEL_PATH):
        print(f"Model saved successfully at {Config.MODEL_PATH}")
    else:
        # If validation didn't improve (unlikely in 2 epochs starting from scratch?), force save for demo
        print(
            "Model not saved by early stopping (metric didn't improve?), saving manually for demo."
        )
        torch.save(model.state_dict(), Config.MODEL_PATH)

    # 6. Generate Submission
    print("\n[Generating Submission]")
    generate_submission(model, test_loader)

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        df = pd.read_csv(Config.SUBMISSION_PATH)
        print("Submission file loaded.")
        print("Columns:", df.columns.tolist())
        print("Shape:", df.shape)

        expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
        assert list(df.columns) == expected_cols, "Submission columns mismatch."
        assert len(df) == 240, f"Expected 240 predictions, got {len(df)}"

        # Check for non-negative values (physics constraint applied in generate_submission)
        if (df["formation_energy_ev_natom"] < 0).any() or (
            df["bandgap_energy_ev"] < 0
        ).any():
            print("Warning: Negative values found in submission (should be clamped).")
        else:
            print("All predictions are non-negative.")

    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    run_demo()
