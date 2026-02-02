import sys
import os
import torch
import numpy as np
import pandas as pd

# 1. Setup and Config Patching
# We patch the configuration to run a fast demonstration (fewer epochs, small data subset)
import library.config as config

print("Patching configuration for demonstration...")
config.DEBUG_MODE = True
config.DEBUG_SUBSET_SIZE = 50  # Use a very small subset for speed
config.NUM_EPOCHS = 2
config.BATCH_SIZE = 4
config.WORKING_DIR = "./working/demo_execution"
config.TRAIN_CACHE_PATH = os.path.join(config.WORKING_DIR, "train_data.npz")
config.VAL_CACHE_PATH = os.path.join(config.WORKING_DIR, "val_data.npz")
config.TEST_CACHE_PATH = os.path.join(config.WORKING_DIR, "test_data.npz")
config.SCALERS_CACHE_PATH = os.path.join(config.WORKING_DIR, "scalers.npz")
config.MODEL_SAVE_PATH = os.path.join(config.WORKING_DIR, "best_model.pt")
config.SUBMISSION_PATH = "./working/demo_submission/demo_submission.csv"

# Ensure working directory exists
os.makedirs(config.WORKING_DIR, exist_ok=True)
os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

# Import library modules after config patching
from library.physics_utils import (
    calculate_angular_distortion,
    get_pbc_neighbors,
    calculate_bond_hardness,
)
from library.data_processing import get_dataloaders, MaterialDataset
from library.model import GPIMSDS
from library.training import Trainer


def test_physics_utils():
    print("\n--- Testing Physics Utils ---")
    # Test angular distortion
    # Perfect angles (90, 90, 90) -> distortion 0
    dist_0 = calculate_angular_distortion(90.0, 90.0, 90.0)
    assert dist_0 == 0.0, f"Expected 0.0 distortion, got {dist_0}"

    # Distorted angles
    dist_1 = calculate_angular_distortion(90.0, 100.0, 85.0)
    expected = abs(90 - 90) + abs(100 - 90) + abs(85 - 90)  # 0 + 10 + 5 = 15
    assert abs(dist_1 - 15.0) < 1e-6, f"Expected 15.0 distortion, got {dist_1}"
    print("calculate_angular_distortion: OK")

    # Test PBC Neighbors (Mock data)
    # Simple cubic lattice, a=10
    lattice = np.eye(3) * 10.0
    # Two atoms close to each other
    coords = np.array([[5.0, 5.0, 5.0], [5.0, 5.0, 6.0]])  # dist = 1.0
    atom_types = ["Ga", "Al"]

    # Find 2 nearest neighbors
    dists, type_indices = get_pbc_neighbors(coords, lattice, atom_types, max_k=2)

    # For atom 0:
    # Nearest is self (dist=0.0), second nearest is atom 1 (dist=1.0)
    assert dists.shape == (2, 2)
    assert np.allclose(
        dists[0], [0.0, 1.0]
    ), f"Unexpected distances for atom 0: {dists[0]}"

    print("get_pbc_neighbors: OK")

    # Test Bond Hardness Proxy
    # d_min = 2.0
    # Central: Ga (r=1.22), Neighbor: Al (r=1.21)
    # Hardness = 2.0 / (1.22 + 1.21) = 2.0 / 2.43 = 0.823...
    d_min = np.array([2.0])
    central_types = ["Ga"]
    # Neighbor type index for Al is 0 (based on ATOM_TYPES=["Al", "Ga", "In", "O"])
    neigh_types = np.array([[config.ATOM_TO_IDX["Al"]]])  # (1, 1)
    neigh_dists = np.array([[2.0]])  # (1, 1)

    hardness = calculate_bond_hardness(
        d_min, central_types, neigh_types, neigh_dists, k_context=1
    )
    expected_h = 2.0 / (1.22 + 1.21)
    assert (
        np.abs(hardness[0] - expected_h) < 1e-4
    ), f"Hardness mismatch: {hardness[0]} vs {expected_h}"
    print("calculate_bond_hardness: OK")


def test_data_pipeline():
    print("\n--- Testing Data Pipeline ---")
    # Force reload=False to ensure processing logic runs and creates cache
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Inspect one batch
    batch = next(iter(train_loader))
    print("Batch keys:", list(batch.keys()))

    # Check shapes
    # atom_features: (Total_Atoms_In_Batch, ATOM_INPUT_DIM)
    assert batch["atom_features"].dim() == 2
    assert batch["atom_features"].shape[1] == config.ATOM_INPUT_DIM

    # global_features: (Batch_Size, GLOBAL_INPUT_DIM)
    assert batch["global_features"].dim() == 2
    assert batch["global_features"].shape[1] == config.GLOBAL_INPUT_DIM

    # targets: (Batch_Size, 2)
    assert batch["targets"].dim() == 2
    assert batch["targets"].shape[1] == 2

    print("Data Batch Structure: OK")
    return train_loader, val_loader, test_loader


def test_model_forward(train_loader):
    print("\n--- Testing Model Forward Pass ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPIMSDS().to(device)

    batch = next(iter(train_loader))
    atom_feats = batch["atom_features"].to(device)
    global_feats = batch["global_features"].to(device)
    batch_idx = batch["batch_index"].to(device)

    output = model(atom_feats, global_feats, batch_idx)

    assert output.shape == (batch["targets"].shape[0], 2)
    print(f"Model Output Shape: {output.shape}")
    print("Model Forward Pass: OK")
    return model


def test_training_loop(model, train_loader, val_loader):
    print("\n--- Testing Training Loop ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model, device)

    # Run fit (configured to 2 epochs in setup)
    trainer.fit(train_loader, val_loader)

    # Check if model saved
    assert os.path.exists(
        config.MODEL_SAVE_PATH
    ), "Model checkpoint not found after training"
    print("Training Loop: OK")
    return trainer


def test_inference(trainer, test_loader):
    print("\n--- Testing Inference ---")
    trainer.predict(test_loader)

    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not found"

    df = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission shape: {df.shape}")

    # Check columns
    expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
    assert all(c in df.columns for c in expected_cols)

    # Check no negative values (physical constraint enforced in predict)
    assert (
        df["formation_energy_ev_natom"] >= 0
    ).all(), "Found negative formation energy"
    assert (df["bandgap_energy_ev"] >= 0).all(), "Found negative bandgap energy"

    print("Inference: OK")


if __name__ == "__main__":
    # 1. Verify Physics Utils
    test_physics_utils()

    # 2. Verify Data Pipeline
    train_loader, val_loader, test_loader = test_data_pipeline()

    # 3. Verify Model
    model = test_model_forward(train_loader)

    # 4. Verify Training
    trainer = test_training_loop(model, train_loader, val_loader)

    # 5. Verify Inference
    test_inference(trainer, test_loader)

    print("\nAll demonstrations and verifications passed successfully.")
