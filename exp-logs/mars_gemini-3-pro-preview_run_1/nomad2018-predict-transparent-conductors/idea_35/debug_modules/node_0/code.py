import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import library components
from library.config import Config
from library.utils import (
    parse_xyz,
    compute_pbc_distances,
    get_multi_order_neighbors,
    calculate_apf,
)
from library.data import (
    process_dataframe,
    flatten_atomic_features,
    unflatten_atomic_features,
    fit_and_save_scalers,
    apply_scaling,
    MaterialDataset,
    collate_fn,
    get_data_loaders,
)
from library.model import MNPADSModel
from library.train import train_one_epoch, validate, generate_submission, set_seed


def run_demo():
    print("=== Starting MNPA-DS Library Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Demo
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo...")

    # Use a separate working directory for the demo to avoid overwriting real training data
    DEMO_WORKING_DIR = "./working/demo_execution"
    DEMO_SUBMISSION_DIR = "./working/demo_submission"

    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)
    os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)

    # Override Config attributes
    Config.WORKING_DIR = DEMO_WORKING_DIR
    Config.SUBMISSION_DIR = DEMO_SUBMISSION_DIR
    Config.TRAIN_CACHE_PATH = os.path.join(DEMO_WORKING_DIR, "train_data.npz")
    Config.VAL_CACHE_PATH = os.path.join(DEMO_WORKING_DIR, "val_data.npz")
    Config.TEST_CACHE_PATH = os.path.join(DEMO_WORKING_DIR, "test_data.npz")
    Config.SCALERS_CACHE_PATH = os.path.join(DEMO_WORKING_DIR, "scalers.npz")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_WORKING_DIR, "best_model.pt")
    Config.SUBMISSION_OUTPUT_PATH = os.path.join(
        DEMO_SUBMISSION_DIR, "demo_submission.csv"
    )

    # Set Debug mode to process only a small number of samples
    Config.DEBUG = True
    Config.DEBUG_SIZE = 50  # Process 50 samples for train/val/test
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 2

    print(f"Working directory set to: {Config.WORKING_DIR}")
    print(f"Debug mode: {Config.DEBUG}, Size: {Config.DEBUG_SIZE}")

    # Set seed
    set_seed(42)

    # -------------------------------------------------------------------------
    # 2. Verify Utils
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Pick a sample geometry file
    sample_id = 1
    sample_geo_path = os.path.join(Config.INPUT_DIR, f"test/{sample_id}/geometry.xyz")

    if not os.path.exists(sample_geo_path):
        raise FileNotFoundError(f"Sample file not found: {sample_geo_path}")

    # Test parse_xyz
    atoms = parse_xyz(sample_geo_path)
    print(f"Parsed atoms: {len(atoms)} atoms in unit cell.")
    assert len(atoms) > 0, "Failed to parse atoms."

    # Test compute_pbc_distances
    dists = compute_pbc_distances(atoms)
    # Shape should be (N_atoms, 27 * N_atoms)
    print(f"Distance matrix shape: {dists.shape}")
    assert dists.shape[0] == len(atoms), "Distance matrix row count mismatch."

    # Test get_multi_order_neighbors
    k_neighbors = 3
    neighbor_dists = get_multi_order_neighbors(dists, k=k_neighbors)
    print(f"Neighbor distances shape: {neighbor_dists.shape}")
    assert neighbor_dists.shape == (
        len(atoms),
        k_neighbors,
    ), "Neighbor distances shape mismatch."

    # Test calculate_apf
    apf = calculate_apf(atoms)
    print(f"Atomic Packing Factor: {apf:.4f}")
    assert (
        0.0 < apf < 1.0
    ), "APF calculation seems incorrect (expected between 0 and 1)."

    print("Utils verification passed.")

    # -------------------------------------------------------------------------
    # 3. Verify Data Processing
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Data Processing...")

    # We will use get_data_loaders which internally calls process_dataframe
    # Since we set Config.DEBUG = True, it will only process a small subset.
    # We force load_cached_data=False to ensure processing logic runs.

    train_loader, val_loader, test_loader = get_data_loaders(load_cached_data=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Check a single batch
    batch = next(iter(train_loader))
    atomic_features = batch["atomic_features"]
    global_features = batch["global_features"]
    mask = batch["mask"]
    targets = batch["targets"]
    ids = batch["ids"]

    print(f"Batch Atomic Features Shape: {atomic_features.shape}")  # (B, Max_Atoms, 10)
    print(f"Batch Global Features Shape: {global_features.shape}")  # (B, 13)
    print(f"Batch Mask Shape: {mask.shape}")  # (B, Max_Atoms)
    print(f"Batch Targets Shape: {targets.shape}")  # (B, 2)

    assert (
        atomic_features.dim() == 3 and atomic_features.size(2) == 10
    ), "Atomic feature dim mismatch"
    assert (
        global_features.dim() == 2 and global_features.size(1) == 13
    ), "Global feature dim mismatch"
    assert targets.dim() == 2 and targets.size(1) == 2, "Targets dim mismatch"

    print("Data processing verification passed.")

    # -------------------------------------------------------------------------
    # 4. Verify Model
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNPADSModel(config=Config).to(device)

    # Move batch to device
    atomic_features = atomic_features.to(device)
    global_features = global_features.to(device)
    mask = mask.to(device)

    # Forward pass
    outputs = model(atomic_features, global_features, mask)
    print(f"Model Output Shape: {outputs.shape}")

    assert outputs.shape == (atomic_features.size(0), 2), "Model output shape mismatch"
    print("Model verification passed.")

    # -------------------------------------------------------------------------
    # 5. Verify Training Loop
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Training Loop...")

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Train for 1 epoch
    loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Training Epoch Loss: {loss:.6f}")
    assert not np.isnan(loss), "Training loss is NaN"

    # Validate
    val_loss = validate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.6f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"

    # Save checkpoint
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print(f"Model saved to {Config.MODEL_SAVE_PATH}")
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint not found."

    print("Training loop verification passed.")

    # -------------------------------------------------------------------------
    # 6. Verify Submission Generation
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Submission Generation...")

    generate_submission(model, test_loader, device, Config.SUBMISSION_OUTPUT_PATH)

    if os.path.exists(Config.SUBMISSION_OUTPUT_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_OUTPUT_PATH)
        print(f"Submission file created with {len(df_sub)} rows.")
        print(df_sub.head())

        # Check columns
        expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
        assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

        # Check if we have predictions for the debug size (Config.DEBUG_SIZE)
        # Note: test_loader might have slightly fewer or equal samples depending on the split logic in data.py
        # In data.py debug mode, it takes head(DEBUG_SIZE) of the test dataframe.
        assert (
            len(df_sub) == Config.DEBUG_SIZE
        ), f"Expected {Config.DEBUG_SIZE} predictions, got {len(df_sub)}"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("Submission verification passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
