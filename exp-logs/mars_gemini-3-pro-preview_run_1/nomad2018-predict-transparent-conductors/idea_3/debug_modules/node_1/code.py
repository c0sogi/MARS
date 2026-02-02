import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Ensure the current directory is in the python path
sys.path.append(os.getcwd())

from library.config import Config
from library.features import process_dataset
from library.data import get_datasets, collate_fn
from library.model import RBFDualStreamDeepSets
from library.train import train_model, generate_submission


def run_demo():
    print("=" * 60)
    print("DEMO: Material Property Prediction Pipeline")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Use a separate working directory for this demo to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config parameters
    Config.WORKING_DIR = DEMO_DIR
    Config.NUM_EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 16  # Small batch size
    Config.TRAIN_CACHE_PATH = os.path.join(DEMO_DIR, "train_data.npz")
    Config.VAL_CACHE_PATH = os.path.join(DEMO_DIR, "val_data.npz")
    Config.TEST_CACHE_PATH = os.path.join(DEMO_DIR, "test_data.npz")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "demo_model.pt")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "demo_submission.csv")

    # Set seeds
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Epochs: {Config.NUM_EPOCHS}, Batch Size: {Config.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Feature Extraction Demonstration
    # -------------------------------------------------------------------------
    print("\n[2] Demonstrating Feature Extraction (library.features)...")

    # We will process the validation set metadata as it is smaller
    print(f"Processing validation metadata from: {Config.VAL_METADATA_PATH}")

    # Force re-processing to demonstrate logic (ignore cache if exists)
    if os.path.exists(Config.VAL_CACHE_PATH):
        os.remove(Config.VAL_CACHE_PATH)

    val_data = process_dataset(
        metadata_path=Config.VAL_METADATA_PATH,
        cache_path=Config.VAL_CACHE_PATH,
        load_cached_data=False,
    )

    # Verify outputs
    ids = val_data["ids"]
    atomic_feats = val_data["atomic_features"]
    lattice_feats = val_data["lattice_features"]
    targets = val_data["targets"]

    print(f"Processed {len(ids)} samples.")

    # Assertions
    assert len(ids) == len(atomic_feats) == len(lattice_feats) == len(targets)
    assert (
        lattice_feats.shape[1] == 7
    ), f"Expected 7 lattice features, got {lattice_feats.shape[1]}"
    assert targets.shape[1] == 2, f"Expected 2 targets, got {targets.shape[1]}"

    # Check atomic features (ragged array)
    sample_idx = 0
    sample_atoms = atomic_feats[sample_idx]
    print(f"Sample {ids[sample_idx]} has {sample_atoms.shape[0]} atoms.")
    print(f"Atomic feature dimension: {sample_atoms.shape[1]}")

    expected_atom_dim = Config.ATOMIC_INPUT_DIM
    assert (
        sample_atoms.shape[1] == expected_atom_dim
    ), f"Expected atomic dim {expected_atom_dim}, got {sample_atoms.shape[1]}"

    print("Feature extraction verification passed.")

    # -------------------------------------------------------------------------
    # 3. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n[3] Demonstrating Data Loading (library.data)...")

    # Initialize datasets
    # This will process/load train, val, and test data
    train_ds, val_ds, test_ds = get_datasets(load_cached_data=True)

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Val dataset size:   {len(val_ds)}")
    print(f"Test dataset size:  {len(test_ds)}")

    # Create DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    print("Fetched one batch:")
    print(f"  IDs shape: {batch['ids'].shape}")
    print(f"  Atomic Features shape: {batch['atomic_features'].shape}")
    print(f"  Batch Indices shape: {batch['batch_indices'].shape}")
    print(f"  Lattice Features shape: {batch['lattice_features'].shape}")
    print(f"  Targets shape: {batch['targets'].shape}")

    # Verify batch consistency
    total_atoms_in_batch = batch["atomic_features"].shape[0]
    assert batch["batch_indices"].shape[0] == total_atoms_in_batch
    assert batch["lattice_features"].shape[0] == Config.BATCH_SIZE
    assert batch["targets"].shape[0] == Config.BATCH_SIZE

    print("Data loading verification passed.")

    # -------------------------------------------------------------------------
    # 4. Model Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Demonstrating Model Forward Pass (library.model)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RBFDualStreamDeepSets().to(device)

    # Move batch to device
    b_atomic = batch["atomic_features"].to(device)
    b_lattice = batch["lattice_features"].to(device)
    b_indices = batch["batch_indices"].to(device)

    # Forward pass
    output = model(b_atomic, b_lattice, b_indices)

    print(f"Model Output shape: {output.shape}")

    # Assertions
    assert output.shape == (
        Config.BATCH_SIZE,
        2,
    ), f"Expected output shape ({Config.BATCH_SIZE}, 2), got {output.shape}"

    print("Model forward pass verification passed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Demonstrating Training Loop (library.train)...")

    # Train for a few epochs
    # Note: train_model internally calls get_datasets and handles DataLoaders
    best_loss = train_model(
        num_epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=1e-3,
        patience=2,
        load_cached_data=True,
    )

    print(f"Training finished. Best Validation Loss: {best_loss:.4f}")

    # Verify model file creation
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Model checkpoint found at: {Config.MODEL_SAVE_PATH}")
    else:
        raise FileNotFoundError("Model checkpoint was not created!")

    # -------------------------------------------------------------------------
    # 6. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n[6] Demonstrating Inference and Submission (library.train)...")

    # Generate submission using the trained model
    generate_submission(
        model_path=Config.MODEL_SAVE_PATH,
        batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
    )

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file created at: {Config.SUBMISSION_PATH}")
        print(f"Submission shape: {df_sub.shape}")
        print("Head:")
        print(df_sub.head())

        # Check columns
        expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
        assert (
            list(df_sub.columns) == expected_cols
        ), f"Columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

        # Check row count (Test set has 240 samples)
        assert len(df_sub) == 240, f"Expected 240 predictions, got {len(df_sub)}"

        # Check for NaNs
        assert not df_sub.isnull().values.any(), "Submission contains NaNs"

        # Check for non-negative values (energies should be >= 0)
        assert (
            df_sub["formation_energy_ev_natom"] >= 0
        ).all(), "Negative formation energy found"
        assert (df_sub["bandgap_energy_ev"] >= 0).all(), "Negative bandgap energy found"

        print("Submission verification passed.")
    else:
        raise FileNotFoundError("Submission file was not created!")

    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
