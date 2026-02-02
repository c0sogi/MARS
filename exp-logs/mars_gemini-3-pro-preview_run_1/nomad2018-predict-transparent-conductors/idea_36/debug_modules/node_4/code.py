import os
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.geometry import GeometryProcessor
from library.data import get_dataloaders
from library.model import PGWDS
from library.train import run_training, generate_submission


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration Override
    # -------------------------------------------------------------------------
    print("--- 1. Configuring for Demonstration ---")
    # Reduce hyperparameters to ensure the script runs quickly
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 16
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    print(f"Epochs set to: {Config.NUM_EPOCHS}")
    print(f"Batch size set to: {Config.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Geometry Processing Demonstration
    # -------------------------------------------------------------------------
    print("\n--- 2. Demonstrating Geometry Processor ---")
    processor = GeometryProcessor()

    # Load train metadata to get a valid file path
    train_meta_path = os.path.join(Config.METADATA_DIR, "train.csv")
    if os.path.exists(train_meta_path):
        df = pd.read_csv(train_meta_path)
        # Pick the first sample
        sample_row = df.iloc[0]
        sample_path = sample_row["file_path"]
        print(f"Parsing sample file: {sample_path}")

        # Parse XYZ file
        atoms = processor.parse_xyz(sample_path)
        print(f"Parsed {len(atoms)} atoms.")

        # Extract features
        features = processor.extract_geometric_fingerprints(atoms)
        print("Extracted feature keys:", list(features.keys()))

        # Verify feature shapes
        n_atoms = len(atoms)
        assert features["species_indices"].shape == (
            n_atoms,
        ), "Species indices shape mismatch"
        assert features["centered_coords"].shape == (
            n_atoms,
            3,
        ), "Coords shape mismatch"
        assert features["d_min"].shape == (n_atoms,), "d_min shape mismatch"
        assert features["d_mean"].shape == (n_atoms,), "d_mean shape mismatch"
        print("Geometry feature shapes verified.")
    else:
        print("Train metadata not found, skipping specific file check.")

    # -------------------------------------------------------------------------
    # 3. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n--- 3. Demonstrating Data Loading ---")
    # We use load_cached_data=True to use existing cache if available,
    # otherwise it will compute and cache.
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    print(f"Number of train batches: {len(train_loader)}")

    # Inspect one batch
    batch = next(iter(train_loader))
    print("Batch keys:", list(batch.keys()))

    atomic_feats = batch["atomic_features"]
    global_feats = batch["global_features"]
    batch_indices = batch["batch_indices"]
    targets = batch["targets"]
    ids = batch["id"]

    print(f"Atomic features shape: {atomic_feats.shape}")
    print(f"Global features shape: {global_feats.shape}")
    print(f"Targets shape: {targets.shape}")

    # Verify dimensions based on Config
    # Atomic: 4 (one-hot) + 3 (coords) + 1 (d_min) + 1 (d_mean) = 9
    expected_atomic_dim = Config.ATOMIC_INPUT_DIM
    assert (
        atomic_feats.shape[1] == expected_atomic_dim
    ), f"Expected {expected_atomic_dim} atomic features, got {atomic_feats.shape[1]}"

    # Global: 3 (len) + 3 (ang) + 1 (vol) + 1 (dens) + 3 (stoich) + 1 (num) = 12
    expected_global_dim = Config.GLOBAL_INPUT_DIM
    assert (
        global_feats.shape[1] == expected_global_dim
    ), f"Expected {expected_global_dim} global features, got {global_feats.shape[1]}"

    assert targets.shape[1] == 2, "Expected 2 targets (formation_energy, bandgap)"
    print("Data loader shapes verified.")

    # -------------------------------------------------------------------------
    # 4. Model Instantiation and Forward Pass
    # -------------------------------------------------------------------------
    print("\n--- 4. Demonstrating Model Forward Pass ---")
    device = torch.device(Config.DEVICE)
    model = PGWDS().to(device)
    print(f"Model initialized on {device}")

    # Prepare batch data for model
    batch_data = {
        "atomic_features": atomic_feats.to(device),
        "batch_indices": batch_indices.to(device),
        "global_features": global_feats.to(device),
    }

    # Run forward pass (inference mode)
    model.eval()
    with torch.no_grad():
        outputs = model(batch_data)

    print(f"Model output shape: {outputs.shape}")
    assert outputs.shape == (len(ids), 2), "Output shape mismatch with batch size"
    print("Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n--- 5. Demonstrating Training Loop ---")
    # This runs the full training pipeline for the reduced number of epochs
    trained_model = run_training(
        batch_size=Config.BATCH_SIZE, epochs=Config.NUM_EPOCHS, load_cached_data=True
    )
    print("Training loop execution completed.")

    # -------------------------------------------------------------------------
    # 6. Submission Generation Demonstration
    # -------------------------------------------------------------------------
    print("\n--- 6. Demonstrating Submission Generation ---")
    generate_submission(
        trained_model, batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if os.path.exists(submission_path):
        print(f"Submission file created at: {submission_path}")
        # Verify content format
        sub_df = pd.read_csv(submission_path)
        print("Submission head:")
        print(sub_df.head())

        expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
        assert (
            list(sub_df.columns) == expected_cols
        ), f"Columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"
        print("Submission format verified.")
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    main()
