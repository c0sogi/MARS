import os
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import (
    parse_xyz,
    compute_pbc_distances,
    calculate_idw_chemical_counts,
)
from library.data import MaterialDataset, collate_batch
from library.model import ChemicallyWeightedDeepSets
from library.train import train_model, generate_submission


def run_demo():
    print("=" * 60)
    print("CHEMICALLY-WEIGHTED DEEP SETS: DEMO EXECUTION")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Override
    # -------------------------------------------------------------------------
    print("\n[1] Setting up configuration for demo...")

    # Define a separate working directory for this demo to avoid overwriting real work
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths to point to the demo directory
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_DATA_CACHE = os.path.join(DEMO_DIR, "train_data.npz")
    Config.VAL_DATA_CACHE = os.path.join(DEMO_DIR, "val_data.npz")
    Config.TEST_DATA_CACHE = os.path.join(DEMO_DIR, "test_data.npz")
    Config.SCALERS_CACHE = os.path.join(DEMO_DIR, "scalers.npz")
    Config.MODEL_CHECKPOINT = os.path.join(DEMO_DIR, "best_model.pt")
    Config.SUBMISSION_FILE = os.path.join(DEMO_DIR, "demo_submission.csv")

    # Override Hyperparameters for speed
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.ATOMIC_HIDDEN_DIM = 32
    Config.GLOBAL_HIDDEN_DIM = 32
    Config.FUSION_HIDDEN_DIM = 32

    # Create a small test metadata file to speed up submission generation demo
    full_test_df = pd.read_csv(Config.TEST_METADATA)
    small_test_df = full_test_df.head(10)  # Use only 10 test samples
    small_test_path = os.path.join(DEMO_DIR, "test.csv")
    small_test_df.to_csv(small_test_path, index=False)
    Config.TEST_METADATA = small_test_path

    print(f"Working directory set to: {Config.WORKING_DIR}")
    print(f"Hyperparameters: Epochs={Config.NUM_EPOCHS}, Batch={Config.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Demonstrate Utils (Parsing & Feature Engineering)
    # -------------------------------------------------------------------------
    print("\n[2] Demonstrating Utility Functions...")

    # Pick a real file from the training set
    train_meta = pd.read_csv("./metadata/train.csv")
    sample_row = train_meta.iloc[0]
    sample_xyz_path = os.path.join(Config.INPUT_DIR, sample_row["file_path"])

    print(f"Parsing file: {sample_xyz_path}")
    lattice, atom_types, coords = parse_xyz(sample_xyz_path)

    print(f"  Lattice shape: {lattice.shape}")
    print(f"  Atom types: {atom_types[:5]}... ({len(atom_types)} total)")
    print(f"  Coordinates shape: {coords.shape}")

    # Validation
    assert lattice.shape == (3, 3), "Lattice must be 3x3"
    assert len(atom_types) == coords.shape[0], "Mismatch between atoms and coords"

    print("Computing PBC distances...")
    dists = compute_pbc_distances(coords, lattice)
    print(f"  Distance matrix shape: {dists.shape}")
    # Self-distance (diagonal) should be close to 0
    assert np.allclose(np.diag(dists), 0.0), "Diagonal of distance matrix should be 0"

    print("Calculating IDW Chemical Counts (Local Environment Features)...")
    idw_features = calculate_idw_chemical_counts(coords, atom_types, lattice, k=5)
    print(f"  IDW features shape: {idw_features.shape}")
    assert idw_features.shape == (len(atom_types), Config.NUM_ATOM_TYPES)

    # -------------------------------------------------------------------------
    # 3. Demonstrate Data Loading
    # -------------------------------------------------------------------------
    print("\n[3] Demonstrating Data Loading...")

    # Use a small sample size for the dataset
    SAMPLE_SIZE = 20
    print(f"Initializing Train Dataset (first {SAMPLE_SIZE} samples)...")

    # load_cached_data=False forces processing from scratch
    train_dataset = MaterialDataset(
        mode="train", load_cached_data=False, sample_size=SAMPLE_SIZE
    )

    print(f"  Dataset length: {len(train_dataset)}")
    assert len(train_dataset) == SAMPLE_SIZE

    # Inspect one sample
    atomic_f, global_f, targets, mat_id = train_dataset[0]
    print(
        f"  Sample 0 Atomic Features: {atomic_f.shape} (N_atoms, {Config.ATOMIC_FEATURE_DIM})"
    )
    print(
        f"  Sample 0 Global Features: {global_f.shape} ({Config.GLOBAL_FEATURE_DIM},)"
    )
    print(f"  Sample 0 Targets: {targets}")

    # Demonstrate Collation
    print("Collating a batch...")
    loader = DataLoader(train_dataset, batch_size=4, collate_fn=collate_batch)
    batch = next(iter(loader))

    print(f"  Batch Atomic Features: {batch['atomic_features'].shape}")
    print(f"  Batch Atomic Mask: {batch['atomic_mask'].shape}")
    print(f"  Batch Global Features: {batch['global_features'].shape}")
    print(f"  Batch Targets: {batch['targets'].shape}")

    # -------------------------------------------------------------------------
    # 4. Demonstrate Model
    # -------------------------------------------------------------------------
    print("\n[4] Demonstrating Model Forward Pass...")

    model = ChemicallyWeightedDeepSets()
    # Ensure model is on the correct device (CPU for this quick demo)
    device = "cpu"
    model.to(device)

    # Pass the batch data
    outputs = model(
        batch["atomic_features"].to(device),
        batch["atomic_mask"].to(device),
        batch["global_features"].to(device),
    )

    print(f"  Model Output Shape: {outputs.shape}")
    assert outputs.shape == (4, 2), "Output shape should be (Batch_Size, 2)"
    print("  Forward pass successful.")

    # -------------------------------------------------------------------------
    # 5. Demonstrate Training Loop
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Loop (2 Epochs)...")

    # This function handles the entire training lifecycle:
    # 1. Loads Train/Val datasets (and fits scalers on Train)
    # 2. Initializes Model, Optimizer, Scheduler
    # 3. Runs training and validation loops
    # 4. Saves the best model checkpoint
    train_model(load_cached_data=False, sample_size=SAMPLE_SIZE)

    if os.path.exists(Config.MODEL_CHECKPOINT):
        print(f"  Checkpoint saved successfully at: {Config.MODEL_CHECKPOINT}")
    else:
        raise FileNotFoundError("Model checkpoint was not created!")

    # -------------------------------------------------------------------------
    # 6. Demonstrate Submission Generation
    # -------------------------------------------------------------------------
    print("\n[6] Generating Submission...")

    # This function:
    # 1. Loads the Test dataset (using the small metadata we created)
    # 2. Loads the saved model checkpoint
    # 3. Runs inference and applies inverse transform (expm1)
    # 4. Saves CSV
    generate_submission(load_cached_data=False)

    if os.path.exists(Config.SUBMISSION_FILE):
        print(f"  Submission file created at: {Config.SUBMISSION_FILE}")
        df_sub = pd.read_csv(Config.SUBMISSION_FILE)
        print("  Submission Head:")
        print(df_sub.head())
        assert (
            len(df_sub) == 10
        ), "Submission should have 10 rows (from our small test set)"
    else:
        raise FileNotFoundError("Submission file was not created!")

    print("\n" + "=" * 60)
    print("DEMO COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
