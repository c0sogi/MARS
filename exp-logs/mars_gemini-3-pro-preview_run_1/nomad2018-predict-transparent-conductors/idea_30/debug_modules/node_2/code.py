import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import provided library modules
import library.config as config
from library.utils import calculate_cell_volume, compute_pbc_distances
from library.features import process_geometry, process_dataset
from library.data import MaterialsDataset, collate_crystals
from library.model import GPA_WDS
from library.engine import Engine, set_seed


def demo_main():
    # 1. Set Seed for Reproducibility
    set_seed(config.SEED)
    print("Random seeds set.")

    # 2. Define Demo Directories
    # We use a subdirectory in working to avoid cluttering the main idea folder if possible,
    # but the config paths are hardcoded. We will respect the config paths for scalers/model
    # but use specific cache names for datasets to show we can control them.
    demo_cache_dir = "./working/demo_cache"
    os.makedirs(demo_cache_dir, exist_ok=True)

    # 3. Demonstrate Utility Functions
    print("\n--- Testing Utility Functions ---")
    # Test Volume Calculation
    lattice = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    vol = calculate_cell_volume(lattice)
    print(f"Calculated Volume: {vol}")
    assert np.isclose(vol, 1000.0), "Volume calculation incorrect"

    # Test PBC Distances
    coords = np.array([[1.0, 1.0, 1.0], [9.0, 1.0, 1.0]])
    # Distance along x is 8.0. With PBC (period 10.0), shortest distance is 2.0.
    dists = compute_pbc_distances(coords, lattice)
    print(f"Pairwise Distances Matrix:\n{dists}")
    assert np.isclose(dists[0, 1], 2.0), "PBC distance calculation incorrect"

    # 4. Demonstrate Feature Extraction
    print("\n--- Testing Feature Extraction ---")
    # We pick the first file from the training set metadata to test geometry processing
    train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    sample_row = train_df.iloc[0]
    sample_path = os.path.join(config.INPUT_DIR, sample_row["file_path"])

    if os.path.exists(sample_path):
        atomic_feats, global_feats = process_geometry(sample_path)
        print(f"Processed {sample_row['file_path']}")
        print(f"Atomic Features Shape: {atomic_feats.shape}")  # (N_atoms, 9)
        print(f"Global Features Shape: {global_feats.shape}")  # (15,)

        assert atomic_feats.shape[1] == config.ATOMIC_INPUT_DIM
        assert global_feats.shape[0] == config.GLOBAL_INPUT_DIM
    else:
        print(f"Sample file {sample_path} does not exist. Skipping geometry test.")

    # 5. Demonstrate Dataset Loading
    print("\n--- Testing Dataset & DataLoader ---")
    # We will instantiate the training dataset.
    # We use a custom cache path to avoid overwriting the main experiment's cache if it exists,
    # though re-computing is fast.
    train_cache = os.path.join(demo_cache_dir, "train_data.npz")
    val_cache = os.path.join(demo_cache_dir, "val_data.npz")
    test_cache = os.path.join(demo_cache_dir, "test_data.npz")

    # Initialize Train Dataset (Computes scalers internally)
    print("Initializing Train Dataset...")
    train_dataset = MaterialsDataset(
        metadata_path=config.TRAIN_METADATA_PATH,
        cache_path=train_cache,
        split="train",
        load_cached_data=False,  # Force processing
    )

    # Initialize Val Dataset
    print("Initializing Validation Dataset...")
    val_dataset = MaterialsDataset(
        metadata_path=config.VAL_METADATA_PATH,
        cache_path=val_cache,
        split="val",
        load_cached_data=False,
    )

    # Initialize Test Dataset
    print("Initializing Test Dataset...")
    test_dataset = MaterialsDataset(
        metadata_path=config.TEST_METADATA_PATH,
        cache_path=test_cache,
        split="test",
        load_cached_data=False,
    )

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Create DataLoader
    batch_size = 16
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_crystals
    )

    # Fetch one batch
    batch_atomic, batch_indices, batch_global, batch_targets, batch_ids = next(
        iter(train_loader)
    )
    print(f"Batch Atomic Feats: {batch_atomic.shape}")
    print(f"Batch Global Feats: {batch_global.shape}")
    print(f"Batch Targets: {batch_targets.shape}")

    assert batch_global.shape[0] == batch_size
    assert batch_targets.shape[1] == config.OUTPUT_DIM

    # 6. Demonstrate Model Forward Pass
    print("\n--- Testing Model Forward Pass ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = GPA_WDS().to(device)

    # Move batch to device
    b_atomic = batch_atomic.to(device)
    b_indices = batch_indices.to(device)
    b_global = batch_global.to(device)
    b_ids = batch_ids.to(device)

    outputs = model(b_atomic, b_global, b_indices, b_ids)
    print(f"Model Output Shape: {outputs.shape}")
    assert outputs.shape == (batch_size, config.OUTPUT_DIM)

    # 7. Demonstrate Engine Training
    print("\n--- Testing Engine Training ---")
    engine = Engine(device)

    # We run for 1 epoch to demonstrate the loop
    print("Running training for 1 epoch...")
    # Note: passing train_loader as val_loader just for speed/demo purposes to ensure it runs quickly.
    # In real scenario, use val_loader.
    engine.run_training(train_loader, train_loader, epochs=1, patience=1)

    # 8. Demonstrate Submission Generation
    print("\n--- Testing Submission Generation ---")
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_crystals
    )

    engine.generate_submission(test_loader)

    if os.path.exists(config.SUBMISSION_PATH):
        print(f"Submission file created at: {config.SUBMISSION_PATH}")
        df_sub = pd.read_csv(config.SUBMISSION_PATH)
        print(df_sub.head())
        assert len(df_sub) == len(test_dataset)
    else:
        raise FileNotFoundError("Submission file not generated.")

    print("\nDemo completed successfully.")


if __name__ == "__main__":
    demo_main()
