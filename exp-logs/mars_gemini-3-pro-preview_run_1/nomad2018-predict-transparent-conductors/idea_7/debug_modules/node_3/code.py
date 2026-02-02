import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Import library components
from library.config import Config
from library.geometry import (
    read_xyz,
    center_coordinates,
    compute_pbc_neighbor_distances,
)
from library.features import extract_global_features, FeatureScaler
from library.dataset import MaterialDataset, collate_batch
from library.model import SIRDSModel, predict
from library.train import Trainer


def demo_geometry_and_features():
    print("\n--- Demo: Geometry and Feature Extraction ---")

    # 1. Read a sample geometry file
    sample_id = 1
    xyz_path = os.path.join(Config.INPUT_DIR, f"train/{sample_id}/geometry.xyz")

    if not os.path.exists(xyz_path):
        print(f"Sample file {xyz_path} not found. Skipping geometry demo.")
        return

    lattice, atom_types, atom_coords = read_xyz(xyz_path)
    print(f"Loaded geometry for ID {sample_id}:")
    print(f"  Lattice shape: {lattice.shape}")
    print(f"  Number of atoms: {len(atom_types)}")
    print(f"  Coordinates shape: {atom_coords.shape}")

    assert lattice.shape == (3, 3), "Lattice should be 3x3"
    assert len(atom_types) == atom_coords.shape[0], "Atom types and coords mismatch"

    # 2. Center coordinates
    centered = center_coordinates(atom_coords, lattice)
    print(f"  Centered coordinates mean: {centered.mean(axis=0)}")
    # Centroid should be roughly around 0 relative to cell center logic,
    # but the function centers around cell centroid.

    # 3. Compute PBC neighbor distances
    neighbor_dists = compute_pbc_neighbor_distances(atom_coords, lattice)
    print(f"  Neighbor distances shape: {neighbor_dists.shape}")
    assert neighbor_dists.shape[0] == len(atom_types)
    assert np.all(neighbor_dists > 0), "Neighbor distances must be positive"

    # 4. Extract global features from metadata
    # Load one row from metadata
    df = pd.read_csv(Config.TRAIN_CSV, nrows=1)
    row = df.iloc[0]
    global_feats = extract_global_features(row)
    print(f"  Global features shape: {global_feats.shape}")
    # Expected dim is 11: a, b, c, alpha, beta, gamma, vol, density, stoich(3)
    assert (
        global_feats.shape[0] == 11
    ), f"Expected 11 global features, got {global_feats.shape[0]}"


def demo_dataset_and_dataloader():
    print("\n--- Demo: Dataset and DataLoader ---")

    # Modify Config for this demo to use a temporary directory and small size
    Config.WORKING_DIR = "./working/demo_execution"
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_data.npz")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_data.npz")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pt")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Initialize Dataset with debug size
    debug_size = 50
    print(f"Initializing MaterialDataset (train) with {debug_size} samples...")
    train_dataset = MaterialDataset(
        metadata_path=Config.TRAIN_CSV,
        geometry_dir=Config.GEOMETRY_DIR,
        cache_path=Config.TRAIN_CACHE,
        load_cached_data=False,  # Force processing to test logic
        debug_sample_size=debug_size,
        mode="train",
    )

    print(f"Dataset length: {len(train_dataset)}")
    assert len(train_dataset) == debug_size

    # Check a single item
    item = train_dataset[0]
    print("Sample item keys:", item.keys())
    print("Atomic features shape:", item["atomic_features"].shape)
    print("Global features shape:", item["global_features"].shape)
    print("Targets:", item["targets"])

    # Feature dim check: 4 (one-hot) + 3 (coords) + 1 (dist) = 8
    assert item["atomic_features"].shape[1] == 8
    assert item["global_features"].shape[0] == 11

    # Initialize DataLoader
    batch_size = 4
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Batch atomic features shape: {batch['atomic_features'].shape}")
    print(f"Batch mask shape: {batch['mask'].shape}")

    # Verify padding
    # Shape: (Batch, Max_Atoms_In_Batch, Feats)
    assert batch["atomic_features"].dim() == 3
    assert batch["atomic_features"].size(0) == batch_size
    assert batch["mask"].size(0) == batch_size

    return train_dataset.scaler  # Return scaler to use for val/test


def demo_model_training(scaler):
    print("\n--- Demo: Model Training ---")

    # Configure for speed
    Config.HIDDEN_DIM = 32
    Config.NUM_RES_BLOCKS = 1
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Instantiate Model
    model = SIRDSModel(Config)
    model.to(device)
    print("Model instantiated.")

    # 2. Prepare DataLoaders
    # Re-use train dataset from cache (created in previous step)
    train_dataset = MaterialDataset(
        metadata_path=Config.TRAIN_CSV,
        geometry_dir=Config.GEOMETRY_DIR,
        cache_path=Config.TRAIN_CACHE,
        load_cached_data=True,
        debug_sample_size=50,
        mode="train",
        scaler=scaler,  # Use fitted scaler
    )

    # Create a small validation set
    val_dataset = MaterialDataset(
        metadata_path=Config.VAL_CSV,
        geometry_dir=Config.GEOMETRY_DIR,
        cache_path=Config.VAL_CACHE,
        load_cached_data=False,
        debug_sample_size=20,
        mode="val",
        scaler=scaler,  # Use same scaler
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )

    # 3. Test Forward Pass
    batch = next(iter(train_loader))
    atomic = batch["atomic_features"].to(device)
    global_f = batch["global_features"].to(device)
    sym = batch["symmetry"].to(device)
    mask = batch["mask"].to(device)

    with torch.no_grad():
        output = model(atomic, global_f, sym, mask)

    print(f"Model output shape: {output.shape}")
    assert output.shape == (atomic.size(0), 2), "Output shape mismatch"

    # 4. Run Training Loop using Trainer
    print("Starting training loop...")
    trainer = Trainer(model, Config, device)
    trainer.fit(train_loader, val_loader)

    return trainer.model


def demo_prediction(model, scaler):
    print("\n--- Demo: Prediction ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config for test
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_data.npz")

    # Load Test Dataset
    test_dataset = MaterialDataset(
        metadata_path=Config.TEST_CSV,
        geometry_dir=Config.GEOMETRY_DIR,
        cache_path=Config.TEST_CACHE,
        load_cached_data=False,
        debug_sample_size=10,  # Predict only a few
        mode="test",
        scaler=scaler,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )

    print("Running prediction...")
    preds, ids = predict(model, test_loader, device)

    print(f"Predictions shape: {preds.shape}")
    print(f"Number of IDs: {len(ids)}")
    print("Sample predictions (first 3):")
    for i in range(min(3, len(ids))):
        print(
            f"ID: {ids[i]}, Formation E: {preds[i][0]:.4f}, Bandgap: {preds[i][1]:.4f}"
        )

    assert preds.shape[1] == 2
    assert len(ids) == preds.shape[0]

    # Save dummy submission
    sub_dir = "./working/demo_submission"
    os.makedirs(sub_dir, exist_ok=True)
    sub_path = os.path.join(sub_dir, "demo_submission.csv")

    df = pd.DataFrame(
        {
            "id": ids,
            "formation_energy_ev_natom": preds[:, 0],
            "bandgap_energy_ev": preds[:, 1],
        }
    )
    df.to_csv(sub_path, index=False)
    print(f"Demo submission saved to {sub_path}")


if __name__ == "__main__":
    print("Starting SI-RDS Demonstration...")

    # 1. Geometry and Features
    demo_geometry_and_features()

    # 2. Dataset
    # We capture the scaler fitted on the training data to reuse it
    scaler = demo_dataset_and_dataloader()

    # 3. Model & Training
    trained_model = demo_model_training(scaler)

    # 4. Prediction
    demo_prediction(trained_model, scaler)

    print("\nDemonstration completed successfully.")
