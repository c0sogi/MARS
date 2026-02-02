import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library modules
from library import utils
from library import features
from library import data
from library import model as model_lib
from library import train as train_lib

# Set seeds
torch.manual_seed(42)
np.random.seed(42)


def demo_utils():
    print("\n--- Demo: library.utils ---")

    # Test calculate_cell_volume
    # Cubic cell 10x10x10
    vol = utils.calculate_cell_volume(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    print(f"Calculated volume (10^3): {vol}")
    assert np.isclose(vol, 1000.0), f"Volume should be 1000, got {vol}"

    # Test calculate_angular_distortion
    dist = utils.calculate_angular_distortion(90.0, 90.0, 90.0)
    print(f"Angular distortion (cubic): {dist}")
    assert np.isclose(dist, 0.0), "Distortion should be 0 for cubic"

    dist_distorted = utils.calculate_angular_distortion(85.0, 95.0, 90.0)
    print(f"Angular distortion (85, 95, 90): {dist_distorted}")
    assert np.isclose(dist_distorted, 10.0), "Distortion should be 10"

    # Test get_pbc_distances
    # Simple cubic lattice with 2 atoms
    lattice = np.eye(3) * 10.0
    positions = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    # Nearest neighbor for atom 0 should be atom 1 at dist 5.0
    # Also image of atom 1 at -5.0 (dist 5.0)
    dists, indices = utils.get_pbc_distances(positions, lattice, k_neighbors=2)
    print(f"PBC Distances shape: {dists.shape}")
    print(f"Nearest neighbor distance for atom 0: {dists[0, 0]}")

    assert dists.shape == (2, 2)
    assert np.isclose(dists[0, 0], 5.0), f"Expected distance 5.0, got {dists[0, 0]}"


def demo_features_and_data():
    print("\n--- Demo: library.features and library.data ---")

    # 1. Create dummy metadata in ./working/dummy_meta
    dummy_meta_dir = "./working/dummy_meta"
    os.makedirs(dummy_meta_dir, exist_ok=True)

    # Read original train.csv to get valid file paths
    orig_train = pd.read_csv("./metadata/train.csv")

    # Take top 10 samples for train, 5 for val
    dummy_train = orig_train.iloc[:10].copy()
    dummy_val = orig_train.iloc[10:15].copy()

    dummy_train.to_csv(os.path.join(dummy_meta_dir, "train.csv"), index=False)
    dummy_val.to_csv(os.path.join(dummy_meta_dir, "val.csv"), index=False)

    print(f"Created dummy metadata in {dummy_meta_dir}")

    # 2. Test parse_xyz directly
    sample_path = os.path.join("./input", dummy_train.iloc[0]["file_path"])
    lattice, atoms, coords = features.parse_xyz(sample_path)
    print(f"Parsed XYZ: {len(atoms)} atoms, Lattice shape {lattice.shape}")
    assert lattice.shape == (3, 3)
    assert len(atoms) == len(coords)

    # 3. Test extract_atomic_features
    atom_feats = features.extract_atomic_features(coords, lattice, atoms)
    print(f"Atomic features shape: {atom_feats.shape}")
    # Expected dim is 21 (4 identity + 4 nn + 3 spatial + 1 dist + 1 pack + 8 context)
    assert atom_feats.shape[1] == 21

    # 4. Test extract_global_features
    row = dummy_train.iloc[0]
    glob_feats = features.extract_global_features(row, atoms)
    print(f"Global features shape: {glob_feats.shape}")
    # Expected dim is 22
    assert glob_feats.shape[0] == 22

    # 5. Use MaterialDataset to process and load
    # Note: This will write to ./working/idea_47/ which is allowed
    print("Initializing MaterialDataset (Train)...")
    train_ds = data.MaterialDataset(
        split="train",
        input_dir="./input",
        metadata_dir=dummy_meta_dir,
        load_cached_data=False,  # Force processing
    )

    print(f"Dataset length: {len(train_ds)}")
    assert len(train_ds) == 10

    # Check item
    af, gf, y, cid = train_ds[0]
    print(f"Sample 0: Atomic {af.shape}, Global {gf.shape}, Target {y.shape}, ID {cid}")
    assert af.ndim == 2
    assert gf.ndim == 1
    assert y.shape == (2,)

    # 6. DataLoader with SparseCollate
    print("Testing DataLoader with SparseCollate...")
    loader = DataLoader(
        train_ds, batch_size=4, collate_fn=data.SparseCollate(), shuffle=False
    )

    batch = next(iter(loader))
    batch_atomic, batch_indices, batch_global, batch_targets, batch_ids = batch

    print(f"Batch Atomic Feats: {batch_atomic.shape}")
    print(f"Batch Indices: {batch_indices.shape}")
    print(f"Batch Global Feats: {batch_global.shape}")
    print(f"Batch Targets: {batch_targets.shape}")
    print(f"Batch IDs: {batch_ids.shape}")

    assert batch_atomic.shape[0] == batch_indices.shape[0]
    assert batch_global.shape[0] == 4
    assert batch_targets.shape[0] == 4

    return loader


def demo_model(loader):
    print("\n--- Demo: library.model ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Instantiate model
    model = model_lib.CEADSModel(
        atomic_input_dim=21,
        global_input_dim=22,
        atomic_hidden=64,  # Reduced for demo speed
        global_hidden=32,
        fusion_hidden=32,
        output_dim=2,
        dropout=0.0,
    ).to(device)

    # Get a batch
    batch = next(iter(loader))
    batch_atomic, batch_indices, batch_global, batch_targets, _ = batch

    # Move to device
    batch_atomic = batch_atomic.to(device)
    batch_indices = batch_indices.to(device)
    batch_global = batch_global.to(device)

    # Forward pass
    output = model(batch_atomic, batch_indices, batch_global)
    print(f"Model Output Shape: {output.shape}")

    assert output.shape == (4, 2)
    print("Forward pass successful.")

    return model


def demo_training():
    print("\n--- Demo: library.train ---")

    # We will use the train function from library.train but with very small parameters
    # Note: The library.train.train function uses get_train_val_loaders which points to ./metadata
    # To use our dummy metadata, we'd need to patch or modify arguments.
    # Since we cannot modify library files, we will manually run the training loop logic
    # using our dummy loaders created in demo_features_and_data.

    # Re-create loaders with dummy metadata
    dummy_meta_dir = "./working/dummy_meta"
    train_ds = data.MaterialDataset(
        "train", metadata_dir=dummy_meta_dir, load_cached_data=True
    )
    val_ds = data.MaterialDataset(
        "val", metadata_dir=dummy_meta_dir, load_cached_data=False
    )

    collate = data.SparseCollate()
    train_loader = DataLoader(train_ds, batch_size=4, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=4, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model_lib.CEADSModel(
        atomic_hidden=32, global_hidden=16, fusion_hidden=16
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Running 1 epoch of training (manual loop)...")
    loss = model_lib.train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {loss:.4f}")

    print("Running validation...")
    val_loss = model_lib.validate(model, val_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}")

    # Test generate_submission logic
    # We need a dummy test set for this
    orig_test = pd.read_csv("./metadata/test.csv")
    dummy_test = orig_test.iloc[:5].copy()
    dummy_test.to_csv(os.path.join(dummy_meta_dir, "test.csv"), index=False)

    # We can't easily inject the dummy test loader into generate_submission
    # because it calls get_test_loader inside.
    # However, we can verify the logic by manually running the loop.

    print("Simulating submission generation...")
    test_ds = data.MaterialDataset(
        "test", metadata_dir=dummy_meta_dir, load_cached_data=False
    )
    test_loader = DataLoader(test_ds, batch_size=4, collate_fn=collate)

    model.eval()
    results = []
    with torch.no_grad():
        for batch in test_loader:
            atomic_feats, batch_indices, global_feats, _, ids = batch
            atomic_feats = atomic_feats.to(device)
            batch_indices = batch_indices.to(device)
            global_feats = global_feats.to(device)

            outputs = model(atomic_feats, batch_indices, global_feats)
            preds = torch.expm1(outputs).cpu().numpy()
            ids = ids.numpy()

            for i in range(len(ids)):
                results.append(
                    {
                        "id": ids[i],
                        "formation_energy_ev_natom": preds[i, 0],
                        "bandgap_energy_ev": preds[i, 1],
                    }
                )

    df = pd.DataFrame(results)
    print("Generated predictions:")
    print(df.head())
    assert len(df) == 5

    # Save to working directory
    out_path = "./working/demo_submission.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    try:
        demo_utils()
        loader = demo_features_and_data()
        demo_model(loader)
        demo_training()
        print("\nAll demos completed successfully!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        raise
