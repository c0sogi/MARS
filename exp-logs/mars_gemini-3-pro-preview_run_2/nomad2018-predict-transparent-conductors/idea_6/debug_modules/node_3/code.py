import os
import shutil
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch

# Force reload of library modules to ensure changes are picked up in persistent environments
import sys
import importlib
import library.data

importlib.reload(library.data)

# Import from the provided library
from library.config import Config
from library.graph_utils import (
    load_structure,
    get_pbc_neighbor_graph,
    get_global_features,
)
from library.data import CrystalDataset, get_dataloaders, get_scalers
from library.model import VNCGCNN
from library.train import train_one_epoch, validate, generate_submission


def main():
    print("=== Setting up Demo Environment ===")
    # 1. Setup paths and directories
    demo_dir = "./working/demo_run"
    meta_dir = "./working/demo_metadata"
    cache_dir = os.path.join(demo_dir, "cache")
    checkpoint_dir = os.path.join(demo_dir, "checkpoints")
    submission_dir = "./working/demo_submission"

    for d in [demo_dir, meta_dir, cache_dir, checkpoint_dir, submission_dir]:
        os.makedirs(d, exist_ok=True)

    # 2. Create subset metadata for speed
    print("Creating subset metadata...")
    # Read original metadata
    orig_train = pd.read_csv("./metadata/train_metadata.csv")
    orig_val = pd.read_csv("./metadata/val_metadata.csv")
    orig_test = pd.read_csv("./metadata/test_metadata.csv")

    # Take small subsets (e.g., 20 train, 10 val, 10 test)
    # Ensure we pick IDs that actually exist in input (metadata generation verified this, but good to be safe)
    # The provided metadata is valid.
    sub_train = orig_train.head(20)
    sub_val = orig_val.head(10)
    sub_test = orig_test.head(10)

    sub_train_path = os.path.join(meta_dir, "train_metadata.csv")
    sub_val_path = os.path.join(meta_dir, "val_metadata.csv")
    sub_test_path = os.path.join(meta_dir, "test_metadata.csv")

    sub_train.to_csv(sub_train_path, index=False)
    sub_val.to_csv(sub_val_path, index=False)
    sub_test.to_csv(sub_test_path, index=False)

    # 3. Monkey-patch Config to use demo paths and settings
    print("Configuring parameters...")
    Config.METADATA_DIR = meta_dir
    Config.TRAIN_METADATA_PATH = sub_train_path
    Config.VAL_METADATA_PATH = sub_val_path
    Config.TEST_METADATA_PATH = sub_test_path
    Config.CACHE_DIR = cache_dir
    Config.CHECKPOINT_DIR = checkpoint_dir
    Config.BEST_MODEL_PATH = os.path.join(checkpoint_dir, "best_model_runfile.pth")
    Config.SUBMISSION_DIR = submission_dir
    Config.SUBMISSION_PATH = os.path.join(submission_dir, "submission.csv")

    # Speed up training for demo
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo script

    # 4. Test Graph Utils
    print("\n=== Testing Graph Utils ===")
    # Pick a sample file from the subset
    sample_rel_path = sub_train.iloc[0]["file_path"]
    sample_full_path = os.path.join(Config.INPUT_DIR, sample_rel_path)
    print(f"Loading structure: {sample_full_path}")

    atoms = load_structure(sample_full_path)
    assert atoms is not None, "Failed to load atoms object"
    print(f"  Atoms loaded: {len(atoms)} atoms")

    edge_index, edge_dist, atom_nums = get_pbc_neighbor_graph(
        atoms, cutoff=Config.CUTOFF_RADIUS
    )
    print(f"  Graph built: {edge_index.shape[1]} edges")
    assert edge_index.shape[0] == 2
    assert edge_dist.shape[0] == edge_index.shape[1]
    assert atom_nums.shape[0] == len(atoms)

    global_feats = get_global_features(atoms)
    print(f"  Global features shape: {global_feats.shape}")
    assert global_feats.shape == (10,), "Global features should be size 10"

    # 5. Test Data Loading
    print("\n=== Testing Data Loading ===")
    # This will internally call get_scalers and initialize datasets
    # We set load_cached_data=False to force reprocessing with the new logic
    train_loader, val_loader, test_loader, target_scaler = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,
    )

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Fetch one batch
    batch = next(iter(train_loader))
    print(f"  Batch content: {batch}")
    assert batch.x.ndim == 1
    assert batch.edge_index.shape[0] == 2

    # Assert correct shapes for batched 2D tensors
    assert batch.y.shape == (
        batch.num_graphs,
        2,
    ), f"Expected y shape ({batch.num_graphs}, 2), got {batch.y.shape}"
    assert batch.global_features.shape == (
        batch.num_graphs,
        10,
    ), f"Expected global_features shape ({batch.num_graphs}, 10), got {batch.global_features.shape}"

    # 6. Test Model
    print("\n=== Testing Model Architecture ===")
    device = torch.device("cpu")  # Use CPU for demo to ensure compatibility everywhere
    model = VNCGCNN(Config).to(device)
    print("  Model initialized.")

    # Forward pass
    batch = batch.to(device)
    output = model(batch)
    print(f"  Forward pass output shape: {output.shape}")
    assert output.shape == (batch.num_graphs, 2), "Output shape mismatch"

    # 7. Test Training Loop Components
    print("\n=== Testing Training Loop ===")
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Train one epoch
    loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"  Train Loss (Epoch 1): {loss:.4f}")
    assert not np.isnan(loss), "Training loss is NaN"

    # Validate
    val_loss = validate(model, val_loader, criterion, device)
    print(f"  Val Loss: {val_loss:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"

    # Save checkpoint
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print(f"  Saved checkpoint to {Config.BEST_MODEL_PATH}")

    # 8. Test Submission Generation
    print("\n=== Testing Submission Generation ===")
    # Reload model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    generate_submission(
        model, test_loader, target_scaler, device, Config.SUBMISSION_PATH
    )

    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"  Submission generated with {len(df_sub)} rows.")
        print(f"  Columns: {df_sub.columns.tolist()}")
        assert len(df_sub) == len(sub_test), "Submission row count mismatch"
        assert "id" in df_sub.columns
        assert "formation_energy_ev_natom" in df_sub.columns
        assert "bandgap_energy_ev" in df_sub.columns
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
