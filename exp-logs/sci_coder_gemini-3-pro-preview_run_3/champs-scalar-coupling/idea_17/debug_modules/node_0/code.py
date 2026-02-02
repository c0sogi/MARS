import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.preprocess import DataPreprocessor
from library.dataset import MoleculeDataset, collate_molecular_graphs
from library.model import MPDCFN
from library.engine import Trainer
from library.utils import set_seed


def run_demo():
    print("=== Starting MP-DCFN Pipeline Demo ===")

    # ==========================================
    # 1. Configure for Speed/Demo
    # ==========================================
    print("\n[1] Configuring environment for rapid execution...")

    # Modify Config global state directly
    Config.DEBUG_SAMPLE_SIZE = 500  # Small subset for speed
    Config.MAX_EPOCHS = 2  # Minimal epochs to prove loop works
    Config.BATCH_SIZE = 16  # Small batch size
    Config.NUM_WORKERS = 0  # Main process only to avoid overhead
    Config.PATIENCE = 2  # Short patience

    # Redirect output to a demo directory
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "processed")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    # Clean up previous demo runs if they exist
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Re-create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Set seeds
    set_seed(Config.SEED)
    print("Configuration updated.")

    # ==========================================
    # 2. Preprocessing
    # ==========================================
    print("\n[2] Running Data Preprocessing...")

    preprocessor = DataPreprocessor()
    # Force processing from scratch with debug sample size
    preprocessor.process(
        load_cached_data=False, debug_sample_size=Config.DEBUG_SAMPLE_SIZE
    )

    # Verify cache generation
    expected_files = [
        "atom_coords.npy",
        "edge_indices.npy",
        "coupling_values.npy",
        "completed.flag",
    ]
    for f in expected_files:
        fpath = os.path.join(Config.CACHE_DIR, f)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Preprocessing failed to generate {f}")

    print("Preprocessing verification passed. Cache files exist.")

    # ==========================================
    # 3. Dataset & DataLoader
    # ==========================================
    print("\n[3] Verifying Dataset and Collation...")

    # Load training dataset
    train_ds = MoleculeDataset(split="train")
    print(f"Train Dataset Size: {len(train_ds)} molecules")

    if len(train_ds) == 0:
        raise ValueError("Dataset is empty. Check preprocessing logic.")

    # Create DataLoader
    loader = DataLoader(
        train_ds, batch_size=4, collate_fn=collate_molecular_graphs, shuffle=True
    )

    # Fetch one batch
    batch = next(iter(loader))

    # Verify Batch Structure
    required_keys = [
        "x",
        "pos",
        "edge_index",
        "edge_attr",
        "coupling_index",
        "coupling_value",
        "batch",
    ]
    for k in required_keys:
        if k not in batch:
            raise KeyError(f"Batch missing key: {k}")

    # Verify Shapes
    num_nodes = batch["x"].size(0)
    num_edges = batch["edge_index"].size(1)
    num_couplings = batch["coupling_value"].size(0)

    # Edge index should be within node range
    if num_edges > 0:
        assert (
            batch["edge_index"].max() < num_nodes
        ), "Edge indices exceed number of nodes"

    # Coupling index should be within node range
    assert (
        batch["coupling_index"].max() < num_nodes
    ), "Coupling indices exceed number of nodes"

    print(
        f"Batch verification passed. Nodes: {num_nodes}, Edges: {num_edges}, Couplings: {num_couplings}"
    )

    # ==========================================
    # 4. Model Initialization & Forward Pass
    # ==========================================
    print("\n[4] Initializing Model and Testing Forward Pass...")

    device = torch.device(Config.DEVICE)
    model = MPDCFN().to(device)

    # Move batch to device
    batch_gpu = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(batch_gpu)

    # Verify Output
    assert output.shape == (
        num_couplings,
    ), f"Output shape mismatch. Expected ({num_couplings},), got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    print("Model forward pass successful.")

    # ==========================================
    # 5. Full Training Loop
    # ==========================================
    print("\n[5] Executing Training Loop (Trainer.fit)...")

    trainer = Trainer()
    trainer.fit()

    # Verify Checkpoint
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError("Training finished but best_model.pth was not created.")

    print(f"Training loop complete. Checkpoint saved at {Config.BEST_MODEL_PATH}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
