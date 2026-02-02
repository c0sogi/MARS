import os
import sys
import torch
import numpy as np
import pandas as pd
import ase.io

# 1. Import and Monkey-patch Configuration
# We must do this BEFORE importing other library modules to ensure they pick up the changes.
import library.config as config

# Define a separate working directory for this demo to avoid interference
DEMO_WORKING_DIR = "./working/demo_run"
os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

# Patch paths to use the demo directory
config.WORKING_DIR = DEMO_WORKING_DIR
config.TRAIN_CACHE_PATH = os.path.join(
    DEMO_WORKING_DIR, "cache", "train_graphs_debug.npz"
)
config.VAL_CACHE_PATH = os.path.join(DEMO_WORKING_DIR, "cache", "val_graphs_debug.npz")
config.TEST_CACHE_PATH = os.path.join(
    DEMO_WORKING_DIR, "cache", "test_graphs_debug.npz"
)
config.SCALER_CACHE_PATH = os.path.join(DEMO_WORKING_DIR, "cache", "scalers_debug.npy")
config.CHECKPOINT_PATH = os.path.join(
    DEMO_WORKING_DIR, "checkpoints", "best_model_runfile.pth"
)

config.SUBMISSION_DIR = "./working/demo_submission"
config.SUBMISSION_PATH = os.path.join(config.SUBMISSION_DIR, "submission.csv")

# Patch Training Hyperparameters for Speed
config.NUM_EPOCHS = 2
config.BATCH_SIZE = 32  # Smaller batch size for demo

# Ensure directories exist
os.makedirs(os.path.dirname(config.TRAIN_CACHE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(config.CHECKPOINT_PATH), exist_ok=True)
os.makedirs(config.SUBMISSION_DIR, exist_ok=True)

# 2. Import Library Modules
# These imports must happen AFTER patching config
from library.preprocessing import get_pbc_graph, extract_global_features
from library.data import CrystalDataset, get_loaders
from library.model import DSGCN
from library.trainer import Trainer


def demo_preprocessing():
    print("=== Demo: Preprocessing ===")
    # Load a sample geometry file
    # We use ID 1 from train set if available
    sample_id = 1
    sample_rel_path = f"train/{sample_id}/geometry.xyz"
    sample_full_path = os.path.join(config.INPUT_DIR, sample_rel_path)

    if not os.path.exists(sample_full_path):
        # Fallback to test/1/geometry.xyz if train/1 doesn't exist
        sample_rel_path = "test/1/geometry.xyz"
        sample_full_path = os.path.join(config.INPUT_DIR, sample_rel_path)

    if not os.path.exists(sample_full_path):
        print("No sample file found to demonstrate preprocessing.")
        return

    print(f"Processing file: {sample_full_path}")
    atoms = ase.io.read(sample_full_path)

    # Test Graph Construction
    graph_data = get_pbc_graph(atoms)
    print(f"Graph Data Keys: {list(graph_data.keys())}")
    print(f"Num Nodes: {len(graph_data['node_feats'])}")
    print(f"Num Edges: {graph_data['edge_index'].shape[1]}")

    # Verification
    assert "node_feats" in graph_data
    assert "edge_index" in graph_data
    assert "edge_dist" in graph_data
    assert graph_data["edge_index"].shape[0] == 2

    # Test Global Features
    global_feats = extract_global_features(atoms)
    print(f"Global Features Shape: {global_feats.shape}")
    print(f"Global Features: {global_feats}")

    # Verification
    assert global_feats.shape == (config.NUM_GLOBAL_FEATURES,)
    print("Preprocessing verification passed.")


def demo_data_loading():
    print("\n=== Demo: Data Loading ===")
    # This will process the data and save to the debug cache paths
    print("Initializing loaders (this may take a moment to process data)...")
    train_loader, val_loader, test_loader, scalers = get_loaders(load_cached_data=True)

    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches: {len(val_loader)}")
    print(f"Test Loader Batches: {len(test_loader)}")

    # Inspect a batch
    batch = next(iter(train_loader))
    print(f"Batch Sample: {batch}")
    print(f"  x shape: {batch.x.shape}")
    print(f"  edge_index shape: {batch.edge_index.shape}")
    print(f"  global_feat shape: {batch.global_feat.shape}")
    print(f"  y shape: {batch.y.shape}")

    # Verification
    assert batch.x.ndim == 1
    assert batch.edge_index.shape[0] == 2
    assert batch.global_feat.shape[1] == config.NUM_GLOBAL_FEATURES
    assert batch.y.shape[1] == config.NUM_TARGETS
    print("Data Loading verification passed.")
    return train_loader


def demo_model_forward(train_loader):
    print("\n=== Demo: Model Forward Pass ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = DSGCN().to(device)
    batch = next(iter(train_loader)).to(device)

    # Forward pass
    output = model(batch)
    print(f"Model Output Shape: {output.shape}")

    # Verification
    assert output.shape == (batch.num_graphs, config.NUM_TARGETS)
    print("Model forward pass verification passed.")


def demo_training_pipeline():
    print("\n=== Demo: Training Pipeline ===")
    # Initialize Trainer
    # This will re-load loaders, but since cache is now created, it should be fast
    trainer = Trainer()

    # Run training
    print(f"Starting training for {config.NUM_EPOCHS} epochs...")
    trainer.run()

    # Check if checkpoint exists
    if os.path.exists(config.CHECKPOINT_PATH):
        print(f"Checkpoint successfully saved at: {config.CHECKPOINT_PATH}")
    else:
        raise AssertionError("Checkpoint file was not created!")

    # Generate Submission
    print("Generating submission...")
    trainer.generate_submission()

    # Check if submission exists
    if os.path.exists(config.SUBMISSION_PATH):
        print(f"Submission successfully saved at: {config.SUBMISSION_PATH}")
        df = pd.read_csv(config.SUBMISSION_PATH)
        print("Submission Head:")
        print(df.head())

        # Verify submission format
        assert "id" in df.columns
        assert "formation_energy_ev_natom" in df.columns
        assert "bandgap_energy_ev" in df.columns
        assert len(df) > 0
        print("Submission format verification passed.")
    else:
        raise AssertionError("Submission file was not created!")


if __name__ == "__main__":
    try:
        demo_preprocessing()
        loader = demo_data_loading()
        demo_model_forward(loader)
        demo_training_pipeline()
        print("\nAll demos completed successfully!")
    except Exception as e:
        print(f"\nDemo FAILED with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
