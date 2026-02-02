import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# 1. Configuration Override
# We must import Config and modify it BEFORE importing other modules that might
# use these values as default arguments or during initialization.
from library.config import Config

# Set up a demo working directory
DEMO_WORKING_DIR = "./working/demo_run"
os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

# Override Config for speed and demonstration purposes
print("Configuring environment for demo run...")
Config.WORKING_DIR = DEMO_WORKING_DIR
Config.CACHE_TRAIN_PATH = os.path.join(DEMO_WORKING_DIR, "cached_train.npz")
Config.CACHE_VAL_PATH = os.path.join(DEMO_WORKING_DIR, "cached_val.npz")
Config.MODEL_SAVE_PATH = os.path.join(DEMO_WORKING_DIR, "best_model.pt")
Config.DEBUG = True
Config.DEBUG_SIZE = 200  # Small subset for speed
Config.BATCH_SIZE = 16
Config.EPOCHS = 1
Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
Config.HIDDEN_DIM = 64  # Reduced capacity
Config.NUM_MP_LAYERS = 2
Config.NUM_TRANSFORMER_LAYERS = 1
Config.NUM_ATTENTION_HEADS = 4
Config.TRANSFORMER_DIM_FEEDFORWARD = 128
Config.RBF_SIZE = 16
Config.SBF_SIZE = 8

# Now import the rest of the library
from library.utils import set_seed, CouplingStandardizer
from library.dataset import ChampsDataset, collate_graphs
from library.model import HGANet
from library.engine import train_one_epoch, evaluate, predict


def main():
    # Set seed for reproducibility
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # -------------------------------------------------------------------------
    # 2. Dataset & Data Loading
    # -------------------------------------------------------------------------
    print("\n--- Initializing Dataset ---")
    # We use the training metadata. In a real run, we would use separate train/val splits.
    # For this demo, we use the same small subset for everything.
    train_metadata_path = Config.TRAIN_META_PATH

    # Instantiate dataset
    # This will trigger process_dataset, which parses XYZ files and builds graphs
    dataset = ChampsDataset(
        metadata_path=train_metadata_path,
        cache_path=Config.CACHE_TRAIN_PATH,
        split="train",
        debug=Config.DEBUG,
        debug_size=Config.DEBUG_SIZE,
    )

    print(f"Dataset size: {len(dataset)}")

    # Verification: Check dataset integrity
    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check metadata paths or debug size.")

    sample_data = dataset[0]
    print(f"Sample Graph Keys: {sample_data.keys}")

    # Verify essential graph attributes
    assert hasattr(sample_data, "x"), "Graph missing atom features 'x'"
    assert hasattr(sample_data, "edge_index"), "Graph missing 'edge_index'"
    assert hasattr(
        sample_data, "coupling_atom_0"
    ), "Graph missing task specific 'coupling_atom_0'"
    assert sample_data.x.dim() == 1, "Atom features should be 1D (atomic numbers)"
    assert sample_data.edge_index.shape[0] == 2, "Edge index should be [2, NumEdges]"

    # Create DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=Config.NUM_WORKERS,
    )

    # Verification: Check Batching Logic
    batch = next(iter(train_loader))
    print(f"\nBatch Information:")
    print(f"  Num Graphs: {batch.num_graphs}")
    print(f"  Num Nodes: {batch.num_nodes}")
    print(f"  Num Edges: {batch.num_edges}")

    # Verify coupling indices are shifted correctly in the batch
    # The max index in coupling_atom_0 should be less than the total number of nodes in batch
    assert (
        batch.coupling_atom_0.max() < batch.num_nodes
    ), "Coupling indices out of bounds after batching"
    assert (
        batch.coupling_atom_0.shape[0] == batch.num_graphs
    ), "One coupling pair per graph expected"

    # -------------------------------------------------------------------------
    # 3. Pre-processing (Standardization)
    # -------------------------------------------------------------------------
    print("\n--- Fitting Standardizer ---")
    standardizer = CouplingStandardizer()
    standardizer.fit(dataset.df)

    # Verify standardizer
    sample_type = dataset.df.iloc[0]["type"]
    mean, std = standardizer.get_params(sample_type)
    print(f"  Type {sample_type}: Mean={mean:.4f}, Std={std:.4f}")
    assert std > 0, "Standard deviation should be positive"

    # -------------------------------------------------------------------------
    # 4. Model Initialization
    # -------------------------------------------------------------------------
    print("\n--- Initializing Model ---")
    model = HGANet().to(device)

    # Verification: Forward pass on a single batch
    model.eval()
    with torch.no_grad():
        batch = batch.to(device)
        output = model(batch)

    print(f"  Output Shape: {output.shape}")
    assert output.shape == (
        batch.num_graphs,
        1,
    ), f"Expected output shape [B, 1], got {output.shape}"

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Starting Training Demo ---")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Train for one epoch
    avg_loss = train_one_epoch(model, train_loader, optimizer, device, standardizer)
    print(f"  Epoch 1 Loss: {avg_loss:.6f}")

    assert not np.isnan(avg_loss), "Training loss is NaN"
    assert avg_loss >= 0, "MAE loss should be non-negative"

    # -------------------------------------------------------------------------
    # 6. Evaluation Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Starting Evaluation Demo ---")
    # Using train_loader as val_loader for demonstration
    val_metric = evaluate(model, train_loader, device, standardizer)
    print(f"  Validation LogMAE: {val_metric:.6f}")

    # Metric can be negative (log of small error), just checking it's a number
    assert isinstance(val_metric, (float, np.floating)), "Metric should be a float"

    # -------------------------------------------------------------------------
    # 7. Prediction Demonstration
    # -------------------------------------------------------------------------
    print("\n--- Starting Prediction Demo ---")
    # Using train_loader as test_loader for demonstration
    submission_df = predict(model, train_loader, device, standardizer)

    print("  Predictions Generated:")
    print(submission_df.head())

    # Verification: Submission format
    assert "id" in submission_df.columns, "Submission missing 'id' column"
    assert (
        "scalar_coupling_constant" in submission_df.columns
    ), "Submission missing target column"
    assert len(submission_df) == len(dataset), "Submission length mismatch"
    assert (
        not submission_df["scalar_coupling_constant"].isnull().any()
    ), "Predictions contain NaNs"

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    main()
