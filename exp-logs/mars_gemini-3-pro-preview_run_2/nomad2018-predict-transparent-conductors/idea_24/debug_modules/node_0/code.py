import os
import shutil
import numpy as np
import torch
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, compute_rmsle, TargetScaler
from library.data import get_dataloaders
from library.model import kRACGN
from library.train import Trainer


def run_demo():
    print("=" * 80)
    print("Running k-RA-CGN Implementation Demo")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("\n[1] Setting up Configuration...")

    # Override Config parameters for a fast demonstration
    Config.WORKING_DIR = "./working/demo_run"

    # Define subdirectory paths for the demo
    cache_dir = os.path.join(Config.WORKING_DIR, "cache")
    ckpt_dir = os.path.join(Config.WORKING_DIR, "checkpoints")
    sub_dir = os.path.join(Config.WORKING_DIR, "submission")

    # Ensure directories exist
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sub_dir, exist_ok=True)

    # Update Config paths
    Config.TRAIN_GRAPHS_PATH = os.path.join(cache_dir, "train_graphs.npz")
    Config.VAL_GRAPHS_PATH = os.path.join(cache_dir, "val_graphs.npz")
    Config.TEST_GRAPHS_PATH = os.path.join(cache_dir, "test_graphs.npz")
    Config.TARGET_SCALER_PATH = os.path.join(cache_dir, "target_scaler.npz")
    Config.CHECKPOINT_PATH = os.path.join(ckpt_dir, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(sub_dir, "submission.csv")

    # Set hyperparameters for speed
    Config.DEBUG_SAMPLE_SIZE = 50  # Only process 50 samples per split
    Config.NUM_EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.K_NEIGHBORS = 6  # Reduced neighbors for speed

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"Num Epochs: {Config.NUM_EPOCHS}")

    # Set random seed
    set_seed(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Verify Utilities
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utilities...")

    # Test RMSLE computation
    y_true = np.array([[1.0, 10.0], [2.0, 20.0]])
    y_pred = np.array([[1.1, 9.5], [2.1, 21.0]])

    # Manual calculation
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    sq_diff = (log_pred - log_true) ** 2
    mse = np.mean(sq_diff, axis=0)
    rmse = np.sqrt(mse)
    expected_rmsle = np.mean(rmse)

    computed_rmsle = compute_rmsle(y_true, y_pred)
    print(f"Computed RMSLE: {computed_rmsle:.6f}")
    assert np.isclose(computed_rmsle, expected_rmsle), "RMSLE calculation mismatch!"
    print("RMSLE check passed.")

    # Test TargetScaler
    scaler = TargetScaler()
    dummy_data = np.random.randn(100, 2) * 5 + 10  # Mean ~10, Std ~5
    scaler.fit(dummy_data)

    transformed = scaler.transform(dummy_data)
    assert np.allclose(transformed.mean(axis=0), 0, atol=1e-6), "Scaler mean not zero"
    assert np.allclose(transformed.std(axis=0), 1, atol=1e-6), "Scaler std not one"

    reconstructed = scaler.inverse_transform(transformed)
    assert np.allclose(
        dummy_data, reconstructed, atol=1e-6
    ), "Scaler reconstruction failed"
    print("TargetScaler check passed.")

    # -------------------------------------------------------------------------
    # 3. Data Loading and Processing
    # -------------------------------------------------------------------------
    print("\n[3] Processing Data and Creating DataLoaders...")
    # Force re-processing by setting load_cached_data=False (or ensure cache is empty)
    # Since we changed paths to a new demo dir, it will process from scratch.
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        load_cached_data=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    # Inspect a single batch
    sample_batch = next(iter(train_loader))
    print("\nSample Batch Structure:")
    print(f"  Batch size: {sample_batch.num_graphs}")
    print(f"  Node features (x) shape: {sample_batch.x.shape}")  # (num_nodes, )
    print(f"  Edge index shape: {sample_batch.edge_index.shape}")  # (2, num_edges)
    print(f"  Edge attr shape: {sample_batch.edge_attr.shape}")  # (num_edges, 1)
    print(f"  Targets (y) shape: {sample_batch.y.shape}")  # (batch_size, 2)

    # Basic assertions on data structure
    assert sample_batch.x.ndim == 1, "Node features should be 1D (atomic numbers)"
    assert sample_batch.edge_index.shape[0] == 2, "Edge index should have 2 rows"
    assert (
        sample_batch.y.shape[1] == 2
    ), "Targets should have 2 columns (formation, bandgap)"
    print("Data loading check passed.")

    # -------------------------------------------------------------------------
    # 4. Model Instantiation
    # -------------------------------------------------------------------------
    print("\n[4] Instantiating Model...")
    model = kRACGN()

    # Move batch to device
    device = Config.DEVICE
    model.to(device)
    sample_batch = sample_batch.to(device)

    # Forward pass check
    with torch.no_grad():
        output = model(sample_batch)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (sample_batch.num_graphs, 2), "Model output shape mismatch"
    print("Model instantiation and forward pass check passed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[5] Running Training Loop...")

    trainer = Trainer(model, train_loader, val_loader, scaler, Config)
    trainer.run()

    # Check if checkpoint was saved
    if os.path.exists(Config.CHECKPOINT_PATH):
        print(f"\nCheckpoint successfully saved at: {Config.CHECKPOINT_PATH}")
    else:
        # It's possible validation didn't improve in 2 epochs if initialized well,
        # but usually it saves at least once.
        print(
            "\nWarning: No checkpoint found (validation loss might not have decreased)."
        )

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
