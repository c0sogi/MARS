import os
import torch
import numpy as np
import shutil
from ase import Atoms

# Import from the provided library
from library.config import Config
from library.data import get_dataloaders
from library.model import MS_RA_CGN
from library.train import Trainer
from library.utils import set_seed, TargetScaler, rmsle
from library.features import compute_pbc_radius_graph


def run_demo():
    print("=== Setting up Demo Configuration ===")
    # Override Config for a fast demonstration
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Use a very small subset for speed
    Config.DEBUG_SAMPLE_SIZE = 20
    Config.BATCH_SIZE = 4
    Config.MAX_EPOCHS = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for this tiny demo

    # Set seed for reproducibility
    set_seed(42)
    print("Configuration updated for demo run.")

    print("\n=== Testing Data Loading and Graph Construction ===")
    # Force recomputation to test graph building logic (load_cached_data=False)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify dataset sizes
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Fetch a single batch to inspect
    batch = next(iter(train_loader))
    print(f"Batch structure: {batch}")

    # Assertions for batch structure
    assert batch.x is not None, "Node features (atomic numbers) missing"
    assert batch.edge_index is not None, "Edge indices missing"
    assert batch.edge_attr is not None, "Edge attributes (distances) missing"
    assert batch.y is not None, "Target labels missing"
    assert batch.y.shape == (
        Config.BATCH_SIZE,
        2,
    ), f"Expected target shape ({Config.BATCH_SIZE}, 2), got {batch.y.shape}"
    assert batch.batch is not None, "Batch indices missing"

    print("Data loading and batching verified.")

    print("\n=== Testing Feature Computation Logic (Manual) ===")
    # Create a dummy crystal structure (FCC Aluminum)
    dummy_atoms = Atoms(
        "Al4",
        scaled_positions=[(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)],
        cell=[4.05, 4.05, 4.05],
        pbc=True,
    )

    graph_data = compute_pbc_radius_graph(dummy_atoms, cutoff=5.0, max_neighbors=12)

    print("Computed graph keys:", graph_data.keys())
    assert "edge_index" in graph_data
    assert "edge_dist" in graph_data
    assert graph_data["edge_index"].shape[0] == 2
    assert graph_data["edge_dist"].dim() == 1
    print("Feature computation verified.")

    print("\n=== Testing Model Architecture ===")
    device = torch.device(Config.DEVICE)
    model = MS_RA_CGN().to(device)
    batch = batch.to(device)

    # Forward pass
    output = model(batch)
    print(f"Model output shape: {output.shape}")

    assert output.shape == (
        Config.BATCH_SIZE,
        2,
    ), f"Expected output shape ({Config.BATCH_SIZE}, 2), got {output.shape}"
    print("Model forward pass verified.")

    print("\n=== Testing Utilities (TargetScaler & RMSLE) ===")
    # Test TargetScaler
    scaler = TargetScaler()
    dummy_targets = torch.tensor([[0.1, 1.5], [0.2, 2.0], [0.15, 1.8]], device=device)
    scaler.fit(dummy_targets)

    transformed = scaler.transform(dummy_targets)
    inverse = scaler.inverse_transform(transformed)

    # Check reconstruction
    assert torch.allclose(
        dummy_targets, inverse, atol=1e-5
    ), "TargetScaler inverse transform failed"
    print("TargetScaler verified.")

    # Test RMSLE
    y_true = np.array([[1.0, 10.0], [2.0, 20.0]])
    y_pred = np.array([[1.1, 9.5], [1.9, 21.0]])
    score = rmsle(y_true, y_pred)
    print(f"RMSLE Score: {score:.4f}")
    assert isinstance(score, float), "RMSLE should return a float"
    print("RMSLE function verified.")

    print("\n=== Testing Training Loop ===")
    trainer = Trainer()

    # Fit scaler on the tiny train set
    trainer.fit_scaler(train_loader)

    # Run a short training loop
    print("Running training epochs...")
    for epoch in range(1, Config.MAX_EPOCHS + 1):
        train_loss = trainer.train_one_epoch(train_loader)
        val_loss, val_rmsle = trainer.validate(val_loader)
        print(
            f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val RMSLE={val_rmsle:.4f}"
        )

        # Basic sanity check: loss should be a number
        assert not np.isnan(train_loss), "Training loss is NaN"
        assert not np.isnan(val_loss), "Validation loss is NaN"

    # Save a checkpoint manually to verify directory permissions
    ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "demo_model.pth")
    torch.save(trainer.model.state_dict(), ckpt_path)
    assert os.path.exists(ckpt_path), "Checkpoint file was not created"
    print(f"Checkpoint saved to {ckpt_path}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
