import sys

# Force reload of library modules to ensure the fix in data.py is picked up
for key in list(sys.modules.keys()):
    if key.startswith("library"):
        del sys.modules[key]

import os
import torch
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, StandardScaler, save_checkpoint, load_checkpoint
from library.data import get_dataloaders, CrystalDataset
from library.model import RA_CGN_AR
from library.engine import run_training


def demo_configuration():
    print("\n=== 1. Configuring for Demo ===")
    # Modify Config parameters for a quick demonstration run
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.SUBSET_SIZE = 20  # Use only 20 samples

    # Redirect outputs to a demo directory
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Create directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Epochs: {Config.NUM_EPOCHS}, Batch Size: {Config.BATCH_SIZE}")


def demo_utils():
    print("\n=== 2. Testing Utils ===")

    # Test Reproducibility
    set_seed(42)
    r1 = torch.rand(5)
    set_seed(42)
    r2 = torch.rand(5)
    assert torch.allclose(r1, r2), "set_seed did not ensure reproducibility"
    print("Seed verification passed.")

    # Test StandardScaler
    scaler = StandardScaler(device="cpu")
    data = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
    scaler.fit(data)

    print(f"Scaler Mean: {scaler.mean}")
    print(f"Scaler Std: {scaler.std}")

    transformed = scaler.transform(data)
    print(f"Transformed Data Mean (should be ~0): {transformed.mean(dim=0)}")
    print(f"Transformed Data Std (should be ~1): {transformed.std(dim=0)}")

    inverse = scaler.inverse_transform(transformed)
    assert torch.allclose(data, inverse, atol=1e-5), "Inverse transform failed"
    print("StandardScaler verification passed.")


def demo_data_loading():
    print("\n=== 3. Testing Data Loading and Processing ===")
    # We use a small subset to speed up graph construction
    # load_cached_data=False forces it to process from raw xyz files
    train_loader, val_loader, test_loader = get_dataloaders(
        subset_size=Config.SUBSET_SIZE, load_cached_data=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Inspect one batch
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Batch size (graphs): {batch.num_graphs}")
    print(f"Node features shape (x): {batch.x.shape}")
    print(f"Edge index shape: {batch.edge_index.shape}")
    print(f"Edge attr shape: {batch.edge_attr.shape}")
    print(f"Target shape (y): {batch.y.shape}")

    # Assertions
    assert batch.x.ndim == 1, "Node features should be 1D (atomic numbers)"
    assert batch.edge_index.shape[0] == 2, "Edge index should have 2 rows"
    assert batch.y.shape[1] == 2, "Target should have 2 columns (formation, bandgap)"

    return batch


def demo_model(sample_batch):
    print("\n=== 4. Testing Model Architecture ===")
    device = "cpu"  # Use CPU for simple demo
    model = RA_CGN_AR().to(device)
    sample_batch = sample_batch.to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(sample_batch)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        sample_batch.num_graphs,
        2,
    ), f"Expected output shape {(sample_batch.num_graphs, 2)}, got {output.shape}"
    print("Model forward pass successful.")


def demo_full_training():
    print("\n=== 5. Running Full Training Loop ===")
    # This function inside library.engine encapsulates the entire pipeline:
    # Data loading -> Scaling -> Model Init -> Loop (Train/Val) -> Checkpointing

    # Ensure we force re-processing or use the cache we just might have created.
    # Since we changed Config paths, we should let it process again or find the cache there.
    # We pass subset_size to keep it fast.

    try:
        run_training(subset_size=Config.SUBSET_SIZE, load_cached_data=True)
        print("\nTraining loop executed successfully.")
    except Exception as e:
        print(f"\nTraining loop failed with error: {e}")
        raise e

    # Verify checkpoint creation
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"Checkpoint found at: {best_model_path}")
    else:
        raise FileNotFoundError("Best model checkpoint was not created.")


if __name__ == "__main__":
    # 1. Setup Configuration
    demo_configuration()

    # 2. Test Utilities
    demo_utils()

    # 3. Test Data Pipeline
    # We capture a batch to use for model testing
    sample_batch = demo_data_loading()

    # 4. Test Model
    demo_model(sample_batch)

    # 5. Run Training Engine
    demo_full_training()

    print("\n=== Demo Completed Successfully ===")
