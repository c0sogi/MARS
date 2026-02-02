import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library import config
from library.utils import GaussianSmearing, Standardizer, set_seed
from library.data import CrystalGraphDataset, collate_batch
from library.model import CompositionAwareCGCNN
from library.train import Trainer


def setup_demo_config():
    """
    Modifies the global Config class to use temporary directories and
    reduced hyperparameters for the demonstration.
    """
    print("[Demo] Setting up configuration...")

    # Define demo working directories
    demo_working_dir = "./working/demo_run"
    demo_cache_dir = os.path.join(demo_working_dir, "cache")
    demo_checkpoint_dir = os.path.join(demo_working_dir, "checkpoints")
    demo_submission_dir = "./working/demo_submission"

    # Clean up previous demo run if exists
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    if os.path.exists(demo_submission_dir):
        shutil.rmtree(demo_submission_dir)

    # Create directories
    os.makedirs(demo_cache_dir, exist_ok=True)
    os.makedirs(demo_checkpoint_dir, exist_ok=True)
    os.makedirs(demo_submission_dir, exist_ok=True)

    # Update Config attributes
    config.Config.WORKING_DIR = demo_working_dir
    config.Config.CACHE_DIR = demo_cache_dir
    config.Config.CHECKPOINT_DIR = demo_checkpoint_dir
    config.Config.SUBMISSION_DIR = demo_submission_dir

    # Reduce model complexity and training duration for demo
    config.Config.N_CONV = 2
    config.Config.ATOM_FEA_LEN = 32
    config.Config.EPOCHS = 2
    config.Config.BATCH_SIZE = 8
    config.Config.PATIENCE = 2

    print(f"[Demo] Working Directory: {config.Config.WORKING_DIR}")


def test_utils():
    """
    Demonstrates and verifies utility classes.
    """
    print("\n[Demo] Testing Utils...")

    # 1. Test GaussianSmearing
    smearing = GaussianSmearing(start=0.0, stop=5.0, n_gaussians=10)
    distances = torch.tensor([0.0, 2.5, 5.0])
    features = smearing(distances)

    assert features.shape == (3, 10), f"Expected shape (3, 10), got {features.shape}"
    print("  GaussianSmearing: Output shape verified.")

    # 2. Test Standardizer
    data = torch.randn(100, 5)
    mean = data.mean(dim=0)
    std = data.std(dim=0)

    scaler = Standardizer()
    scaler.fit(data)

    # Check fitted stats
    assert torch.allclose(scaler.mean, mean, atol=1e-5), "Standardizer mean mismatch"
    assert torch.allclose(scaler.std, std, atol=1e-5), "Standardizer std mismatch"

    # Check transform and inverse transform
    transformed = scaler.transform(data)
    assert (
        torch.abs(transformed.mean()).item() < 1e-5
    ), "Transformed data mean should be ~0"
    assert (
        torch.abs(transformed.std() - 1.0).item() < 1e-5
    ), "Transformed data std should be ~1"

    reconstructed = scaler.inverse_transform(transformed)
    assert torch.allclose(
        data, reconstructed, atol=1e-5
    ), "Inverse transform failed to reconstruct data"

    print("  Standardizer: Logic verified.")


def test_data_loading():
    """
    Demonstrates data loading and batch collation.
    """
    print("\n[Demo] Testing Data Loading...")

    # Initialize dataset in debug mode (loads first 50 samples)
    # Note: This will create cache files in the demo cache directory
    dataset = CrystalGraphDataset(
        metadata_path=config.Config.TRAIN_METADATA_PATH,
        split="train",
        load_cached_data=False,  # Force processing for demo
        debug=True,
    )

    print(f"  Dataset size (debug): {len(dataset)}")
    assert len(dataset) > 0, "Dataset is empty"

    # Inspect one sample
    sample = dataset[0]
    print(f"  Sample 0 keys: {sample.keys}")

    # Verify shapes
    # x: (num_nodes, num_atom_types) -> (N, 4) for Al, Ga, In, O
    assert (
        sample.x.shape[1] == 4
    ), f"Node feature dim mismatch. Expected 4, got {sample.x.shape[1]}"
    # edge_attr: (num_edges, RBF_N_BINS)
    assert (
        sample.edge_attr.shape[1] == config.Config.RBF_N_BINS
    ), "Edge feature dim mismatch"
    # global_feat: (10,)
    assert sample.global_feat.shape[-1] == 10, "Global feature dim mismatch"
    # y: (1, 2)
    assert sample.y.shape == (1, 2), "Target shape mismatch"

    # Test Collation
    batch_list = [dataset[i] for i in range(4)]
    batch = collate_batch(batch_list)

    print(f"  Batch size: {batch.num_graphs}")
    assert batch.num_graphs == 4
    assert batch.batch.shape[0] == batch.x.shape[0]

    print("  Data Loading: Verified.")
    return batch


def test_model(batch):
    """
    Demonstrates model instantiation and forward pass.
    """
    print("\n[Demo] Testing Model...")

    model = CompositionAwareCGCNN(config.Config)

    # Move batch to same device as model (CPU for this demo if CUDA not forced)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    batch = batch.to(device)

    # Forward pass
    output = model(batch)

    print(f"  Output shape: {output.shape}")

    # Expected output: (Batch_Size, 2) -> (4, 2)
    assert output.shape == (4, 2), f"Expected output shape (4, 2), got {output.shape}"

    print("  Model: Forward pass verified.")


def run_training_pipeline():
    """
    Demonstrates the full training and prediction pipeline using the Trainer class.
    """
    print("\n[Demo] Running Training Pipeline...")

    # Instantiate Trainer
    # This will initialize the model, optimizer, loss, etc.
    trainer = Trainer(config.Config)

    # Train
    # debug=True ensures we use a small subset of data for speed
    print("  > Starting training loop...")
    trainer.train(
        epochs=config.Config.EPOCHS, batch_size=config.Config.BATCH_SIZE, debug=True
    )

    # Predict
    # Uses the best model saved during training
    print("  > Starting prediction...")
    trainer.predict(batch_size=config.Config.BATCH_SIZE, debug=True)

    # Verify submission file
    submission_path = os.path.join(config.Config.SUBMISSION_DIR, "submission.csv")
    if os.path.exists(submission_path):
        df = pd.read_csv(submission_path)
        print(f"  > Submission file generated at {submission_path}")
        print(f"  > Rows: {len(df)}")
        assert len(df) > 0, "Submission file is empty"
        assert "id" in df.columns, "id column missing"
        assert (
            "formation_energy_ev_natom" in df.columns
        ), "formation energy column missing"
        assert "bandgap_energy_ev" in df.columns, "bandgap energy column missing"
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("  Training Pipeline: Verified.")


if __name__ == "__main__":
    # 1. Setup
    set_seed(42)
    setup_demo_config()

    # 2. Test Utils
    test_utils()

    # 3. Test Data
    # We keep the batch to test the model next
    sample_batch = test_data_loading()

    # 4. Test Model
    test_model(sample_batch)

    # 5. Run Full Pipeline
    run_training_pipeline()

    print("\n[Demo] All demonstrations completed successfully.")
