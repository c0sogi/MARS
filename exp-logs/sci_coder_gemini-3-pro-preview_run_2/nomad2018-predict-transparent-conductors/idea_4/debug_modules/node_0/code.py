import os
import shutil
import numpy as np
import pandas as pd
import torch
import random

# Import from the provided library
from library.config import Config
from library.utils import GaussianRBF, Standardizer, compute_rmsle
from library.data import get_dataloaders
from library.model import AngleAwareGNN
from library.train import Trainer, generate_submission


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_demo_config():
    """
    Overrides Config attributes to create a lightweight environment for demonstration.
    """
    print("[Demo] Setting up demo configuration...")

    # Paths
    Config.METADATA_DIR = "./working/demo_metadata"
    Config.CACHE_DIR = "./working/demo_cache"

    # Ensure directories exist
    os.makedirs(Config.METADATA_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Reduce model complexity for speed
    Config.EMBEDDING_DIM = 16
    Config.RBF_NUM_BINS = 20
    Config.NUM_BLOCKS = 2

    # Reduce training load
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 2
    Config.PATIENCE = 2
    Config.SCHEDULER_PATIENCE = 1

    # Reduce data processing cost
    Config.MAX_NEIGHBORS = 6
    # We keep CUTOFF_RADIUS as is to find neighbors correctly

    Config.print_config()


def create_subset_metadata():
    """
    Creates a small subset of the metadata files in the working directory
    to allow the data loader to process only a few files.
    """
    print("[Demo] Creating subset metadata...")

    # Source paths (read-only)
    src_train = "./metadata/train_metadata.csv"
    src_val = "./metadata/val_metadata.csv"
    src_test = "./metadata/test_metadata.csv"

    # Destination paths (writable)
    dst_train = os.path.join(Config.METADATA_DIR, "train_metadata.csv")
    dst_val = os.path.join(Config.METADATA_DIR, "val_metadata.csv")
    dst_test = os.path.join(Config.METADATA_DIR, "test_metadata.csv")

    # Read and slice
    # We take 20 samples for train, 10 for val, 10 for test
    pd.read_csv(src_train).head(20).to_csv(dst_train, index=False)
    pd.read_csv(src_val).head(10).to_csv(dst_val, index=False)
    pd.read_csv(src_test).head(10).to_csv(dst_test, index=False)

    print("[Demo] Subset metadata created.")


def test_utils():
    print("\n" + "=" * 40)
    print("Testing Utils")
    print("=" * 40)

    # 1. GaussianRBF
    rbf = GaussianRBF(start=0.0, stop=5.0, n_centers=10)
    dummy_input = torch.tensor([0.0, 2.5, 5.0])
    output = rbf(dummy_input)
    print(f"GaussianRBF Output shape: {output.shape}")
    assert output.shape == (3, 10), "GaussianRBF output shape mismatch"
    # Check that activations are reasonable (close to 1 at centers)
    assert (
        output.max() <= 1.0 and output.min() >= 0.0
    ), "GaussianRBF values out of range"

    # 2. Standardizer
    std = Standardizer(device="cpu")
    data = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    std.fit(data)
    print(f"Standardizer Mean: {std.mean}, Std: {std.std}")

    transformed = std.transform(data)
    # Mean of transformed should be close to 0, std close to 1
    assert torch.allclose(
        transformed.mean(dim=0), torch.zeros(2), atol=1e-6
    ), "Standardization mean failed"

    reconstructed = std.inverse_transform(transformed)
    assert torch.allclose(
        data, reconstructed, atol=1e-6
    ), "Standardizer inverse transform failed"
    print("Standardizer logic verified.")

    # 3. Compute RMSLE
    y_true = torch.tensor([[1.0], [10.0]])
    y_pred = torch.tensor([[1.1], [9.0]])  # Small errors
    rmsle = compute_rmsle(y_true, y_pred)
    print(f"Computed RMSLE: {rmsle:.4f}")
    assert rmsle > 0, "RMSLE should be positive"


def test_data_pipeline():
    print("\n" + "=" * 40)
    print("Testing Data Pipeline")
    print("=" * 40)

    # This will trigger processing of the subset metadata created earlier
    # and save .npz files to ./working/demo_cache
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Inspect one batch
    batch = next(iter(train_loader))
    print("\nSample Batch Inspection:")
    print(f"  Batch size: {batch.num_graphs}")
    print(f"  Node features (x): {batch.x.shape}")
    print(f"  Edge index: {batch.edge_index.shape}")
    print(f"  Edge attr (distances): {batch.edge_attr.shape}")
    print(f"  Line Graph Edge index: {batch.line_edge_index.shape}")
    print(f"  Line Graph Edge attr (angles): {batch.line_edge_attr.shape}")
    print(f"  Targets (y): {batch.y.shape}")

    # Basic assertions
    assert batch.x.dim() == 1, "Node features should be 1D (atomic numbers/indices)"
    assert batch.edge_index.shape[0] == 2, "Edge index should be (2, E)"
    assert batch.y.shape[1] == 2, "Targets should have 2 columns"

    return train_loader, val_loader, test_loader


def test_model(loader):
    print("\n" + "=" * 40)
    print("Testing Model Forward Pass")
    print("=" * 40)

    model = AngleAwareGNN()
    # Move model to CPU for this quick test if CUDA is not forced
    device = "cpu"
    model.to(device)

    batch = next(iter(loader))
    batch = batch.to(device)

    output = model(batch)
    print(f"Model Output Shape: {output.shape}")

    assert output.shape == (batch.num_graphs, 2), "Model output shape mismatch (B, 2)"
    assert not torch.isnan(output).any(), "Model produced NaNs"
    print("Model forward pass successful.")


def test_training_loop(train_loader, val_loader, test_loader):
    print("\n" + "=" * 40)
    print("Testing Training Loop")
    print("=" * 40)

    # Initialize Trainer (uses Config settings we overrode)
    trainer = Trainer()

    # Run training
    trainer.train_loop(train_loader, val_loader, epochs=Config.NUM_EPOCHS)

    # Check if checkpoint was created
    checkpoint_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file not created!"
    print("Training loop completed and checkpoint saved.")

    # Load best model
    trainer.load_best_model()

    # Predict on test set
    ids, preds = trainer.predict(test_loader)
    print(f"Predictions generated for {len(ids)} test samples.")
    assert preds.shape == (len(ids), 2), "Prediction shape mismatch"

    # Generate submission file
    submission_path = "./working/demo_submission/submission.csv"
    generate_submission(trainer, test_loader, output_path=submission_path)
    assert os.path.exists(submission_path), "Submission file not created"


if __name__ == "__main__":
    # 1. Setup
    set_seed(42)
    setup_demo_config()
    create_subset_metadata()

    # 2. Verify Utils
    test_utils()

    # 3. Verify Data Loading
    # We pass load_cached_data=False to force processing of our new subset
    train_loader, val_loader, test_loader = test_data_pipeline()

    # 4. Verify Model
    test_model(train_loader)

    # 5. Verify Training
    test_training_loop(train_loader, val_loader, test_loader)

    print("\n" + "=" * 40)
    print("All demonstrations and verifications passed!")
    print("=" * 40)
