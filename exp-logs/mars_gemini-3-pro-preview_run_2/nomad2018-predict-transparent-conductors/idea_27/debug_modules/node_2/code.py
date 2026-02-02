import os
import shutil
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, StandardScaler, compute_metric
from library.data import get_dataloaders, process_structure
from library.model import SPRACGN
from library.train import Trainer, train_model, generate_submission


def run_demo():
    print("Initializing Demo...")

    # 1. Configuration Override for Demo
    # We use a separate working directory to avoid conflicts with main runs
    # and to ensure we process a small subset of data fresh.
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)

    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    Config.SUBMISSION_DIR = "./working/demo_submission"

    # Update file paths to point to the demo cache
    Config.TRAIN_GRAPHS_PATH = os.path.join(Config.CACHE_DIR, "train_graphs.npz")
    Config.VAL_GRAPHS_PATH = os.path.join(Config.CACHE_DIR, "val_graphs.npz")
    Config.TEST_GRAPHS_PATH = os.path.join(Config.CACHE_DIR, "test_graphs.npz")
    Config.TARGET_SCALER_PATH = os.path.join(Config.CACHE_DIR, "target_scaler.npz")
    Config.MODEL_CHECKPOINT_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_FILE_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce hyperparameters for speed
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Setup directories
    Config.setup()
    set_seed(Config.SEED)

    print("\n" + "=" * 40)
    print("1. Testing Utilities")
    print("=" * 40)

    # Test StandardScaler
    scaler = StandardScaler(device="cpu")
    dummy_data = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    scaler.fit(dummy_data)

    # Check mean and std
    expected_mean = torch.tensor([2.0, 20.0])
    expected_std = torch.tensor(
        [1.0, 10.0]
    )  # Population std for N=3? torch.std is sample std (N-1)
    # torch.std([1,2,3]) = 1.0

    assert torch.allclose(
        scaler.mean, expected_mean
    ), f"Scaler mean mismatch: {scaler.mean}"
    assert torch.allclose(
        scaler.std, expected_std
    ), f"Scaler std mismatch: {scaler.std}"

    transformed = scaler.transform(dummy_data)
    reverted = scaler.inverse_transform(transformed)
    assert torch.allclose(
        reverted, dummy_data, atol=1e-5
    ), "Scaler inverse transform failed"
    print("StandardScaler passed.")

    # Test Metric
    y_true = np.array([[1.0, 10.0]])
    y_pred = np.array([[1.0, 10.0]])
    metric = compute_metric(y_true, y_pred)
    assert metric < 1e-6, f"Metric should be 0 for identical inputs, got {metric}"
    print("Metric computation passed.")

    print("\n" + "=" * 40)
    print("2. Testing Data Loading (Subset)")
    print("=" * 40)

    # Load a tiny subset of data
    subset_size = 10
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False,  # Force processing from raw files
        dataset_size=subset_size,
    )

    print(f"Train loader batches: {len(train_loader)}")

    # Verify batch structure
    sample_batch = next(iter(train_loader))
    print(f"Sample batch: {sample_batch}")
    assert sample_batch.z is not None, "Batch missing atomic numbers (z)"
    assert sample_batch.edge_index is not None, "Batch missing edge_index"
    assert (
        sample_batch.y.shape[1] == 2
    ), f"Target shape mismatch, expected (N, 2), got {sample_batch.y.shape}"
    print("Data loading passed.")

    print("\n" + "=" * 40)
    print("3. Testing Model Architecture")
    print("=" * 40)

    model = SPRACGN()
    # Move model to CPU for simple testing if CUDA not forced
    model.to("cpu")
    sample_batch = sample_batch.to("cpu")

    # Forward pass
    output = model(sample_batch)
    print(f"Model output shape: {output.shape}")

    expected_batch_size = sample_batch.num_graphs
    assert output.shape == (
        expected_batch_size,
        2,
    ), f"Output shape mismatch: {output.shape}"
    print("Model forward pass passed.")

    print("\n" + "=" * 40)
    print("4. Testing Training Loop")
    print("=" * 40)

    # We use the train_model wrapper which handles everything
    # Using a slightly larger subset for training test to ensure we have enough for batches
    train_model(load_cached_data=True, dataset_size=20, epochs=2)

    assert os.path.exists(Config.MODEL_CHECKPOINT_PATH), "Model checkpoint not created"
    assert os.path.exists(Config.TARGET_SCALER_PATH), "Scaler state not saved"
    print("Training loop executed successfully.")

    print("\n" + "=" * 40)
    print("5. Testing Inference and Submission")
    print("=" * 40)

    generate_submission(load_cached_data=True, dataset_size=10)

    assert os.path.exists(Config.SUBMISSION_FILE_PATH), "Submission file not created"

    df_sub = pd.read_csv(Config.SUBMISSION_FILE_PATH)
    print(f"Submission head:\n{df_sub.head()}")
    assert (
        len(df_sub) == 10
    ), f"Submission length mismatch. Expected 10, got {len(df_sub)}"
    assert "formation_energy_ev_natom" in df_sub.columns
    assert "bandgap_energy_ev" in df_sub.columns
    print("Inference pipeline passed.")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
