import os
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import Config
from library.utils import GaussianSmearing, TargetScaler
from library.data import get_data_loaders
from library.layers import SDG_CFC
from library.trainer import ModelTrainer


def setup_demo_config():
    """
    Overrides Config parameters to run a fast, isolated demo.
    """
    print(">>> Setting up Demo Configuration...")

    # 1. Set paths to a temporary working directory
    Config.WORK_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "processed")

    # Update dependent paths manually since they were initialized at import time
    Config.TRAIN_CACHE = os.path.join(Config.CACHE_DIR, "train_data.pt")
    Config.VAL_CACHE = os.path.join(Config.CACHE_DIR, "val_data.pt")
    Config.TEST_CACHE = os.path.join(Config.CACHE_DIR, "test_data.pt")
    Config.STATS_CACHE = os.path.join(Config.CACHE_DIR, "stats.pt")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORK_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORK_DIR, "submission.csv")

    # 2. Reduce compute requirements for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 50  # Process only 50 molecules
    Config.BATCH_SIZE = 8
    Config.MAX_EPOCHS = 1  # Run only 1 epoch
    Config.NUM_LAYERS = 2  # Reduce model depth
    Config.HIDDEN_CHANNELS = 64  # Reduce model width

    # Ensure directories exist
    os.makedirs(Config.WORK_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set seeds via Config.setup
    Config.setup()
    print(f"    Work Dir: {Config.WORK_DIR}")
    print(f"    Debug Mode: {Config.DEBUG}")


def test_gaussian_smearing():
    """
    Verifies the GaussianSmearing utility.
    """
    print("\n>>> Testing GaussianSmearing...")
    start, stop, n_gaussians = 0.0, 5.0, 10
    smearing = GaussianSmearing(start=start, stop=stop, num_gaussians=n_gaussians)

    # Create dummy distances
    dist = torch.tensor([0.0, 2.5, 5.0], dtype=torch.float32)

    # Forward pass
    out = smearing(dist)

    # Checks
    assert out.shape == (
        3,
        n_gaussians,
    ), f"Expected shape (3, {n_gaussians}), got {out.shape}"
    assert torch.all(out >= 0) and torch.all(
        out <= 1.0
    ), "RBF values should be in [0, 1]"
    print("    GaussianSmearing test passed.")


def test_target_scaler():
    """
    Verifies TargetScaler logic.
    """
    print("\n>>> Testing TargetScaler...")
    scaler = TargetScaler(device="cpu")

    # Create dummy data
    # Type 0: Mean 10, Std 2
    # Type 1: Mean 100, Std 5
    type_0_vals = np.random.normal(10, 2, 100)
    type_1_vals = np.random.normal(100, 5, 100)

    df = pd.DataFrame(
        {
            "type": [Config.COUPLING_TYPES[0]] * 100 + [Config.COUPLING_TYPES[1]] * 100,
            "scalar_coupling_constant": np.concatenate([type_0_vals, type_1_vals]),
        }
    )

    # Fit
    scaler.fit(df)

    # Check fitted stats
    mean_0 = scaler.means[Config.COUPLING_TYPES[0]]
    assert 9.0 < mean_0 < 11.0, "Fitted mean for type 0 incorrect"

    # Transform
    targets = torch.tensor([10.0, 100.0], dtype=torch.float32)
    type_indices = torch.tensor([0, 1], dtype=torch.long)  # Indices for type 0 and 1

    scaled = scaler.transform(targets, type_indices)

    # Expect scaled values close to 0
    assert torch.allclose(scaled, torch.zeros_like(scaled), atol=1.0), "Scaling failed"

    # Inverse Transform
    restored = scaler.inverse_transform(scaled, type_indices)
    assert torch.allclose(restored, targets, atol=1e-5), "Inverse transform failed"

    print("    TargetScaler test passed.")


def test_data_pipeline():
    """
    Tests data loading and batch structure.
    """
    print("\n>>> Testing Data Pipeline (get_data_loaders)...")

    # Force reload to ensure we use the debug subset
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)
        os.makedirs(Config.CACHE_DIR)

    train_loader, val_loader, test_loader, scaler = get_data_loaders(
        batch_size=Config.BATCH_SIZE, debug=Config.DEBUG, load_cached_data=False
    )

    assert len(train_loader) > 0, "Train loader is empty"

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify Batch attributes
    print(f"    Batch retrieved. Graphs: {batch.num_graphs}, Nodes: {batch.num_nodes}")
    assert hasattr(batch, "x"), "Batch missing node features 'x'"
    assert hasattr(batch, "edge_index"), "Batch missing 'edge_index'"
    assert hasattr(batch, "coupling_index"), "Batch missing 'coupling_index'"
    assert hasattr(batch, "y"), "Batch missing target 'y'"

    # Check shapes
    # x should be [Num_Nodes] (atom types)
    assert batch.x.dim() == 1, "Node features should be 1D (atom indices)"
    # coupling_index should be [2, Num_Couplings]
    assert batch.coupling_index.shape[0] == 2, "Coupling index should be [2, K]"

    return batch, scaler


def test_model_forward(batch):
    """
    Tests the SDG_CFC model forward pass.
    """
    print("\n>>> Testing Model Forward Pass...")

    device = Config.DEVICE
    model = SDG_CFC().to(device)
    batch = batch.to(device)

    # Forward
    pred_c, pred_s, pred_q = model(batch)

    # Check Coupling Predictions
    # Should match number of couplings in batch
    num_couplings = batch.coupling_index.shape[1]
    assert (
        pred_c.shape[0] == num_couplings
    ), f"Prediction shape mismatch. Expected {num_couplings}, got {pred_c.shape[0]}"

    # Check Auxiliary Predictions
    # Shielding: [Num_Nodes, 9]
    assert pred_s.shape == (batch.num_nodes, 9), "Shielding prediction shape mismatch"
    # Charge: [Num_Nodes, 1]
    assert pred_q.shape == (batch.num_nodes, 1), "Charge prediction shape mismatch"

    print("    Model forward pass passed.")


def run_full_training_cycle():
    """
    Runs the ModelTrainer to verify the full loop.
    """
    print("\n>>> Running Full Training Cycle (Trainer)...")

    trainer = ModelTrainer(debug=Config.DEBUG, load_cached_data=True)
    trainer.run()

    # Verify submission
    if os.path.exists(Config.SUBMISSION_PATH):
        df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"    Submission generated with {len(df)} rows.")
        assert len(df) > 0, "Submission file is empty"
        assert (
            "id" in df.columns and "scalar_coupling_constant" in df.columns
        ), "Invalid submission format"
    else:
        raise FileNotFoundError("Submission file was not generated.")


if __name__ == "__main__":
    # 0. Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # 1. Setup
    setup_demo_config()

    # 2. Unit Tests
    test_gaussian_smearing()
    test_target_scaler()

    # 3. Integration Tests
    batch, scaler = test_data_pipeline()
    test_model_forward(batch)

    # 4. Full Run
    run_full_training_cycle()

    print("\n>>> Demo Completed Successfully.")
