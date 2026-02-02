import os
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, TargetScaler
from library.model import CrystalGraphResNet, GaussianRBF
from library.train import run_training
from torch_geometric.data import Data, Batch


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    warnings.filterwarnings("ignore")
    set_seed(42)

    print("Initializing demonstration...")

    # Modify Config for the demo to ensure speed and isolation from previous runs
    # We use a separate working directory for this demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce training hyperparameters for a quick execution
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 32
    Config.NUM_WORKERS = (
        0  # Use main process to avoid potential multiprocessing overhead in demo
    )

    # Ensure directories exist
    Config.prepare_directories()
    print(f"Configuration updated. Working directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Unit Testing Utility Classes
    # -------------------------------------------------------------------------
    print("\n--- Testing TargetScaler ---")
    scaler = TargetScaler(device="cpu")
    # Create dummy target data: 3 samples, 2 targets
    dummy_targets = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    # Fit scaler
    scaler.fit(dummy_targets)

    # Check computed statistics
    # Mean should be [2.0, 20.0]
    expected_mean = torch.tensor([2.0, 20.0])
    # Std (population) for [1,2,3] is 0.816, sample std is 1.0. torch.std calculates sample std.
    expected_std = torch.tensor([1.0, 10.0])

    assert torch.allclose(
        scaler.mean, expected_mean
    ), f"Scaler mean mismatch: {scaler.mean}"
    assert torch.allclose(
        scaler.std, expected_std
    ), f"Scaler std mismatch: {scaler.std}"

    # Test Transform
    transformed = scaler.transform(dummy_targets)
    expected_transformed = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    assert torch.allclose(
        transformed, expected_transformed
    ), "Scaler transform mismatch"

    # Test Inverse Transform
    inverted = scaler.inverse_transform(transformed)
    assert torch.allclose(inverted, dummy_targets), "Scaler inverse transform mismatch"
    print("TargetScaler tests passed.")

    # -------------------------------------------------------------------------
    # 3. Unit Testing Model Components
    # -------------------------------------------------------------------------
    print("\n--- Testing Model Components ---")

    # Test GaussianRBF
    # Expands scalar distances into a vector of RBF values
    rbf = GaussianRBF(start=0.0, stop=5.0, num_gaussians=10)
    distances = torch.tensor([0.0, 2.5, 5.0])  # 3 edges
    rbf_out = rbf(distances)
    # Expected shape: [num_edges, num_gaussians] -> [3, 10]
    assert rbf_out.shape == (3, 10), f"RBF output shape mismatch: {rbf_out.shape}"
    print("GaussianRBF shape check passed.")

    # Test Full Model Forward Pass
    # Initialize model with current Config
    model = CrystalGraphResNet(config=Config)
    model.eval()

    # Create a dummy batch of 2 graphs to simulate a DataLoader batch
    # Graph 1: 2 nodes (H, C), 2 edges (0->1, 1->0)
    x1 = torch.tensor([1, 6], dtype=torch.long)
    edge_index1 = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr1 = torch.tensor([1.5, 1.5], dtype=torch.float).unsqueeze(1)
    y1 = torch.tensor([[0.5, 1.5]], dtype=torch.float)  # Dummy targets

    # Graph 2: 3 nodes (O, H, H), 4 edges (connectivity)
    x2 = torch.tensor([8, 1, 1], dtype=torch.long)
    edge_index2 = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], dtype=torch.long)
    edge_attr2 = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float).unsqueeze(1)
    y2 = torch.tensor([[0.2, 2.5]], dtype=torch.float)

    # Wrap in Data objects
    data1 = Data(x=x1, edge_index=edge_index1, edge_attr=edge_attr1, y=y1)
    data2 = Data(x=x2, edge_index=edge_index2, edge_attr=edge_attr2, y=y2)

    # Create batch
    batch = Batch.from_data_list([data1, data2])

    # Forward pass
    with torch.no_grad():
        out = model(batch)

    # Output should be [batch_size, num_targets] -> [2, 2]
    assert out.shape == (
        2,
        2,
    ), f"Model output shape mismatch: {out.shape}, expected (2, 2)"
    print("Model forward pass passed.")

    # -------------------------------------------------------------------------
    # 4. Integration Test: Full Training Pipeline
    # -------------------------------------------------------------------------
    print("\n--- Running Full Training Pipeline (1 Epoch) ---")
    print("This will process data, train, validate, and generate predictions.")

    # We set load_cached_data=False to force the processing of raw XYZ files
    # This verifies the data loading logic works with the provided input files.
    submission_df = run_training(
        num_epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        load_cached_data=False,
    )

    # -------------------------------------------------------------------------
    # 5. Validation of Artifacts
    # -------------------------------------------------------------------------
    print("\n--- Validating Output Artifacts ---")

    # 1. Check Submission File
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file generated at: {Config.SUBMISSION_PATH}")
        df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission shape: {df.shape}")

        # Verify columns
        expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
        assert (
            list(df.columns) == expected_cols
        ), f"Submission columns mismatch. Found: {list(df.columns)}"

        # Verify IDs against test metadata
        test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
        assert len(df) == len(
            test_meta
        ), f"Submission row count ({len(df)}) matches test set ({len(test_meta)})"
        assert set(df["id"]) == set(
            test_meta["id"]
        ), "Submission IDs match test set IDs"

        # Verify values are numeric and not NaN
        assert not df.isnull().values.any(), "Submission contains NaN values"
        assert np.issubdtype(
            df["formation_energy_ev_natom"].dtype, np.number
        ), "Formation energy is not numeric"
        assert np.issubdtype(
            df["bandgap_energy_ev"].dtype, np.number
        ), "Bandgap energy is not numeric"

        print("Submission file content validated.")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    # 2. Check Model Checkpoint
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"Model checkpoint generated at: {checkpoint_path}")
        # Verify it can be loaded
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        assert (
            "embedding.weight" in state_dict
        ), "Checkpoint seems invalid (missing embedding weights)"
    else:
        raise FileNotFoundError("Model checkpoint was not generated.")

    # 3. Check Scaler Cache
    scaler_path = os.path.join(Config.CACHE_DIR, "target_scaler.npz")
    if os.path.exists(scaler_path):
        print(f"Target Scaler saved at: {scaler_path}")
    else:
        raise FileNotFoundError("Target scaler was not saved.")

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
