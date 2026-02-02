import sys
import os
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, StandardScaler
from library.data import get_dataloaders, CrystalGraphDataset
from library.model import RBFExpansion, LRCGCNNLayer, LRCGCNN
from library.train import train_one_epoch, evaluate, run_training


def main():
    print("Starting demonstration script...")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    print("\n[1] Configuring for fast demonstration...")

    # Override Config parameters for a fast, isolated execution
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Update derived paths
    Config.TRAIN_GRAPHS_CACHE = os.path.join(Config.CACHE_DIR, "train_graphs.npz")
    Config.VAL_GRAPHS_CACHE = os.path.join(Config.CACHE_DIR, "val_graphs.npz")
    Config.TEST_GRAPHS_CACHE = os.path.join(Config.CACHE_DIR, "test_graphs.npz")
    Config.TARGET_SCALER_CACHE = os.path.join(Config.CACHE_DIR, "target_scaler.npz")
    Config.MODEL_CHECKPOINT = os.path.join(Config.CHECKPOINT_DIR, "demo_model.pth")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Set hyperparameters for speed
    Config.DEBUG = True
    Config.MAX_SAMPLES = 20  # Use only 20 samples per split
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead

    print(f"Working directory set to: {Config.WORKING_DIR}")
    print(f"Debug mode: {Config.DEBUG}")
    print(f"Max samples: {Config.MAX_SAMPLES}")

    # -------------------------------------------------------------------------
    # 2. Utils Demonstration (StandardScaler)
    # -------------------------------------------------------------------------
    print("\n[2] Demonstrating library.utils (StandardScaler)...")
    set_seed(42)

    # Create dummy data: 3 samples, 2 features
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    scaler = StandardScaler()
    scaler.fit(data)

    print(f"Mean: {scaler.mean}")
    print(f"Std:  {scaler.std}")

    # Verification: Mean should be [3.0, 4.0], Std (unbiased) should be [2.0, 2.0]
    expected_mean = torch.tensor([3.0, 4.0])
    expected_std = torch.tensor([2.0, 2.0])

    assert torch.allclose(
        scaler.mean, expected_mean
    ), "Scaler mean calculation incorrect"
    assert torch.allclose(scaler.std, expected_std), "Scaler std calculation incorrect"

    # Transform
    transformed = scaler.transform(data)
    expected_transformed = torch.tensor([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    assert torch.allclose(
        transformed, expected_transformed
    ), "Scaler transform incorrect"

    # Inverse Transform
    inverse = scaler.inverse_transform(transformed)
    assert torch.allclose(inverse, data), "Scaler inverse_transform incorrect"

    print("StandardScaler verification passed.")

    # -------------------------------------------------------------------------
    # 3. Data Demonstration (DataLoaders & Graph Processing)
    # -------------------------------------------------------------------------
    print("\n[3] Demonstrating library.data...")

    # Generate data loaders (this will trigger processing of geometry files)
    # forcing load_cached_data=False to ensure processing logic runs
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        load_cached_data=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Verify batch structure
    sample_batch = next(iter(train_loader))
    print(f"Sample batch: {sample_batch}")

    # Check dimensions
    # x should be atomic numbers (N_nodes)
    assert (
        sample_batch.x.ndim == 1
    ), "Node features should be 1D tensor of atomic numbers"
    # edge_index should be (2, N_edges)
    assert sample_batch.edge_index.shape[0] == 2, "Edge index should have 2 rows"
    # y should be (Batch_Size, 2)
    assert (
        sample_batch.y.shape[1] == 2
    ), "Target should have 2 columns (formation, bandgap)"

    print("Data loading verification passed.")

    # -------------------------------------------------------------------------
    # 4. Model Demonstration (Components & Full Model)
    # -------------------------------------------------------------------------
    print("\n[4] Demonstrating library.model...")

    # Test RBFExpansion
    rbf = RBFExpansion(dmin=0, dmax=5, n_rbf=10)
    dist = torch.tensor([1.0, 2.5, 4.0])
    rbf_out = rbf(dist)
    print(f"RBF output shape: {rbf_out.shape}")
    assert rbf_out.shape == (3, 10), "RBF output shape mismatch"

    # Test LRCGCNNLayer
    atom_fea_len = 32
    edge_fea_len = 32
    n_nodes = 10
    n_edges = 20

    layer = LRCGCNNLayer(atom_fea_len, edge_fea_len)
    x_dummy = torch.randn(n_nodes, atom_fea_len)
    edge_index_dummy = torch.randint(0, n_nodes, (2, n_edges))
    edge_attr_dummy = torch.randn(n_edges, edge_fea_len)

    layer_out = layer(x_dummy, edge_index_dummy, edge_attr_dummy)
    print(f"Layer output shape: {layer_out.shape}")
    assert layer_out.shape == (n_nodes, atom_fea_len), "Layer output shape mismatch"

    # Test Full Model with Real Batch
    model = LRCGCNN(
        atom_fea_len=Config.ATOM_FEA_LEN,
        h_fea_len=Config.H_FEA_LEN,
        n_conv=Config.N_CONV,
        n_h=Config.N_H,
        n_rbf=Config.N_RBF,
        radius=Config.RADIUS,
    )

    # Forward pass
    # Note: The model handles embedding of atomic numbers and RBF expansion of distances internally
    model_out = model(sample_batch)
    print(f"Model output shape: {model_out.shape}")
    assert model_out.shape == (
        sample_batch.num_graphs,
        2,
    ), "Model output shape mismatch"
    print("Model verification passed.")

    # -------------------------------------------------------------------------
    # 5. Training Pipeline Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Demonstrating library.train (Full Pipeline)...")

    # Execute the training run
    # This uses the Config settings we modified (EPOCHS=1, DEBUG=True)
    # It will reload the data from the cache we just created
    try:
        run_training(load_cached_data=True)
        print("Training pipeline execution successful.")
    except Exception as e:
        print(f"Training pipeline failed: {e}")
        raise e

    # Verify output
    if os.path.exists(Config.SUBMISSION_FILE):
        print(f"Submission file created at: {Config.SUBMISSION_FILE}")
        sub_df = pd.read_csv(Config.SUBMISSION_FILE)
        print("Submission head:")
        print(sub_df.head())

        # Check integrity
        assert sub_df.shape[1] == 3, "Submission should have 3 columns"
        assert "id" in sub_df.columns
        assert "formation_energy_ev_natom" in sub_df.columns
        assert "bandgap_energy_ev" in sub_df.columns
        print(f"Submission contains {len(sub_df)} rows.")
    else:
        raise FileNotFoundError("Submission file was not created!")

    print("\nDemonstration complete.")


if __name__ == "__main__":
    main()
