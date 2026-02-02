import os
import shutil
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, Standardizer
from library.data import get_pbc_graph, CrystalDataset, get_dataloaders
from library.model import GaussianSmearing, AmplifiedCGCNNConv, HASCNet
from library.train import run_training, generate_submission


def run_demo():
    print("--- Starting Demonstration ---")

    # 1. Setup Configuration for Speed
    # We modify the Config class attributes directly to control the execution flow
    print("\n[1] Configuring for fast execution...")
    Config.DEBUG_DATA_LIMIT = 20  # Limit dataset size to 20 samples for speed
    Config.NUM_EPOCHS = 2  # Train for only 2 epochs

    # Ensure a clean working directory for this run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Working directory: {Config.WORKING_DIR}")
    print(f"Debug limit: {Config.DEBUG_DATA_LIMIT}")
    print(f"Epochs: {Config.NUM_EPOCHS}")

    # 2. Test Utilities
    print("\n[2] Testing Utilities...")
    set_seed(42)

    # Test Standardizer logic
    dummy_data = torch.randn(100, 5)
    scaler = Standardizer()
    scaler.fit(dummy_data)
    transformed = scaler.transform(dummy_data)

    # Check standardization properties (mean ~ 0, std ~ 1)
    mean_diff = torch.abs(transformed.mean(dim=0)).max().item()
    std_diff = torch.abs(transformed.std(dim=0) - 1.0).max().item()

    print(f"Standardizer Mean Diff from 0: {mean_diff:.6f}")
    print(f"Standardizer Std Diff from 1: {std_diff:.6f}")

    if mean_diff > 1e-5 or std_diff > 1e-5:
        raise AssertionError("Standardizer failed to normalize data correctly.")

    # Check inverse transform
    reconstructed = scaler.inverse_transform(transformed)
    recon_diff = torch.abs(reconstructed - dummy_data).max().item()
    if recon_diff > 1e-5:
        raise AssertionError("Standardizer inverse transform failed.")
    print("Standardizer verified.")

    # 3. Test Data Processing
    print("\n[3] Testing Data Processing...")
    # Ensure metadata exists
    train_meta_path = os.path.join(Config.METADATA_DIR, "train_metadata.csv")
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(
            "Metadata not found. Please ensure metadata is generated."
        )

    # Load a sample to test graph construction
    train_df = pd.read_csv(train_meta_path)
    sample_row = train_df.iloc[0]
    file_path = sample_row["file_path"]
    mat_id = sample_row["id"]
    targets = [sample_row["formation_energy_ev_natom"], sample_row["bandgap_energy_ev"]]

    # Test get_pbc_graph
    data = get_pbc_graph(file_path, mat_id, targets)
    print(
        f"Graph created for ID {mat_id}. Nodes: {data.x.shape[0]}, Edges: {data.edge_index.shape[1]}"
    )

    # Validate graph structure
    if data.x.dim() != 1:
        raise AssertionError("Node features should be 1D (atomic numbers).")
    if data.global_x.shape[1] != Config.GLOBAL_FEATURE_DIM:
        raise AssertionError(
            f"Global features dim mismatch. Expected {Config.GLOBAL_FEATURE_DIM}, got {data.global_x.shape[1]}"
        )

    # Test DataLoaders (this will trigger CrystalDataset processing and caching)
    # We set load_cached_data=False to force processing for this demo
    train_loader, val_loader, test_loader, target_scaler = get_dataloaders(
        load_cached_data=False
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Verify batch structure
    batch = next(iter(train_loader))
    print(f"Sample Batch: {batch}")
    if batch.batch is None:
        raise AssertionError("Batch vector missing in DataLoader output.")
    if batch.y.shape[1] != 2:
        raise AssertionError("Target shape mismatch in batch.")

    # 4. Test Model Architecture
    print("\n[4] Testing Model Architecture...")
    # Test Gaussian Smearing
    rbf = GaussianSmearing(start=0.0, stop=5.0, n_gaussians=10)
    dist = torch.tensor([1.0, 2.5, 4.0])
    rbf_out = rbf(dist)
    if rbf_out.shape != (3, 10):
        raise AssertionError(
            f"GaussianSmearing output shape mismatch. Expected (3, 10), got {rbf_out.shape}"
        )

    # Test Full Model
    model = HASCNet()
    # Forward pass with the sample batch
    output = model(batch)
    print(f"Model Output Shape: {output.shape}")

    expected_shape = (batch.num_graphs, 2)
    if output.shape != expected_shape:
        raise AssertionError(
            f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"
        )

    # 5. Test Training Loop
    print("\n[5] Running Training Loop (Fast)...")
    # run_training uses get_dataloaders internally.
    # Since we already generated cache in step 3, we can set load_cached_data=True
    run_training(load_cached_data=True)

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.WORKING_DIR, "checkpoints", "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise AssertionError("Checkpoint file was not created.")
    print("Training finished and checkpoint verified.")

    # 6. Test Submission Generation
    print("\n[6] Generating Submission...")
    generate_submission(load_cached_data=True)

    submission_path = "./submission/submission.csv"
    if not os.path.exists(submission_path):
        raise AssertionError("Submission file was not created.")

    sub_df = pd.read_csv(submission_path)
    print(f"Submission rows: {len(sub_df)}")
    print(f"Submission columns: {sub_df.columns.tolist()}")

    expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
    if not all(col in sub_df.columns for col in expected_cols):
        raise AssertionError(f"Submission columns mismatch. Expected {expected_cols}")

    print("\n--- Demonstration Complete ---")


if __name__ == "__main__":
    run_demo()
