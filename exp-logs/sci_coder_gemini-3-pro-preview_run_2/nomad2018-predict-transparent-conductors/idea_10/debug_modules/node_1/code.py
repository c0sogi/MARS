import os
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.data_utils import (
    read_geometry,
    get_global_features,
    build_pbc_graph,
    GaussianRBF,
    StandardScaler,
)
from library.dataset import CrystalGraphDataset, collate_graphs
from library.model import DualStreamNet
from library.engine import run_training, generate_submission


def main():
    print("Initializing Demo Script...")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    # We modify the Config class attributes to use a separate directory for this demo
    # to avoid overwriting or interfering with the main 'idea_10' run.
    print("\n[1] Setting up Demo Configuration...")
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce parameters for speed
    Config.BATCH_SIZE = 4
    Config.NUM_EPOCHS = 2
    Config.PATIENCE = 2

    # Create directories
    Config.setup_directories()
    print(f"Working directory set to: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Utilities Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Utilities...")

    # Load train metadata to get a sample file path
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    sample_row = train_meta.iloc[0]
    sample_file_path = sample_row["file_path"]
    print(f"Testing with sample file: {sample_file_path}")

    # Test read_geometry
    atoms = read_geometry(sample_file_path)
    print(f"Successfully read geometry. Number of atoms: {len(atoms)}")
    assert len(atoms) > 0, "Atom list should not be empty"

    # Test get_global_features
    global_feats = get_global_features(atoms)
    print(f"Global features shape: {global_feats.shape}")
    # Expecting 10 features: 6 lattice params + 4 composition fractions
    assert global_feats.shape == (
        10,
    ), f"Expected 10 global features, got {global_feats.shape[0]}"

    # Test build_pbc_graph
    z, src, dst, dists = build_pbc_graph(atoms, cutoff=5.0, max_neighbors=12)
    print(f"Graph constructed. Nodes: {len(z)}, Edges: {len(src)}")
    assert len(z) == len(atoms), "Number of nodes should match number of atoms"
    assert len(src) == len(dst) == len(dists), "Edge arrays must have same length"

    # Test GaussianRBF
    rbf = GaussianRBF(start=0.0, stop=5.0, n_rbf=20)
    dists_tensor = torch.tensor(dists, dtype=torch.float32)
    rbf_out = rbf(dists_tensor)
    print(f"Gaussian RBF output shape: {rbf_out.shape}")
    assert rbf_out.shape == (len(dists), 20), "RBF output shape mismatch"

    # Test StandardScaler
    scaler = StandardScaler()
    dummy_data = torch.randn(10, 5)
    scaler.fit(dummy_data)
    transformed = scaler.transform(dummy_data)
    inverse = scaler.inverse_transform(transformed)
    print("StandardScaler fit/transform/inverse check passed.")
    assert torch.allclose(dummy_data, inverse, atol=1e-5), "Inverse transform failed"

    # -------------------------------------------------------------------------
    # 3. Dataset and Dataloader Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Dataset and Dataloader...")

    # Initialize Dataset with a small sample limit
    # We use a unique cache prefix for the demo to avoid loading full dataset cache
    sample_limit = 20
    dataset = CrystalGraphDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        cache_prefix="train_debug",
        load_cached_data=False,  # Force processing for demo
        fit_scalers=True,
        sample_limit=sample_limit,
    )

    print(f"Dataset length: {len(dataset)}")
    assert (
        len(dataset) == sample_limit
    ), f"Expected {sample_limit} samples, got {len(dataset)}"

    # Check a single item
    data_item = dataset[0]
    print(f"Single Data Item keys: {data_item.keys()}")
    assert data_item.x.dim() == 1, "Node features should be 1D (atomic numbers)"
    assert data_item.edge_index.shape[0] == 2, "Edge index should be (2, E)"
    assert data_item.global_x.shape == (1, 10), "Global features should be (1, 10)"
    assert data_item.y.shape == (1, 2), "Target should be (1, 2)"

    # Test DataLoader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_graphs)
    batch = next(iter(loader))
    print(f"Batch size: {batch.num_graphs}")
    assert batch.num_graphs == 4, "Batch size mismatch"
    # Check batching of global features and targets
    assert batch.global_x.shape == (
        4,
        10,
    ), f"Batch global_x shape mismatch: {batch.global_x.shape}"
    assert batch.y.shape == (4, 2), f"Batch y shape mismatch: {batch.y.shape}"

    # -------------------------------------------------------------------------
    # 4. Model Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualStreamNet().to(device)
    batch = batch.to(device)

    # Forward pass
    output = model(batch)
    print(f"Model output shape: {output.shape}")
    assert output.shape == (4, 2), f"Expected output shape (4, 2), got {output.shape}"

    # Check for NaN
    if torch.isnan(output).any():
        raise ValueError("Model output contains NaNs")
    print("Forward pass successful.")

    # -------------------------------------------------------------------------
    # 5. Engine / Training Loop Verification
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Training Engine...")

    # We will run the full training routine on a small subset
    # This tests train_one_epoch, evaluate, EarlyStopping, and saving checkpoints
    print("Running short training session...")

    # Note: run_training handles dataset creation internally.
    # We rely on it using the modified Config paths.
    g_scaler, t_scaler = run_training(sample_size=30, epochs=2)

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created"
    print("Training loop completed successfully.")

    # -------------------------------------------------------------------------
    # 6. Submission Generation Verification
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Submission Generation...")

    # Generate submission for a small subset of test data
    # We pass the scalers returned from training
    generate_submission(global_scaler=g_scaler, target_scaler=t_scaler, sample_size=10)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Verify submission format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")
    expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"
    assert len(sub_df) == 10, "Submission should have 10 rows (based on sample_size)"

    # Check for valid values (non-negative as enforced in generate_submission)
    if (sub_df["formation_energy_ev_natom"] < 0).any() or (
        sub_df["bandgap_energy_ev"] < 0
    ).any():
        print(
            "Warning: Negative values found in submission (clamped in generation but checking CSV content)."
        )

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    # Ensure we are in the correct directory context
    # (The script assumes it is run from the root where 'library' exists)
    if not os.path.exists("library"):
        raise FileNotFoundError(
            "Could not find 'library' directory. Please run from the root of the project."
        )

    main()
