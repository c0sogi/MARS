import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import set_seed, TargetScaler
from library.features import RBFExpansion, SBFExpansion
from library.data import MolecularGraphDataset, collate_graphs
from library.model import HGANet
from library.train import run_training


def test_feature_expansions():
    """Verifies the geometric feature expansion layers."""
    print("\n=== Testing Feature Expansions ===")

    # 1. RBF Expansion
    num_rbf = 16
    rbf_layer = RBFExpansion(num_rbf=num_rbf, cutoff=5.0)

    # Create dummy distances (batch of 10 edges)
    dist = torch.linspace(0.5, 4.5, 10)
    rbf_out = rbf_layer(dist)

    assert rbf_out.shape == (
        10,
        num_rbf,
    ), f"RBF output shape mismatch. Expected (10, {num_rbf}), got {rbf_out.shape}"
    print(f"RBF Expansion verified. Output shape: {rbf_out.shape}")

    # 2. SBF Expansion
    num_sbf = 7
    sbf_layer = SBFExpansion(num_sbf=num_sbf, num_rbf=num_rbf, cutoff=5.0)

    # Create dummy triplet data
    # 5 triplets
    dist_edges = torch.rand(20)  # Pool of edge distances
    angles = torch.rand(5) * 3.14  # 5 triplet angles
    idx_kj = torch.tensor([0, 1, 2, 3, 4])  # Indices pointing to edges in dist_edges

    sbf_out = sbf_layer(dist_edges, angles, idx_kj)

    expected_dim = num_sbf * num_rbf
    assert sbf_out.shape == (
        5,
        expected_dim,
    ), f"SBF output shape mismatch. Expected (5, {expected_dim}), got {sbf_out.shape}"
    print(f"SBF Expansion verified. Output shape: {sbf_out.shape}")


def test_target_scaler():
    """Verifies the TargetScaler logic."""
    print("\n=== Testing Target Scaler ===")

    # Create dummy dataframe
    df = pd.DataFrame(
        {
            "type": ["1JHC", "1JHC", "2JHH", "2JHH"],
            "scalar_coupling_constant": [10.0, 20.0, 1.0, 3.0],
        }
    )

    scaler = TargetScaler()
    scaler.fit(df)

    # Check stats
    stats_1jhc = scaler.stats["1JHC"]
    assert (
        stats_1jhc["mean"] == 15.0
    ), f"Mean calculation failed. Got {stats_1jhc['mean']}"
    assert (
        stats_1jhc["std"] == 7.0710678118654755
    ), f"Std calculation failed. Got {stats_1jhc['std']}"

    # Transform
    transformed = scaler.transform(df)
    expected_val_0 = (10.0 - 15.0) / 7.0710678118654755
    assert np.isclose(transformed[0], expected_val_0), "Transform calculation failed."

    # Inverse Transform
    types = df["type"].values
    inversed = scaler.inverse_transform(transformed, types)
    assert np.allclose(
        inversed, df["scalar_coupling_constant"].values
    ), "Inverse transform failed."

    print("TargetScaler verified successfully.")


def test_dataset_and_model():
    """Verifies dataset loading, batching, and model forward pass."""
    print("\n=== Testing Dataset and Model ===")

    # Use a custom split name to avoid interfering with main training cache
    # and force processing of a small subset
    split_name = "demo_test"

    # Initialize Dataset
    # Config.DEBUG is already True, so it will slice the metadata
    dataset = MolecularGraphDataset(
        metadata_path=Config.TRAIN_CSV,
        split_name=split_name,
        load_cached_data=False,  # Force processing to test logic
    )

    print(f"Dataset initialized with {len(dataset)} molecules (DEBUG mode).")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check input data or debug sampling size.")

    # Test __getitem__
    sample = dataset[0]
    required_keys = [
        "atom_z",
        "pos",
        "edge_index",
        "edge_rbf",
        "triplet_indices",
        "triplet_sbf",
    ]
    for k in required_keys:
        assert k in sample, f"Missing key {k} in dataset sample."

    print(f"Sample molecule: {sample['mol_name']}")
    print(f"  Atoms: {sample['atom_z'].shape[0]}")
    print(f"  Edges: {sample['edge_index'].shape[1]}")
    print(f"  Triplets: {sample['triplet_indices'].shape[1]}")

    # Test DataLoader and Collation
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_graphs)

    batch = next(iter(loader))
    print(f"Batch loaded. Batch size: {batch['batch_size']}")

    # Verify batch construction
    assert "batch_atom" in batch
    assert batch["batch_atom"].max() < 4

    # Test Model Forward Pass
    model = HGANet(Config).to(Config.DEVICE)

    # Move batch to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(Config.DEVICE)

    with torch.no_grad():
        preds = model(batch)

    num_couplings = batch["coupling_value"].shape[0]
    assert preds.shape == (
        num_couplings,
    ), f"Prediction shape mismatch. Expected ({num_couplings},), got {preds.shape}"

    print(f"Model forward pass successful. Predictions shape: {preds.shape}")


def run_pipeline_demo():
    """Runs the full training pipeline using the library function."""
    print("\n=== Running Full Training Pipeline Demo ===")

    # Ensure clean slate for the 'best_model.pt'
    if os.path.exists(Config.MODEL_SAVE_PATH):
        os.remove(Config.MODEL_SAVE_PATH)

    # Run training
    # This will use the Config overrides set in main
    run_training(
        debug=True,
        max_epochs=1,
        batch_size=4,
        learning_rate=1e-3,
        num_workers=0,  # Avoid multiprocessing overhead for small demo
    )

    # Verify outputs
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError("Model file was not saved.")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Pipeline completed. Submission generated with {len(df_sub)} rows.")
    print("Head of submission:")
    print(df_sub.head())


if __name__ == "__main__":
    # 1. Setup Configuration for Speed
    set_seed(42)

    # Override Config for a fast demo run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Only process 20 molecules
    Config.MAX_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Run Verifications
    try:
        test_feature_expansions()
        test_target_scaler()
        test_dataset_and_model()
        run_pipeline_demo()
        print("\nAll demonstrations completed successfully.")
    except Exception as e:
        print(f"\nAn error occurred during demonstration: {e}")
        # Clean up if needed, though usually better to leave for inspection
        raise e
