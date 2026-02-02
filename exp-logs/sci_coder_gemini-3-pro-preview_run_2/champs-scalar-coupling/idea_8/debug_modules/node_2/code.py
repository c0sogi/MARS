import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import TargetScaler, LogMAE
from library.geometry import Geometry
from library.dataset import CouplingDataset, collate_graphs
from library.model import HGANet

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Demo
    # -------------------------------------------------------------------------
    print("Setting up configuration for demo run...")

    # Override Config parameters for a fast, isolated execution
    Config.EXPERIMENT_NAME = "demo_verification_run"
    Config.WORKING_DIR = os.path.join("./working", Config.EXPERIMENT_NAME)

    # Ensure working directory exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update cache paths to use the new working directory
    Config.CACHE_TRAIN_PATH = os.path.join(Config.WORKING_DIR, "cached_train.npz")
    Config.CACHE_STATS_PATH = os.path.join(Config.WORKING_DIR, "target_stats.npz")

    # Enable Debug mode to process only a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Process only 20 molecules
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1

    # Set reproducible seed
    Config.set_seed(42)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Verify TargetScaler (Preprocessing)
    # -------------------------------------------------------------------------
    print("\n[1/5] Verifying TargetScaler...")

    # Load a small chunk of metadata for testing
    df_meta = pd.read_csv(Config.TRAIN_METADATA_PATH).head(100)

    scaler = TargetScaler()
    # Fit scaler on the small metadata subset
    scaler.fit(df_meta, load_cache=False)

    # Test Transform
    scaled_values = scaler.transform(df_meta)

    # Test Inverse Transform
    reconstructed_values = scaler.inverse_transform(
        scaled_values, df_meta["type"].values
    )

    # Verification: Original values should match reconstructed values
    original_values = df_meta["scalar_coupling_constant"].values
    max_error = np.abs(original_values - reconstructed_values).max()

    assert max_error < 1e-5, f"Scaler reconstruction error too high: {max_error}"
    print("TargetScaler logic verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Verify Dataset and Graph Construction
    # -------------------------------------------------------------------------
    print("\n[2/5] Verifying CouplingDataset and Graph Construction...")

    # Initialize Dataset (this will trigger processing of the 20 debug molecules)
    dataset = CouplingDataset(
        metadata_path=Config.TRAIN_METADATA_PATH,
        cache_path=Config.CACHE_TRAIN_PATH,
        load_cached_data=False,  # Force re-computation
        split="train",
    )

    # Verify dataset length
    assert (
        len(dataset) == Config.DEBUG_SUBSET_SIZE
    ), f"Expected {Config.DEBUG_SUBSET_SIZE} molecules, got {len(dataset)}"

    # Verify single item structure
    sample = dataset[0]
    required_keys = [
        "atom_types",
        "coords",
        "edge_index",
        "edge_rbf",
        "triplet_indices",
        "triplet_sbf",
        "coupling_atom_0",
        "coupling_atom_1",
        "coupling_types",
        "coupling_targets",
    ]
    for key in required_keys:
        assert key in sample, f"Missing key in dataset sample: {key}"

    # Verify shapes for the first sample
    num_atoms = len(sample["atom_types"])
    num_edges = sample["edge_index"].shape[1]

    assert sample["edge_index"].shape == (2, num_edges)
    assert sample["coords"].shape == (num_atoms, 3)

    print(f"Processed {len(dataset)} molecules successfully.")

    # -------------------------------------------------------------------------
    # 4. Verify DataLoader and Batching
    # -------------------------------------------------------------------------
    print("\n[3/5] Verifying DataLoader and Batching...")

    loader = DataLoader(
        dataset, batch_size=Config.BATCH_SIZE, collate_fn=collate_graphs, shuffle=True
    )

    # Fetch one batch
    batch = next(iter(loader))

    # Verify batch structure
    assert "batch" in batch, "Batch indices missing from collated data"
    assert batch["edge_index"].shape[0] == 2

    # Move batch to device
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(Config.DEVICE)

    print(
        f"Batch loaded. Atoms: {batch['atom_types'].shape[0]}, Edges: {batch['edge_index'].shape[1]}"
    )

    # -------------------------------------------------------------------------
    # 5. Verify Model and Training Step
    # -------------------------------------------------------------------------
    print("\n[4/5] Verifying HGANet Model and Training Step...")

    # Initialize Model
    model = HGANet().to(Config.DEVICE)
    model.train()

    # Setup Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    # --- Forward Pass ---
    preds = model(batch)

    # Verify output shape: (Num_Couplings, 1)
    num_couplings = batch["coupling_targets"].shape[0]
    assert preds.shape == (
        num_couplings,
        1,
    ), f"Output shape mismatch. Expected ({num_couplings}, 1), got {preds.shape}"

    # --- Loss Calculation ---
    # We need to scale the targets to match the model's expected output range (standardized)
    # 1. Map integer types back to strings
    inv_type_map = {v: k for k, v in Geometry.COUPLING_TYPE_MAP.items()}
    batch_types_int = batch["coupling_types"].cpu().numpy()
    batch_types_str = [inv_type_map[t] for t in batch_types_int]

    # 2. Get stats for these types
    means = torch.tensor(
        [scaler.means[t] for t in batch_types_str], device=Config.DEVICE
    ).unsqueeze(1)
    stds = torch.tensor(
        [scaler.stds[t] for t in batch_types_str], device=Config.DEVICE
    ).unsqueeze(1)

    # 3. Scale targets: (y - mean) / std
    targets_raw = batch["coupling_targets"].unsqueeze(1)
    targets_scaled = (targets_raw - means) / (stds + 1e-8)

    # Calculate Loss
    loss = loss_fn(preds, targets_scaled)

    # --- Backward Pass ---
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Forward and Backward pass successful. Loss: {loss.item():.4f}")

    # -------------------------------------------------------------------------
    # 6. Verify Metric
    # -------------------------------------------------------------------------
    print("\n[5/5] Verifying LogMAE Metric...")

    # Create dummy data
    y_true_dummy = np.array([10.0, 5.0, 1.0])
    y_pred_dummy = np.array([10.1, 4.9, 1.2])  # Small errors
    types_dummy = ["1JHC", "2JHH", "1JHC"]

    # Calculate score
    score, per_type_score = LogMAE.score(y_true_dummy, y_pred_dummy, types_dummy)

    # MAE for 1JHC: (|10-10.1| + |1-1.2|) / 2 = (0.1 + 0.2)/2 = 0.15
    # MAE for 2JHH: |5-4.9| = 0.1
    # Log MAE: mean(log(0.15), log(0.1))
    expected_score = np.mean([np.log(0.15), np.log(0.1)])

    assert np.isclose(score, expected_score, atol=1e-5), "Metric calculation mismatch"
    print(f"LogMAE Score verified: {score:.4f}")

    print("\nAll demonstrations and verifications passed successfully!")
