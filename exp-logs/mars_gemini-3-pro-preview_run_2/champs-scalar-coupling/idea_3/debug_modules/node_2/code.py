import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Import library modules
# We need to import specific modules to patch their internal constants for the demo
import library.config as config
import library.data as data_module
import library.engine as engine_module
import library.model as model_module
import library.geometry as geometry_module
import library.utils as utils_module

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Logic Verification and Demo Script ===")

    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Patching
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid execution...")

    # Set seeds for reproducibility
    utils_module.set_seed(42)

    # Define demo-specific constants
    DEMO_CACHE_DIR = os.path.join(config.WORKING_DIR, "demo_run")
    if os.path.exists(DEMO_CACHE_DIR):
        shutil.rmtree(DEMO_CACHE_DIR)
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)

    # Patch library.data constants to force debug mode and small sample size
    data_module.DEBUG = True
    data_module.DEBUG_SAMPLE_SIZE = 50  # Process only 50 molecules
    data_module.CACHE_DIR = DEMO_CACHE_DIR  # Use separate cache to avoid conflicts

    # Patch library.engine constants for a quick training loop
    engine_module.MAX_EPOCHS = 1
    engine_module.BATCH_SIZE = 4
    engine_module.NUM_WORKERS = 0  # Use main process for simplicity
    engine_module.IDEA_NAME = "demo_run"
    engine_module.DEBUG = True

    # Patch config module (referenced by some classes directly)
    config.DEBUG = True
    config.DEBUG_SAMPLE_SIZE = 50

    print("Configuration patched: DEBUG=True, Sample Size=50, Epochs=1")

    # -------------------------------------------------------------------------
    # 2. Geometry Module Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Geometry Modules...")

    # Test Envelope
    envelope = geometry_module.Envelope(exponent=5)
    dist_ratios = torch.tensor([0.0, 0.5, 1.0, 1.5])
    env_vals = envelope(dist_ratios)

    # Assertions: 0.0 -> 1.0, 1.0 -> 0.0, >1.0 -> 0.0
    assert torch.isclose(env_vals[0], torch.tensor(1.0)), "Envelope at 0 should be 1"
    assert torch.isclose(
        env_vals[2], torch.tensor(0.0), atol=1e-6
    ), "Envelope at 1 should be 0"
    assert torch.all(env_vals[3:] == 0), "Envelope > 1 should be 0"
    print("  -> Envelope function verified.")

    # Test Radial Basis Functions
    num_rbf = 16
    rbf_fn = geometry_module.RadialBasisFunctions(num_rbf=num_rbf, cutoff=5.0)
    dists = torch.tensor([1.0, 2.5, 4.0])
    rbf_out = rbf_fn(dists)

    assert rbf_out.shape == (3, num_rbf), f"RBF output shape mismatch: {rbf_out.shape}"
    print("  -> RadialBasisFunctions verified.")

    # Test Spherical Basis Functions
    num_spherical = 3
    num_radial = 4
    sbf_fn = geometry_module.SphericalBasisFunctions(
        num_spherical, num_radial, cutoff=5.0
    )

    # Synthetic inputs: 5 triplets
    t_dists = torch.tensor([1.5, 2.0, 2.5, 3.0, 3.5])  # Distances
    t_angles = torch.tensor([0.5, 1.0, 1.5, 2.0, 2.5])  # Angles in radians
    # Index kj is needed for internal gathering, but for the SBF forward signature in this lib,
    # it takes (dist, angle, idx_kj).
    # Wait, looking at library.geometry.SphericalBasisFunctions.forward:
    # def forward(self, dist: torch.Tensor, angle: torch.Tensor, idx_kj: torch.Tensor)
    # It uses dist[idx_kj]. So we need edge distances and indices pointing to them.

    edge_dists = torch.tensor(
        [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    )  # Pool of edge distances
    idx_kj = torch.tensor([1, 2, 3, 4, 5])  # Indices into edge_dists for the 5 triplets

    sbf_out = sbf_fn(edge_dists, t_angles, idx_kj)
    expected_dim = num_spherical * num_radial

    assert sbf_out.shape == (
        5,
        expected_dim,
    ), f"SBF output shape mismatch: {sbf_out.shape}"
    assert not torch.isnan(sbf_out).any(), "SBF output contains NaNs"
    print("  -> SphericalBasisFunctions verified.")

    # -------------------------------------------------------------------------
    # 3. Data Loading and Processing Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Data Pipeline (MolecularGraphDataset)...")

    # Initialize Dataset (this will trigger processing of 50 molecules)
    # We use 'train' split which uses TRAIN_METADATA_PATH
    dataset = data_module.MolecularGraphDataset(
        metadata_path=config.TRAIN_METADATA_PATH,
        split_name="train_debug",  # Unique name to avoid loading main cache
        load_cached_data=False,
    )

    print(f"  -> Dataset initialized with {len(dataset)} graphs.")
    assert len(dataset) > 0, "Dataset is empty."

    # Check single item
    sample = dataset[0]
    required_keys = [
        "x",
        "pos",
        "edge_index",
        "edge_attr",
        "triplet_index",
        "triplet_attr",
        "y",
    ]
    for key in required_keys:
        assert key in sample, f"Sample missing key: {key}"

    print(
        f"  -> Sample 0: {sample['x'].shape[0]} atoms, {sample['edge_index'].shape[1]} edges."
    )

    # Check Collation
    batch_size = 4
    batch_list = [dataset[i] for i in range(batch_size)]
    batch = data_module.collate_graphs(batch_list)

    assert "batch" in batch, "Collated batch missing 'batch' index."
    assert batch["x"].shape[0] == sum(
        d["x"].shape[0] for d in batch_list
    ), "Batch node count mismatch."
    assert batch["y"].shape[0] == sum(
        d["y"].shape[0] for d in batch_list
    ), "Batch target count mismatch."
    print("  -> Collation verified.")

    # -------------------------------------------------------------------------
    # 4. Model Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying DMPNN Model...")

    model = model_module.DMPNN()
    model.to(config.DEVICE)

    # Move batch to device
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(config.DEVICE)

    # Forward pass
    model.eval()
    with torch.no_grad():
        preds = model(batch)

    # Output should be (NumTargets, 1)
    num_targets = batch["y"].shape[0]
    assert preds.shape == (
        num_targets,
        1,
    ), f"Model output shape mismatch. Expected ({num_targets}, 1), got {preds.shape}"
    print("  -> Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 5. Full Training Loop Verification (Trainer)
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Trainer (Fit Scaler, Train, Validate)...")

    trainer = engine_module.Trainer()

    # 1. Fit Scaler (Uses the full metadata file, but fast enough)
    trainer.fit_scaler()
    assert trainer.scaler.fitted, "Scaler failed to fit."

    # 2. Run Training
    # This calls get_dataloader, which uses our patched data_module.DEBUG settings
    # It will run for 1 epoch on the small subset
    print("  -> Starting short training run...")
    trainer.run()

    # Check if artifacts were created
    best_model_path = os.path.join(DEMO_CACHE_DIR, "best_model.pt")
    assert os.path.exists(best_model_path), "Best model checkpoint was not created."
    print("  -> Training loop completed and model saved.")

    # 3. Prediction (Optional check)
    # We won't run full prediction as it loads test set, but we verified the components.

    print("\n=== All verifications passed successfully! ===")


if __name__ == "__main__":
    run_demo()
