import os
import shutil
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import TrainConfig, ModelConfig
from library.features import RadialBasisFunctions, SphericalBasisFunctions
from library.data import MoleculeDataset, collate_graphs
from library.model import HybridModel
from library.engine import Engine, set_seed


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    print("==== Step 1: Configuration Setup ====")

    # Set seed for reproducibility
    set_seed(42)

    # Define a working directory for this demo to avoid overwriting main results
    demo_working_dir = "./working/demo_run"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Create a lightweight TrainConfig for speed
    train_config = TrainConfig(
        working_dir=demo_working_dir,
        model_path=os.path.join(demo_working_dir, "best_model.pt"),
        submission_path=os.path.join(demo_working_dir, "submission.csv"),
        batch_size=4,  # Small batch size
        epochs=1,  # Single epoch for demonstration
        learning_rate=1e-3,
        num_workers=0,  # 0 workers for simple debugging/demo
        debug=True,  # Enable debug mode to sample subset of data
        debug_samples=100,  # Only use 100 samples
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Create a lightweight ModelConfig for speed
    model_config = ModelConfig(
        hidden_dim=32,  # Reduced dimension
        num_mp_layers=2,  # Fewer layers
        num_transformer_layers=1,
        num_heads=4,
        num_rbf=16,
        num_sbf=8,
        cutoff=5.0,
    )

    print(f"Running on device: {train_config.device}")
    print("Configuration initialized successfully.")

    # -------------------------------------------------------------------------
    # 2. Feature Engineering Verification
    # -------------------------------------------------------------------------
    print("\n==== Step 2: Feature Engineering Verification ====")

    # Test Radial Basis Functions
    rbf = RadialBasisFunctions(
        num_radial=model_config.num_rbf, cutoff=model_config.cutoff
    )
    dummy_dist = torch.tensor([1.0, 2.5, 4.0])
    rbf_out = rbf(dummy_dist)

    assert rbf_out.shape == (
        3,
        model_config.num_rbf,
    ), f"RBF output shape mismatch. Expected (3, {model_config.num_rbf}), got {rbf_out.shape}"
    print(f"RBF Check Passed. Output shape: {rbf_out.shape}")

    # Test Spherical Basis Functions
    sbf = SphericalBasisFunctions(
        num_radial=model_config.num_rbf,
        num_spherical=model_config.num_sbf,
        cutoff=model_config.cutoff,
    )
    dummy_theta = torch.tensor([0.5, 1.5, 2.0])  # Radians
    sbf_out = sbf(dummy_dist, dummy_theta)

    expected_sbf_dim = model_config.num_rbf * model_config.num_sbf
    assert sbf_out.shape == (
        3,
        expected_sbf_dim,
    ), f"SBF output shape mismatch. Expected (3, {expected_sbf_dim}), got {sbf_out.shape}"
    print(f"SBF Check Passed. Output shape: {sbf_out.shape}")

    # -------------------------------------------------------------------------
    # 3. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n==== Step 3: Data Pipeline Verification ====")

    # Initialize Dataset (Debug mode loads a small subset)
    print("Initializing MoleculeDataset (Train)...")
    train_dataset = MoleculeDataset(split="train", config=train_config)

    assert len(train_dataset) > 0, "Dataset is empty."
    print(f"Dataset loaded with {len(train_dataset)} molecules (debug subset).")

    # Check single item structure
    sample = train_dataset[0]
    required_keys = [
        "z",
        "pos",
        "edge_index",
        "edge_rbf",
        "triplet_indices",
        "triplet_sbf",
        "coupling_target",
        "coupling_type",
    ]
    for key in required_keys:
        assert key in sample, f"Missing key {key} in dataset sample."

    print("Single sample keys verified.")
    print(f"Sample Atom Types (z): {sample['z'].shape}")
    print(f"Sample Coupling Targets: {sample['coupling_target'].shape}")

    # Test Collate Function
    print("Testing DataLoader and Collate...")
    loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        collate_fn=collate_graphs,
        shuffle=False,
    )
    batch = next(iter(loader))

    assert "batch" in batch, "Batch index missing in collated data."
    assert (
        batch["batch_size"] == train_config.batch_size
    ), f"Batch size mismatch. Expected {train_config.batch_size}, got {batch['batch_size']}"

    # Verify concatenated dimensions
    total_nodes = batch["z"].shape[0]
    assert (
        batch["batch"].shape[0] == total_nodes
    ), "Batch index shape mismatch with nodes."
    print(f"Batch collation successful. Total nodes in batch: {total_nodes}")

    # -------------------------------------------------------------------------
    # 4. Model Verification
    # -------------------------------------------------------------------------
    print("\n==== Step 4: Model Verification ====")

    model = HybridModel(model_config)
    model.to(train_config.device)

    # Move batch to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(train_config.device)

    # Forward pass
    print("Running Model Forward Pass...")
    with torch.no_grad():
        preds = model(batch)

    num_couplings = batch["coupling_target"].shape[0]
    assert preds.shape == (
        num_couplings,
    ), f"Prediction shape mismatch. Expected ({num_couplings},), got {preds.shape}"

    print(f"Forward pass successful. Predictions shape: {preds.shape}")

    # -------------------------------------------------------------------------
    # 5. Engine & Training Loop Verification
    # -------------------------------------------------------------------------
    print("\n==== Step 5: Engine & Training Loop Verification ====")

    # Initialize Engine
    engine = Engine(train_config, model_config)

    # Run Training (1 Epoch)
    print("Starting Engine.train()...")
    engine.train()

    # Check if model checkpoint was saved
    assert os.path.exists(train_config.model_path), "Model checkpoint was not saved."
    print(f"Training complete. Checkpoint found at {train_config.model_path}")

    # Run Submission Generation
    print("Starting Engine.generate_submission()...")
    # Note: We use the same debug config, so it will load a subset of test data if debug is handled in test split
    # The provided MoleculeDataset handles debug logic for any split.
    engine.generate_submission()

    assert os.path.exists(
        train_config.submission_path
    ), "Submission file was not generated."

    # Verify Submission Content
    df_sub = pd.read_csv(train_config.submission_path)
    print(f"Submission generated with shape: {df_sub.shape}")
    assert (
        "id" in df_sub.columns and "scalar_coupling_constant" in df_sub.columns
    ), "Submission file missing required columns."

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
