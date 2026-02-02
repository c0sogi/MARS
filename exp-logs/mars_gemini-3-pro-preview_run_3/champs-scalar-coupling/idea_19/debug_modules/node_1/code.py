import os
import torch
import numpy as np
import pandas as pd
import shutil
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, Standardizer
from library.data import MoleculeDataset, collate_molecules
from library.model import MPDIN
from library.train import Trainer, predict_submission


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("--- 1. Initializing Configuration ---")

    # Create a specific working directory for this demo to avoid cache conflicts
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Initialize Config with overrides for speed
    config = Config(
        WORKING_DIR=demo_working_dir,
        # Enable debug mode to use only 200 molecules
        debug=True,
        debug_samples=200,
        # Minimal training parameters
        epochs=2,
        batch_size=16,
        num_workers=0,  # Avoid multiprocessing overhead for small demo
        # Model parameters (keep small for speed)
        node_dim=64,
        num_layers=2,
        num_rbf=32,
    )

    # Set seeds
    seed_everything(config.seed)
    print(f"Working Directory: {config.WORKING_DIR}")
    print(f"Debug Mode: {config.debug}")
    print(f"Device: {config.device}")

    # ==========================================
    # 2. Data Pipeline Verification
    # ==========================================
    print("\n--- 2. Verifying Data Pipeline ---")

    # Initialize Dataset (this triggers preprocessing and caching)
    train_ds = MoleculeDataset(config, split="train", load_cached_data=False)

    # Assertion: Check dataset size
    print(f"Dataset length: {len(train_ds)}")
    assert (
        len(train_ds) == config.debug_samples
    ), f"Expected {config.debug_samples} samples in debug mode, got {len(train_ds)}"

    # Assertion: Check single item structure
    sample = train_ds[0]
    required_keys = [
        "atom_types",
        "atom_coords",
        "coupling_atom_index_0",
        "coupling_atom_index_1",
        "coupling_type",
        "coupling_value",
    ]
    for key in required_keys:
        assert key in sample, f"Missing key {key} in dataset sample"
        assert torch.is_tensor(sample[key]), f"{key} is not a tensor"

    print("Sample keys and types verified.")

    # Test Collation
    batch_list = [train_ds[i] for i in range(4)]
    collated_batch = collate_molecules(batch_list)

    # Verify Batch Index
    total_atoms = sum(item["num_atoms"] for item in batch_list)
    assert (
        collated_batch["batch_index"].shape[0] == total_atoms
    ), "Batch index shape mismatch"
    assert (
        collated_batch["batch_index"].max() == 3
    ), "Batch index should go up to batch_size - 1"

    # Verify Coupling Index Shifting
    # The coupling indices in the batch should point to atoms in the concatenated arrays
    # We check if the max index used in couplings is within the range of total atoms
    max_coupling_idx = max(
        collated_batch["coupling_atom_index_0"].max(),
        collated_batch["coupling_atom_index_1"].max(),
    )
    assert (
        max_coupling_idx < total_atoms
    ), "Coupling indices out of bounds after collation"

    print("Collation logic verified.")

    # ==========================================
    # 3. Model Logic Verification
    # ==========================================
    print("\n--- 3. Verifying Model Forward Pass ---")

    model = MPDIN(config).to(config.device)

    # Move batch to device
    for k, v in collated_batch.items():
        if isinstance(v, torch.Tensor):
            collated_batch[k] = v.to(config.device)

    # Forward pass
    with torch.no_grad():
        output = model(collated_batch)

    # Verify Output Shape: [num_couplings, 1]
    num_couplings_in_batch = collated_batch["coupling_value"].shape[0]
    assert output.shape == (
        num_couplings_in_batch,
        1,
    ), f"Model output shape mismatch. Expected ({num_couplings_in_batch}, 1), got {output.shape}"

    print(f"Model forward pass successful. Output shape: {output.shape}")

    # ==========================================
    # 4. Standardizer Verification
    # ==========================================
    print("\n--- 4. Verifying Standardizer Logic ---")

    std = Standardizer(config)
    # It should have been fit during MoleculeDataset init
    std.load()

    # Create dummy data
    dummy_vals = torch.tensor([10.0, 20.0, 30.0], device=config.device)
    dummy_types = torch.tensor([0, 1, 0], device=config.device)  # Types 0 and 1

    # Transform
    transformed = std.transform(dummy_vals, dummy_types)

    # Inverse Transform
    reconstructed = std.inverse_transform(transformed, dummy_types)

    # Check reconstruction
    assert torch.allclose(
        dummy_vals, reconstructed, atol=1e-5
    ), "Standardizer reconstruction failed"

    print("Standardizer transform/inverse logic verified.")

    # ==========================================
    # 5. Training Loop Execution
    # ==========================================
    print("\n--- 5. Executing Training Loop ---")

    trainer = Trainer(config)
    trainer.fit()

    assert os.path.exists(
        config.MODEL_SAVE_PATH
    ), "Model file was not saved after training"

    print("Training loop completed successfully.")

    # ==========================================
    # 6. Prediction & Submission
    # ==========================================
    print("\n--- 6. Generating Submission ---")

    predict_submission(config)

    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not generated"

    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {df_sub.shape}")
    print(df_sub.head())

    # Basic checks on submission
    assert (
        "id" in df_sub.columns and "scalar_coupling_constant" in df_sub.columns
    ), "Submission missing required columns"
    assert not df_sub.isnull().values.any(), "Submission contains NaN values"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
