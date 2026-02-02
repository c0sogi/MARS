import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import components from the provided library
from library.config import Config
from library.data import CrystalDataset, collate_batch
from library.model import IALCDS
from library.train import train_model, set_seed
from library.utils import inverse_log_transform, compute_rmsle, log_transform


def clean_cache():
    """Removes cached .npz files to ensure data is re-processed with debug limits."""
    cache_files = [
        Config.CACHE_TRAIN_DATA,
        Config.CACHE_VAL_DATA,
        Config.CACHE_TEST_DATA,
    ]
    for cf in cache_files:
        if os.path.exists(cf):
            try:
                os.remove(cf)
                print(f"Removed cache file: {cf}")
            except OSError as e:
                print(f"Error removing {cf}: {e}")


def run_demo():
    print("=== Starting Library Demo ===\n")

    # 1. Configure for Speed and Debugging
    # We modify the Config class attributes directly to control the execution
    print("[1] Configuring parameters for fast execution...")
    Config.NUM_EPOCHS = 2
    Config.Batch_SIZE = 4
    Config.DEBUG_DATA_LIMIT = 20  # Limit to 20 samples for speed

    # Ensure working directories exist
    Config.setup()

    # Clean existing cache to force re-processing with the small data limit
    clean_cache()

    # Set reproducibility
    set_seed(Config.SEED)
    print("Configuration complete.\n")

    # 2. Data Loading and Processing
    print("[2] Testing Data Loading...")
    # Initialize datasets (this will trigger processing of the first 20 samples)
    train_dataset = CrystalDataset(
        mode="train", load_cached_data=False, limit=Config.DEBUG_DATA_LIMIT
    )
    val_dataset = CrystalDataset(
        mode="val", load_cached_data=False, limit=Config.DEBUG_DATA_LIMIT
    )

    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Val dataset length: {len(val_dataset)}")

    # Validate dataset size
    assert (
        len(train_dataset) == Config.DEBUG_DATA_LIMIT
    ), f"Expected {Config.DEBUG_DATA_LIMIT} train samples, got {len(train_dataset)}"

    # Validate single item structure
    sample = train_dataset[0]
    required_keys = [
        "atom_types",
        "dist_matrix",
        "lattice_features",
        "mask",
        "target",
        "id",
    ]
    for key in required_keys:
        assert key in sample, f"Missing key in dataset sample: {key}"

    # Check tensor shapes (MAX_ATOMS is 100 in library.data)
    MAX_ATOMS = 100
    assert sample["atom_types"].shape == (
        MAX_ATOMS,
    ), f"Incorrect atom_types shape: {sample['atom_types'].shape}"
    assert sample["dist_matrix"].shape == (
        MAX_ATOMS,
        MAX_ATOMS,
    ), f"Incorrect dist_matrix shape: {sample['dist_matrix'].shape}"
    assert sample["lattice_features"].shape == (
        6,
    ), f"Incorrect lattice_features shape: {sample['lattice_features'].shape}"
    assert sample["target"].shape == (
        2,
    ), f"Incorrect target shape: {sample['target'].shape}"

    print("Dataset sample structure verified.")

    # Test Batch Collation
    print("Testing batch collation...")
    batch_size = 4
    batch_list = [train_dataset[i] for i in range(batch_size)]
    batch = collate_batch(batch_list)

    assert batch["atom_types"].shape == (batch_size, MAX_ATOMS)
    assert batch["dist_matrix"].shape == (batch_size, MAX_ATOMS, MAX_ATOMS)
    assert batch["target"].shape == (batch_size, 2)
    print("Batch collation successful.\n")

    # 3. Model Instantiation and Forward Pass
    print("[3] Testing Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = IALCDS().to(device)

    # Move batch to device
    atom_types = batch["atom_types"].to(device)
    dist_matrix = batch["dist_matrix"].to(device)
    lattice_features = batch["lattice_features"].to(device)
    mask = batch["mask"].to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(atom_types, dist_matrix, lattice_features, mask)

    print(f"Model output shape: {outputs.shape}")
    assert outputs.shape == (batch_size, 2), "Model output shape mismatch"
    print("Model forward pass successful.\n")

    # 4. Training Loop
    print("[4] Testing Training Loop...")
    # train_model handles dataloaders and the loop internally
    # We pass the limit_data parameter to ensure it uses the small subset
    trained_model = train_model(
        num_epochs=Config.NUM_EPOCHS, limit_data=Config.DEBUG_DATA_LIMIT
    )

    # Verify checkpoint exists
    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), "Model checkpoint not found after training"
    print("Training loop executed successfully.\n")

    # 5. Utilities and Metrics
    print("[5] Testing Utilities...")

    # Test Log Transform and Inverse
    dummy_targets = np.array([[0.5, 2.0], [1.0, 3.5]], dtype=np.float32)
    transformed = log_transform(dummy_targets)
    reconstructed = inverse_log_transform(transformed)

    print(f"Original: {dummy_targets[0]}")
    print(f"Reconstructed: {reconstructed[0]}")

    assert np.allclose(
        dummy_targets, reconstructed, atol=1e-5
    ), "Log transform inversion failed"

    # Test RMSLE Calculation
    # RMSLE should be 0 if pred == true
    score_perfect = compute_rmsle(dummy_targets, dummy_targets)
    assert (
        score_perfect < 1e-6
    ), f"RMSLE for perfect prediction should be ~0, got {score_perfect}"

    # RMSLE with error
    # log(1+0.5) vs log(1+0.6) -> error
    dummy_preds = np.array([[0.6, 2.1], [0.9, 3.4]], dtype=np.float32)
    score_error = compute_rmsle(dummy_targets, dummy_preds)
    print(f"RMSLE with small error: {score_error:.6f}")
    assert score_error > 0, "RMSLE should be positive for different inputs"

    print("Utilities verified.\n")

    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
