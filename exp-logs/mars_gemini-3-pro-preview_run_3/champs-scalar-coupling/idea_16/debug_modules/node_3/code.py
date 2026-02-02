import os
import torch
import numpy as np
import pandas as pd
import shutil
import time

# Import from the provided library
from library.config import Config
from library.data import get_dataloaders, MoleculeDataset, collate_molecules
from library.model import SDIN
from library.utils import Standardizer, Logger, load_checkpoint
from library.train import Trainer, predict_test


def run_demo():
    print("===========================================================")
    print("   Starting Demonstration of Molecular Prediction Pipeline")
    print("===========================================================")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Verification
    # ---------------------------------------------------------
    print("\n[1] Overriding Configuration for Fast Execution...")

    # Set up a specific working directory for this demo
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)

    # Modify Config class attributes directly to create a lightweight run
    Config.WORKING_DIR = DEMO_DIR
    Config.TRAIN_CACHE_DIR = os.path.join(DEMO_DIR, "train_cache")
    Config.VAL_CACHE_DIR = os.path.join(DEMO_DIR, "val_cache")
    Config.TEST_CACHE_DIR = os.path.join(DEMO_DIR, "test_cache")
    Config.STATS_PATH = os.path.join(DEMO_DIR, "stats.npy")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission", "submission.csv")

    # Enable Debug mode to sample a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Only use 100 molecules

    # Reduce Model Complexity
    Config.HIDDEN_DIM = 32
    Config.N_LAYERS = 2
    Config.N_RBF = 10

    # Training Loop Optimization
    Config.MAX_EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple debug run

    # Re-run setup to create the new directories
    Config.setup()

    print("    Configuration updated: DEBUG=True, EPOCHS=2, SAMPLE_SIZE=100")

    # ---------------------------------------------------------
    # 2. Data Loading and Verification
    # ---------------------------------------------------------
    print("\n[2] Initializing Data Loaders...")

    # Force processing from scratch to verify pipeline
    train_loader, val_loader, test_loader, standardizer = get_dataloaders(
        load_cached_data=False
    )

    print("    Data loaders created successfully.")

    # Verify Batch Structure
    print("    Verifying batch structure...")
    batch = next(iter(train_loader))

    required_keys = [
        "atom_types",
        "atom_coords",
        "batch_index",
        "coupling_pairs",
        "coupling_types",
        "coupling_values",
        "coupling_ids",
        "mol_names",
        "num_graphs",
    ]

    for key in required_keys:
        assert key in batch, f"Batch missing key: {key}"

    # Verify Shapes
    num_atoms = batch["atom_types"].size(0)
    num_couplings = batch["coupling_values"].size(0)

    assert batch["atom_coords"].shape == (num_atoms, 3), "Incorrect atom_coords shape"
    assert batch["batch_index"].shape == (num_atoms,), "Incorrect batch_index shape"
    assert batch["coupling_pairs"].shape == (
        num_couplings,
        2,
    ), "Incorrect coupling_pairs shape"
    assert batch["coupling_types"].shape == (
        num_couplings,
    ), "Incorrect coupling_types shape"

    print(
        f"    Batch verification passed. (Atoms: {num_atoms}, Couplings: {num_couplings})"
    )

    # ---------------------------------------------------------
    # 3. Standardizer Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying Standardizer Logic...")

    # Create dummy data
    dummy_vals = torch.tensor([10.0, 20.0, 30.0], device=Config.DEVICE)
    dummy_types = torch.tensor([0, 1, 0], device=Config.DEVICE)  # Types 0 and 1

    # Manually set stats for deterministic check
    standardizer.means = torch.zeros(Config.NUM_COUPLING_TYPES, device=Config.DEVICE)
    standardizer.stds = torch.ones(Config.NUM_COUPLING_TYPES, device=Config.DEVICE)

    # Set specific stats for type 0 and 1
    standardizer.means[0] = 5.0
    standardizer.stds[0] = 2.0
    standardizer.means[1] = 10.0
    standardizer.stds[1] = 5.0

    # Transform: z = (x - mean) / std
    # Type 0: (10-5)/2 = 2.5, (30-5)/2 = 12.5
    # Type 1: (20-10)/5 = 2.0
    transformed = standardizer.transform(dummy_vals, dummy_types)
    expected_trans = torch.tensor([2.5, 2.0, 12.5], device=Config.DEVICE)

    assert torch.allclose(
        transformed, expected_trans, atol=1e-5
    ), "Standardizer transform failed"

    # Inverse Transform
    restored = standardizer.inverse_transform(transformed, dummy_types)
    assert torch.allclose(
        restored, dummy_vals, atol=1e-5
    ), "Standardizer inverse_transform failed"

    print("    Standardizer transform/inverse logic verified.")

    # Re-fit standardizer on actual data for training
    print("    Re-fitting standardizer on loaded training data...")
    # Flatten arrays for DF construction
    train_data_dict = train_loader.dataset.__dict__
    all_types = np.concatenate(train_data_dict["coupling_types"])
    all_values = np.concatenate(train_data_dict["coupling_values"])
    std_df = pd.DataFrame({"type": all_types, "scalar_coupling_constant": all_values})
    standardizer.fit(std_df, load_cached_data=False)

    # ---------------------------------------------------------
    # 4. Model Initialization and Forward Pass
    # ---------------------------------------------------------
    print("\n[4] Initializing SDIN Model...")
    model = SDIN().to(Config.DEVICE)
    print("    Model initialized.")

    print("    Running forward pass on a single batch...")
    # Move batch to device
    b_atom_types = batch["atom_types"].to(Config.DEVICE)
    b_atom_coords = batch["atom_coords"].to(Config.DEVICE)
    b_batch_index = batch["batch_index"].to(Config.DEVICE)
    b_coupling_pairs = batch["coupling_pairs"].to(Config.DEVICE)
    b_coupling_types = batch["coupling_types"].to(Config.DEVICE)

    output = model(
        atom_types=b_atom_types,
        atom_coords=b_atom_coords,
        batch_index=b_batch_index,
        coupling_pairs=b_coupling_pairs,
        coupling_types=b_coupling_types,
    )

    assert output.shape == (
        num_couplings,
    ), f"Model output shape mismatch. Expected ({num_couplings},), got {output.shape}"
    print("    Forward pass successful. Output shape verified.")

    # ---------------------------------------------------------
    # 5. Training Loop Execution
    # ---------------------------------------------------------
    print("\n[5] Executing Training Loop (2 Epochs)...")
    logger = Logger(os.path.join(Config.WORKING_DIR, "train.log"))
    trainer = Trainer(model, train_loader, val_loader, standardizer, logger)

    start_time = time.time()
    best_score = trainer.fit()
    end_time = time.time()

    print(f"    Training completed in {end_time - start_time:.2f} seconds.")
    print(f"    Best Validation LMAE Score: {best_score}")

    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model checkpoint file was not created."
    print("    Checkpoint file verified.")

    # ---------------------------------------------------------
    # 6. Inference and Submission Generation
    # ---------------------------------------------------------
    print("\n[6] Running Inference on Test Set...")

    # Load best model
    loaded_score = load_checkpoint(
        model, path=Config.MODEL_SAVE_PATH, device=Config.DEVICE
    )
    assert loaded_score == best_score, "Loaded score does not match best score."

    # Generate predictions
    predict_test(model, test_loader, standardizer, Config.SUBMISSION_PATH)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission shape: {sub_df.shape}")
    assert (
        "id" in sub_df.columns and "scalar_coupling_constant" in sub_df.columns
    ), "Submission columns missing."
    assert len(sub_df) > 0, "Submission file is empty."

    print("    Submission file verified.")

    print("\n===========================================================")
    print("   Demonstration Completed Successfully")
    print("===========================================================")


if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    run_demo()
