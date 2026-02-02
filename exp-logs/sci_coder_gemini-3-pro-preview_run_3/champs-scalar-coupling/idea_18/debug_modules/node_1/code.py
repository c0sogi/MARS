import os
import torch
import numpy as np
import pandas as pd
import shutil
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.data_preprocessing import SoAPreprocessor
from library.dataset import MoleculeGraphDataset, GraphCollate
from library.model import MPDIN
from library.utils import GroupStandardizer, LogMAE
from library.trainer import Trainer


def run_demo():
    print("=== Starting Demonstration of Scalar Coupling Prediction Pipeline ===\n")

    # ==========================================
    # 1. Configuration Override for Speed
    # ==========================================
    print("[1] Configuring environment for fast execution...")

    # Patch Config to run a lightweight version
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Small subset for demonstration
    Config.MAX_EPOCHS = 1  # Single epoch training
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.WORKING_DIR = "./working/demo_execution"

    # Update derived paths in Config based on new WORKING_DIR
    Config.CACHE_DIR_TRAIN = os.path.join(Config.WORKING_DIR, "train_cache")
    Config.CACHE_DIR_VAL = os.path.join(Config.WORKING_DIR, "val_cache")
    Config.CACHE_DIR_TEST = os.path.join(Config.WORKING_DIR, "test_cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(
        Config.WORKING_DIR, "submission", "submission.csv"
    )

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Setup directories
    Config.setup_directories()
    print("    Configuration patched. Debug mode enabled.")

    # ==========================================
    # 2. Data Preprocessing Verification
    # ==========================================
    print("\n[2] Verifying Data Preprocessing...")

    preprocessor = SoAPreprocessor()

    # Process training split (force reload to test logic)
    print("    Processing 'train' split...")
    data_dict = preprocessor.process_split("train", load_cached_data=False)

    # Assertions
    assert "atom_types" in data_dict, "Missing atom_types in processed data"
    assert "edge_indices" in data_dict, "Missing edge_indices in processed data"
    assert (
        "coupling_values" in data_dict
    ), "Missing coupling_values (targets) in train data"
    assert (
        len(data_dict["mol_names"]) <= Config.DEBUG_SAMPLE_SIZE
    ), "Debug sample size limit ignored"

    print(f"    Processed {len(data_dict['mol_names'])} molecules.")
    print("    Data Preprocessing Verified.")

    # ==========================================
    # 3. Dataset & Collation Verification
    # ==========================================
    print("\n[3] Verifying Dataset and Graph Collation...")

    # Initialize Dataset
    train_dataset = MoleculeGraphDataset(split="train", load_cached_data=True)
    assert len(train_dataset) > 0, "Dataset is empty"

    # Get a single sample
    sample = train_dataset[0]
    assert "atom_types" in sample
    assert "edge_index" in sample
    assert "coupling_value" in sample

    # Initialize Collate function
    collate_fn = GraphCollate()

    # Create a small batch
    batch_list = [train_dataset[i] for i in range(min(4, len(train_dataset)))]
    batch = collate_fn(batch_list)

    # Check Batch Structure
    assert "batch" in batch, "Batch vector missing"
    assert batch["batch"].max() == len(batch_list) - 1, "Batch indexing incorrect"
    assert "coupling_value" in batch, "Targets missing in batch"

    # Verify Edge Index Offsetting
    # The max edge index should correspond to the total number of nodes in the batch
    total_nodes = batch["x"].shape[0]
    if batch["edge_index"].shape[1] > 0:
        assert batch["edge_index"].max() < total_nodes, "Edge indices out of bounds"

    print(f"    Batch created with {len(batch_list)} graphs.")
    print("    Dataset & Collation Verified.")

    # ==========================================
    # 4. Model Architecture Verification
    # ==========================================
    print("\n[4] Verifying Model Architecture (MPDIN)...")

    device = torch.device("cpu")  # Use CPU for simple verification
    model = MPDIN().to(device)

    # Forward Pass
    # Ensure batch is on the correct device (CPU)
    model.eval()
    with torch.no_grad():
        preds = model(batch)

    # Check Output Shape
    num_couplings = batch["coupling_value"].shape[0]
    assert preds.shape == (
        num_couplings,
    ), f"Output shape mismatch. Expected ({num_couplings},), got {preds.shape}"

    print("    Forward pass successful.")
    print("    Model Architecture Verified.")

    # ==========================================
    # 5. Utility Functions Verification
    # ==========================================
    print("\n[5] Verifying Utility Functions...")

    # --- GroupStandardizer ---
    print("    Testing GroupStandardizer...")
    standardizer = GroupStandardizer(device=device)

    # Create dummy data
    # Type 0: 1JHC (Mean ~94.95, Std ~18.25)
    # Type 2: 2JHC (Mean ~-0.28, Std ~4.50)
    dummy_types = torch.tensor([0, 0, 2, 2], device=device)
    dummy_vals = torch.tensor([94.95, 113.2, -0.28, 4.22], device=device)

    # Transform
    z_scores = standardizer.transform(dummy_vals, dummy_types)

    # Inverse Transform
    reconstructed = standardizer.inverse_transform(z_scores, dummy_types)

    # Verify Reconstruction
    assert torch.allclose(
        dummy_vals, reconstructed, atol=1e-5
    ), "Standardizer inverse transform failed"
    print("    Standardizer logic correct.")

    # --- LogMAE ---
    print("    Testing LogMAE Metric...")
    # Perfect prediction should yield very low LogMAE (technically -inf for 0 error, but code handles floats)
    # Let's use small error
    preds = torch.tensor([100.0, 50.0], device=device)
    targets = torch.tensor([100.0, 50.0], device=device)
    types = torch.tensor([0, 1], device=device)

    # Exact match -> MAE=0 -> log(0) -> -inf.
    # Let's test with slight error to get a valid number
    preds = torch.tensor([101.0, 51.0], device=device)  # Error 1.0 for both
    targets = torch.tensor([100.0, 50.0], device=device)

    # MAE for type 0: 1.0 -> log(1.0) = 0.0
    # MAE for type 1: 1.0 -> log(1.0) = 0.0
    # Avg LogMAE = 0.0
    metric = LogMAE.compute(preds, targets, types)
    assert (
        torch.abs(metric) < 1e-5
    ), f"LogMAE calculation incorrect. Expected ~0.0, got {metric}"

    print("    LogMAE metric logic correct.")

    # ==========================================
    # 6. Full Trainer Loop Verification
    # ==========================================
    print("\n[6] Verifying Full Training Loop...")

    # Initialize Trainer
    # This will load datasets (using cached debug data generated above)
    trainer = Trainer()

    # Run Training (fit)
    print("    Running trainer.fit() (1 Epoch)...")
    trainer.fit()

    # Check if model checkpoint was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model checkpoint not found after training"
    print("    Training complete. Checkpoint saved.")

    # Run Prediction/Submission
    print("    Running trainer.predict_and_submit()...")

    # Ensure test cache is populated (Trainer expects it)
    # In a real run, trainer does this, but we want to ensure debug limits apply
    # The Trainer class initializes test_dataset which calls process_split('test')
    # Since we set Config.DEBUG=True globally, it should handle it.

    trainer.predict_and_submit()

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "id" in df_sub.columns and "scalar_coupling_constant" in df_sub.columns
    ), "Submission columns incorrect"
    assert len(df_sub) > 0, "Submission file is empty"

    print(f"    Submission generated with {len(df_sub)} rows.")
    print("    Full Training Loop Verified.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
