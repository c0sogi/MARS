import os
import sys
import shutil
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library files
from library.config import Config
from library.data import get_data, CrystalGraphDataset, collate_graphs
from library.model import DBGT
from library.utils import setup_logger, compute_rmsle, TargetScaler
from library.engine import run_training, generate_submission


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration and Setup
    # -------------------------------------------------------------------------
    print(">>> Setting up configuration for demo...")

    # Modify Config for a fast demonstration run
    Config.DEBUG = True
    Config.DEBUG_SIZE = 50  # Use only 50 samples
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Set up a specific working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create necessary directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Processing and Loading Verification
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Data Loading components...")

    # Load a small subset of training data directly
    # We set load_cached_data=False to ensure the processing logic is tested
    graphs, targets, ids = get_data(
        Config.TRAIN_METADATA_PATH,
        "train",
        load_cached_data=False,
        debug=True,
        debug_size=Config.DEBUG_SIZE,
    )

    # Assertions for data integrity
    assert len(graphs) == len(targets) == len(ids), "Mismatch in data lengths."
    assert len(graphs) > 0, "No graphs loaded."
    assert "atomic_numbers" in graphs[0], "Graph missing atomic_numbers."
    assert "edge_index" in graphs[0], "Graph missing edge_index."
    assert "edge_distances" in graphs[0], "Graph missing edge_distances."

    print(f"    Successfully loaded {len(graphs)} graphs.")

    # Instantiate Dataset
    dataset = CrystalGraphDataset(graphs, targets, ids)
    assert len(dataset) == len(graphs)

    # Instantiate DataLoader with custom collate function
    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_graphs,
        num_workers=0,
    )

    # Fetch a single batch to verify collation
    batch = next(iter(dataloader))

    # Verify batch structure
    assert "x" in batch
    assert "edge_index" in batch
    assert "edge_attr" in batch
    assert "batch" in batch
    assert "y" in batch

    # Verify shapes
    # x: (num_nodes_in_batch,)
    # edge_index: (2, num_edges_in_batch)
    # y: (batch_size, num_targets)
    assert batch["x"].dim() == 1
    assert batch["edge_index"].dim() == 2 and batch["edge_index"].size(0) == 2
    assert batch["y"].dim() == 2 and batch["y"].size(1) == 2
    assert batch["y"].size(0) == Config.BATCH_SIZE or batch["y"].size(0) == len(graphs)

    print("    Data Loading and Collation verified.")

    # -------------------------------------------------------------------------
    # 3. Model Verification
    # -------------------------------------------------------------------------
    print("\n>>> Verifying DBGT Model...")

    model = DBGT(config=Config).to(device)

    # Move batch to device
    batch_device = {
        "x": batch["x"].to(device),
        "edge_index": batch["edge_index"].to(device),
        "edge_attr": batch["edge_attr"].to(device),
        "batch": batch["batch"].to(device),
    }

    # Forward pass
    output = model(batch_device)

    # Verify output shape
    assert output.shape == (
        batch["y"].size(0),
        2,
    ), f"Expected output shape {(batch['y'].size(0), 2)}, got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs."

    print("    Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 4. Utility Verification
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Utilities...")

    # Test TargetScaler
    scaler = TargetScaler()
    dummy_targets = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=np.float32)
    scaler.fit(dummy_targets)

    transformed = scaler.transform(dummy_targets)
    inverse = scaler.inverse_transform(transformed)

    assert np.allclose(
        dummy_targets, inverse, atol=1e-5
    ), "TargetScaler reconstruction failed."
    print("    TargetScaler verified.")

    # Test compute_rmsle
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.1, 1.9], [3.2, 3.8]])
    rmsle = compute_rmsle(y_true, y_pred)
    assert isinstance(rmsle, float), "compute_rmsle should return a float."
    assert rmsle >= 0, "RMSLE cannot be negative."
    print(f"    compute_rmsle verified (val: {rmsle:.4f}).")

    # -------------------------------------------------------------------------
    # 5. Full Pipeline Execution (Training & Inference)
    # -------------------------------------------------------------------------
    print("\n>>> Running Full Training Pipeline (1 Epoch, Debug Mode)...")

    # This function encapsulates the entire training loop including:
    # - Data loading (train/val)
    # - Model initialization
    # - Optimizer setup
    # - Training loop
    # - Validation
    # - Checkpointing
    # We force load_cached_data=False to verify processing logic.
    run_training(load_cached_data=False)

    # Verify checkpoint creation
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model_runfile.pth")
    assert os.path.exists(best_model_path), f"Checkpoint not found at {best_model_path}"
    print(f"    Training complete. Checkpoint found at {best_model_path}")

    print("\n>>> Running Inference Pipeline...")

    # This function loads the best model and generates predictions for the test set
    generate_submission(load_cached_data=False)

    # Verify submission file creation
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        len(sub_df) == Config.DEBUG_SIZE
    ), f"Expected {Config.DEBUG_SIZE} predictions, got {len(sub_df)}"
    assert list(sub_df.columns) == [
        "id",
        "formation_energy_ev_natom",
        "bandgap_energy_ev",
    ], "Incorrect submission columns."
    assert not sub_df.isnull().values.any(), "Submission contains NaNs."

    print(f"    Inference complete. Submission saved to {Config.SUBMISSION_PATH}")
    print("\n>>> All demonstrations and verifications passed successfully.")


if __name__ == "__main__":
    main()
