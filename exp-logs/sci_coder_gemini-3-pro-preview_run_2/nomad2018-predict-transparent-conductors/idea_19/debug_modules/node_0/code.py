import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# 1. Suppress Warnings and Progress Bars
warnings.filterwarnings("ignore")

# Monkey-patch tqdm to disable progress bars before importing library modules
import tqdm


def silent_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.tqdm = silent_tqdm

# 2. Import Library Modules
# We need to modify Config paths *before* other logic might use them,
# though in this codebase they are mostly used at runtime.
from library.config import Config

# Define a demo working directory
DEMO_DIR = "./working/demo_run"
if os.path.exists(DEMO_DIR):
    shutil.rmtree(DEMO_DIR)
os.makedirs(DEMO_DIR, exist_ok=True)

# Override Config paths to use the demo directory
Config.WORKING_DIR = DEMO_DIR
Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
Config.CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
Config.SUBMISSION_DIR = "./working/demo_submission"
Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

# Update cache paths
Config.TRAIN_GRAPH_CACHE = os.path.join(Config.CACHE_DIR, "train_graphs.npz")
Config.VAL_GRAPH_CACHE = os.path.join(Config.CACHE_DIR, "val_graphs.npz")
Config.TEST_GRAPH_CACHE = os.path.join(Config.CACHE_DIR, "test_graphs.npz")
Config.TARGET_SCALER_CACHE = os.path.join(Config.CACHE_DIR, "target_scaler.npz")

# Ensure directories exist
os.makedirs(Config.CACHE_DIR, exist_ok=True)
os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

# Import rest of the library
from library.utils import set_seed, get_device
from library.data import get_dataloaders
from library.model import SR_CGN_DP
from library.train import run_training


def main():
    print("--- Starting Library Usage Demo ---")

    # Set seed for reproducibility
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 1. Data Loading Demonstration
    # -------------------------------------------------------------------------
    print("\n[Demo] 1. Data Loading")
    print("Initializing DataLoaders with a small subset (N=50)...")

    # We use a small batch size and sample size for speed
    BATCH_SIZE = 4
    SAMPLE_SIZE = 50

    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        batch_size=BATCH_SIZE,
        num_workers=0,  # Use 0 workers for simple debugging/demo to avoid multiprocessing overhead
        data_sample_size=SAMPLE_SIZE,
        load_cached_data=False,  # Force processing to demonstrate graph creation
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Inspect a single batch
    batch = next(iter(train_loader))
    print(f"Batch structure: {batch}")

    # Assertions to verify data integrity
    assert batch.x.ndim == 1, "Node features should be 1D (atomic numbers)"
    assert batch.edge_index.shape[0] == 2, "Edge index should have 2 rows"
    assert batch.edge_attr.ndim == 2, "Edge attributes should be 2D"
    assert batch.y.shape == (BATCH_SIZE, 2), f"Targets should be ({BATCH_SIZE}, 2)"
    assert batch.batch.max() == BATCH_SIZE - 1, "Batch indices should match batch size"

    print("Data Loading verification passed.")

    # -------------------------------------------------------------------------
    # 2. Model Instantiation and Forward Pass
    # -------------------------------------------------------------------------
    print("\n[Demo] 2. Model Initialization")

    model = SR_CGN_DP(
        node_dim=64,  # Reduced dim for demo speed
        num_layers=2,  # Reduced layers for demo speed
        dropout_rate=0.1,
        rbf_bins=30,
    ).to(device)

    print("Model architecture created.")

    # Move batch to device
    batch = batch.to(device)

    # Forward pass
    with torch.no_grad():
        output = model(batch)

    print(f"Model output shape: {output.shape}")

    # Assertions for model output
    assert output.shape == (BATCH_SIZE, 2), "Output shape mismatch"
    assert not torch.isnan(output).any(), "Model produced NaNs"

    print("Model forward pass verification passed.")

    # -------------------------------------------------------------------------
    # 3. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[Demo] 3. Running Training Loop")

    # We run the training function provided in library.train
    # This will train, validate, save checkpoints, and generate a submission
    best_val_loss = run_training(
        data_sample_size=SAMPLE_SIZE,
        num_epochs=2,  # Very few epochs for demo
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=2,
        load_cached_data=True,  # Load the cache we just created in step 1
    )

    print(f"Training finished. Best Val Loss: {best_val_loss:.4f}")

    # -------------------------------------------------------------------------
    # 4. Output Verification
    # -------------------------------------------------------------------------
    print("\n[Demo] 4. Verifying Outputs")

    # Check Checkpoint
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found at: {checkpoint_path}")
    else:
        raise FileNotFoundError("Checkpoint file was not created.")

    # Check Submission
    submission_path = Config.SUBMISSION_PATH
    if os.path.exists(submission_path):
        print(f"Submission found at: {submission_path}")

        # Verify submission content
        df_sub = pd.read_csv(submission_path)
        print(f"Submission shape: {df_sub.shape}")
        print("First 3 rows:")
        print(df_sub.head(3))

        expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
        assert (
            list(df_sub.columns) == expected_cols
        ), f"Columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"
        assert len(df_sub) == min(
            240, SAMPLE_SIZE
        ), "Submission row count mismatch"  # Test set has 240 rows, but we subsetted

        # Check for valid values (no NaNs, reasonable range)
        assert not df_sub.isnull().values.any(), "Submission contains NaNs"

    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    main()
