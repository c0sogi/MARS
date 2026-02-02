import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil
from unittest.mock import patch

# Import library components
# Note: We import Config directly to modify class attributes for the demo
from library.config import Config
from library.data import get_dataloaders
from library.model import LI_CGCNN_ELR
from library.train import run_training
from library.utils import set_seed


def main():
    print("=== Starting LI-CGCNN-ELR Library Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override
    # -------------------------------------------------------------------------
    print("\n[1] Configuring demo parameters...")

    # Set a specific working directory for this demo
    demo_dir = "./working/demo_run"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)

    # Override Config class attributes directly
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.CHECKPOINT_DIR = os.path.join(demo_dir, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Update cache paths
    Config.TRAIN_GRAPHS_CACHE = os.path.join(Config.CACHE_DIR, "train_graphs.npz")
    Config.VAL_GRAPHS_CACHE = os.path.join(Config.CACHE_DIR, "val_graphs.npz")
    Config.TEST_GRAPHS_CACHE = os.path.join(Config.CACHE_DIR, "test_graphs.npz")
    Config.TARGET_SCALER_PATH = os.path.join(Config.CACHE_DIR, "target_scaler.npz")
    Config.LATTICE_SCALER_PATH = os.path.join(Config.CACHE_DIR, "lattice_scaler.npz")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    # Reduce compute load for demo
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.HIDDEN_DIM = 32  # Reduce model size if parameter existed (it doesn't in provided Config, but good practice)

    # Ensure reproducibility
    set_seed(Config.SEED)

    print(f"Working directory set to: {Config.WORKING_DIR}")
    print(f"Epochs: {Config.NUM_EPOCHS}, Batch Size: {Config.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Data Loading with Subset (Monkey Patching)
    # -------------------------------------------------------------------------
    print("\n[2] Loading data (using subset for speed)...")

    # We monkey-patch pd.read_csv to load only the first 50 rows of the metadata files.
    # This ensures the demo runs quickly without processing the entire dataset.
    original_read_csv = pd.read_csv

    def subset_read_csv(filepath_or_buffer, *args, **kwargs):
        # Only subset the metadata files we know are large
        if isinstance(filepath_or_buffer, str) and "metadata.csv" in filepath_or_buffer:
            kwargs["nrows"] = 50
        return original_read_csv(filepath_or_buffer, *args, **kwargs)

    with patch("pandas.read_csv", side_effect=subset_read_csv):
        # Force reload from scratch (ignore existing cache) to demonstrate processing logic
        train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Validation of Data Loaders
    print("Validating DataLoaders...")
    assert len(train_loader.dataset) > 0, "Train dataset is empty"
    assert len(val_loader.dataset) > 0, "Val dataset is empty"
    assert len(test_loader.dataset) > 0, "Test dataset is empty"

    # Inspect a single batch
    sample_batch = next(iter(train_loader))
    print(f"Sample Batch Keys: {sample_batch.keys()}")

    # Check dimensions
    # x: [num_nodes] (atomic numbers are scalar longs, but embedding layer expects 1D input)
    assert (
        sample_batch.x.dim() == 1
    ), f"Expected 1D node features, got {sample_batch.x.shape}"
    # edge_index: [2, num_edges]
    assert sample_batch.edge_index.shape[0] == 2, "Edge index should have 2 rows"
    # lattice_params: [batch_size, 6]
    assert sample_batch.lattice_params.shape == (
        sample_batch.num_graphs,
        6,
    ), f"Expected lattice params shape {(sample_batch.num_graphs, 6)}, got {sample_batch.lattice_params.shape}"
    # y: [batch_size, 2]
    assert sample_batch.y.shape == (
        sample_batch.num_graphs,
        2,
    ), f"Expected target shape {(sample_batch.num_graphs, 2)}, got {sample_batch.y.shape}"

    print("Data loading and validation successful.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization and Forward Pass
    # -------------------------------------------------------------------------
    print("\n[3] Initializing Model and testing forward pass...")

    device = torch.device(Config.DEVICE)
    model = LI_CGCNN_ELR(Config).to(device)

    # Move sample batch to device
    sample_batch = sample_batch.to(device)

    # Run forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(sample_batch)

    print(f"Model Output Shape: {outputs.shape}")

    # Verify output shape matches [batch_size, 2]
    assert outputs.shape == (
        sample_batch.num_graphs,
        2,
    ), f"Expected output shape {(sample_batch.num_graphs, 2)}, got {outputs.shape}"

    print("Forward pass successful.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[4] Running Training Loop...")

    # We use the patched read_csv again because run_training calls get_dataloaders internally.
    # However, since we already generated the cache in step 2 (get_dataloaders saves to cache),
    # run_training will pick up the cached subset data automatically if load_cached_data=True.
    # We must patch here because run_training calls generate_submission, which reads metadata directly.

    with patch("pandas.read_csv", side_effect=subset_read_csv):
        trained_model = run_training(
            load_cached_data=True, num_epochs=Config.NUM_EPOCHS
        )

    # Verify checkpoint creation
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {Config.BEST_MODEL_PATH}")
    print(f"Checkpoint verified at {Config.BEST_MODEL_PATH}")

    # -------------------------------------------------------------------------
    # 5. Submission Verification
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Submission...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {submission_df.shape}")
    print(f"Submission columns: {submission_df.columns.tolist()}")

    # Check for NaNs
    if submission_df.isnull().values.any():
        raise AssertionError("Submission file contains NaNs")

    # Check row count matches the subset size (50)
    # Note: We patched read_csv to return 50 rows for test_metadata as well.
    # get_dataloaders processed these 50 rows and saved them to cache.
    # run_training loaded these 50 graphs.
    # generate_submission predicted on these 50 graphs.
    # generate_submission loads test_metadata.csv to get IDs.
    # IMPORTANT: generate_submission reads test_metadata.csv from disk directly using pd.read_csv.
    # If we don't patch it there, it will read 241 rows, but we only have predictions for 50.
    # This would cause a mismatch error in generate_submission.
    # However, run_training calls generate_submission.
    # Let's verify if run_training succeeded. If it did, it means generate_submission worked.
    # Wait, in step 4 call to run_training, I did NOT use the patch context manager.
    # But get_dataloaders was called inside run_training.
    # If load_cached_data=True, get_dataloaders loads the 50 graphs from cache.
    # Then generate_submission is called. It iterates over the loader (50 graphs).
    # Then it reads Config.TEST_METADATA_PATH. This read is NOT patched in step 4.
    # So it reads all 241 rows.
    # 50 predictions vs 241 IDs -> ValueError.
    #
    # CORRECTION: I must ensure run_training runs within the patch context OR
    # I must ensure the cache matches the full metadata if I don't patch.
    # Since I created a subset cache, I MUST patch the metadata read in generate_submission as well
    # to match the subset.
    #
    # To fix this for the demo script, I will manually call generate_submission with the patch
    # to prove it works, acknowledging that the run_training call in step 4 might have failed
    # at the very end (submission generation) but succeeded in training.
    # Actually, let's wrap step 4 in the patch to be safe and correct.

    print("Re-running submission generation with patch to ensure consistency...")
    with patch("pandas.read_csv", side_effect=subset_read_csv):
        # We need to reload the test loader to ensure it's fresh/reset
        _, _, test_loader_subset = get_dataloaders(load_cached_data=True)
        from library.train import generate_submission

        generate_submission(trained_model, test_loader_subset, device)

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        len(submission_df) == 50
    ), f"Expected 50 rows in submission (subset), got {len(submission_df)}"

    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
