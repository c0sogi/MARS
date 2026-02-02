import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.train import run_training, generate_submission
from library.utils import set_seed
from library.model import CEAMSDS
from library.features import FeatureExtractor
from library.data import get_loaders


def main():
    print("Starting CEA-MS-DS Pipeline Demonstration...")

    # 1. Setup and Configuration Override for Demo
    # We override the working directories to avoid interfering with any actual training runs
    # and to ensure we can clean up or inspect specific demo artifacts.
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = "./working/demo_cache"
    Config.SUBMISSION_DIR = "./working/demo_submission"

    # Create these directories
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Update artifact paths in Config to match new directories
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pt")
    Config.SCALER_PATH = os.path.join(Config.WORKING_DIR, "scalers.npz")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Set demo hyperparameters for speed
    Config.BATCH_SIZE = 8
    Config.NUM_EPOCHS = 2
    # We will use a small debug_size to process only a few samples
    DEBUG_SIZE = 50

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration updated for demonstration.")

    # 2. Demonstrate Feature Extraction Logic explicitly
    print("\n--- Demonstrating Feature Extraction ---")
    extractor = FeatureExtractor()

    # We load the metadata manually just to pick one file for demonstration
    train_df = pd.read_csv(Config.TRAIN_CSV).iloc[:2]  # Take 2 samples
    print(f"Processing {len(train_df)} samples manually to verify shapes...")

    # Process without caching to inspect output directly
    # Note: process_data saves to cache, but returns the dict.
    # We use a dummy split name to avoid overwriting main cache if we cared,
    # but here we just want to see the objects.
    data_dict = extractor.process_data(
        train_df, load_cached_data=False, split_name="demo_manual"
    )

    X_atomic = data_dict["X_atomic"]
    X_global = data_dict["X_global"]
    y = data_dict["y"]
    batch_idx = data_dict["batch_idx"]

    print(f"X_atomic shape: {X_atomic.shape}")  # (Total atoms, 21)
    print(f"X_global shape: {X_global.shape}")  # (2, 21)
    print(f"y shape: {y.shape}")  # (2, 2)
    print(f"batch_idx shape: {batch_idx.shape}")

    # Assertions
    assert (
        X_atomic.shape[1] == Config.ATOMIC_INPUT_DIM
    ), f"Expected atomic dim {Config.ATOMIC_INPUT_DIM}, got {X_atomic.shape[1]}"
    assert (
        X_global.shape[1] == Config.GLOBAL_INPUT_DIM
    ), f"Expected global dim {Config.GLOBAL_INPUT_DIM}, got {X_global.shape[1]}"
    assert len(X_global) == 2
    assert len(y) == 2
    assert X_atomic.shape[0] == batch_idx.shape[0]
    print("Feature extraction logic verified.")

    # 3. Demonstrate Data Loading and Collation
    print("\n--- Demonstrating Data Loading ---")
    # This uses the get_loaders function which handles scaling and batching
    train_loader, val_loader, test_loader = get_loaders(
        batch_size=Config.BATCH_SIZE,
        debug_size=DEBUG_SIZE,
        load_cached_scalers=False,  # Force fitting scalers
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    x_atomic_batch, x_global_batch, y_batch, batch_idx_batch = batch

    print(
        f"Batch sizes -> Atomic: {x_atomic_batch.shape}, Global: {x_global_batch.shape}, Targets: {y_batch.shape}"
    )

    # Assertions
    # Note: Batch size might be smaller than Config.BATCH_SIZE if DEBUG_SIZE is small and it's the last batch,
    # but with DEBUG_SIZE=50 and BATCH_SIZE=8, the first batch should be 8.
    assert (
        x_global_batch.shape[0] == Config.BATCH_SIZE
    ), f"Expected batch size {Config.BATCH_SIZE}, got {x_global_batch.shape[0]}"
    assert x_atomic_batch.dim() == 2
    assert x_global_batch.dim() == 2
    print("Data loading and collation verified.")

    # 4. Demonstrate Model Forward Pass
    print("\n--- Demonstrating Model Forward Pass ---")
    device = torch.device(Config.DEVICE)
    model = CEAMSDS().to(device)

    # Move batch to device
    x_atomic_batch = x_atomic_batch.to(device)
    x_global_batch = x_global_batch.to(device)
    batch_idx_batch = batch_idx_batch.to(device)

    # Forward
    output = model(x_atomic_batch, x_global_batch, batch_idx_batch)
    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (x_global_batch.shape[0], 2), "Output shape mismatch"
    print("Model forward pass verified.")

    # 5. Run Full Training Loop (Short)
    print("\n--- Running Training Loop (Demo) ---")
    # This will train for Config.NUM_EPOCHS (2) on DEBUG_SIZE (50) samples
    # It internally calls get_loaders, so it will re-process/load the debug subset.
    run_training(debug_size=DEBUG_SIZE, num_epochs=Config.NUM_EPOCHS)

    # Verify artifact generation
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file was not created at {Config.MODEL_PATH}")
    if not os.path.exists(Config.SCALER_PATH):
        raise FileNotFoundError(f"Scaler file was not created at {Config.SCALER_PATH}")
    print("Training loop completed and artifacts verified.")

    # 6. Generate Submission
    print("\n--- Generating Submission (Demo) ---")
    # generate_submission loads the full test set (240 samples) and the best model we just trained
    generate_submission(model=None, device=device)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")

    # The test set has 240 samples.
    expected_test_size = 240
    assert (
        len(sub_df) == expected_test_size
    ), f"Expected {expected_test_size} predictions, got {len(sub_df)}"
    assert "id" in sub_df.columns
    assert "formation_energy_ev_natom" in sub_df.columns
    assert "bandgap_energy_ev" in sub_df.columns

    # Check for non-negative values (as clipped in generate_submission)
    assert (
        sub_df["formation_energy_ev_natom"] >= 0
    ).all(), "Found negative formation energy predictions"
    assert (
        sub_df["bandgap_energy_ev"] >= 0
    ).all(), "Found negative bandgap energy predictions"

    print("Submission generation verified.")
    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
