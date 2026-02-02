import os
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_dataloaders
from library.model import HybridAttentionResFunnel
from library.trainer import Trainer


def main():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("Setting up configuration for demo run...")

    # Override Config for speed and isolation
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 128  # Smaller batch size for the small subset
    Config.WORKING_DIR = "./working/demo_run"

    # Manually update paths derived from WORKING_DIR (since they were defined at import time)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    Config.PROCESSED_DATA_PATH = os.path.join(Config.WORKING_DIR, "processed_data.npz")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # --------------------------------------------------------------------------
    # 2. Data Loading & Subsetting
    # --------------------------------------------------------------------------
    print("Loading datasets...")
    # We set load_cached_data=False to demonstrate the processing pipeline.
    # In a real scenario, True would be preferred.
    train_loader_full, val_loader_full, test_loader_full = get_dataloaders(
        load_cached_data=False
    )

    # Load Test IDs for submission alignment
    test_meta = pd.read_csv(Config.TEST_META_PATH)
    test_ids_full = test_meta["id"].values

    print("Subsetting data for rapid execution...")
    # Define subset sizes
    N_TRAIN = 2000
    N_VAL = 500
    N_TEST = 500

    # Create Subsets
    train_subset = Subset(train_loader_full.dataset, range(N_TRAIN))
    val_subset = Subset(val_loader_full.dataset, range(N_VAL))
    test_subset = Subset(test_loader_full.dataset, range(N_TEST))
    test_ids_subset = test_ids_full[:N_TEST]

    # Create DataLoaders for subsets
    # num_workers=0 to avoid multiprocessing overhead on small data
    train_loader = DataLoader(
        train_subset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_subset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_subset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(
        f"Subset sizes -> Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}"
    )

    # --------------------------------------------------------------------------
    # 3. Model Initialization & Verification
    # --------------------------------------------------------------------------
    print("Initializing HybridAttentionResFunnel model...")
    model = HybridAttentionResFunnel()

    # Verify Forward Pass
    print("Verifying forward pass logic...")
    dummy_batch = next(iter(train_loader))
    dummy_cont = dummy_batch["cont_features"]
    dummy_cat = dummy_batch["cat_sequence"]

    with torch.no_grad():
        output = model(dummy_cont, dummy_cat)

    # Assert shape is (Batch_Size, 1)
    expected_shape = (dummy_cont.shape[0], 1)
    if output.shape != expected_shape:
        raise AssertionError(
            f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"
        )
    print("Forward pass verified.")

    # --------------------------------------------------------------------------
    # 4. Training
    # --------------------------------------------------------------------------
    print("Initializing Trainer...")
    trainer = Trainer(model)

    print("Starting training loop (1 Epoch)...")
    trainer.fit(train_loader, val_loader)

    # Verify Model Checkpoint
    if not os.path.exists(Config.MODEL_PATH):
        raise AssertionError(f"Model checkpoint not found at {Config.MODEL_PATH}")
    print(f"Model successfully saved to {Config.MODEL_PATH}")

    # --------------------------------------------------------------------------
    # 5. Prediction & Submission
    # --------------------------------------------------------------------------
    print("Generating submission for test subset...")
    trainer.generate_submission(test_loader, test_ids_subset)

    # Verify Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError(f"Submission file not found at {Config.SUBMISSION_PATH}")

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check dimensions
    if len(df_sub) != N_TEST:
        raise AssertionError(
            f"Submission row count mismatch. Expected {N_TEST}, got {len(df_sub)}"
        )

    # Check columns
    if not set(["id", "target"]).issubset(df_sub.columns):
        raise AssertionError(
            "Submission file missing required columns 'id' or 'target'."
        )

    # Check value ranges
    probs = df_sub["target"]
    if probs.min() < 0 or probs.max() > 1:
        raise AssertionError("Predicted probabilities are out of valid range [0, 1].")

    print(f"Submission verified. Head:\n{df_sub.head()}")
    print("\nDemo execution completed successfully.")


if __name__ == "__main__":
    main()
