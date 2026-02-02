import os
import torch
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import (
    DEVICE,
    MODEL_SAVE_PATH,
    SUBMISSION_FILE_PATH,
    SCALERS_CACHE_PATH,
    TRAIN_CACHE_PATH,
    VAL_CACHE_PATH,
    TEST_CACHE_PATH,
)
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import CSNWDS
from library.engine import Trainer, generate_submission


def main():
    print("Initializing demonstration script...")

    # 1. Setup
    # Set seed for reproducibility
    seed_everything(42)
    print(f"Device: {DEVICE}")

    # 2. Data Loading
    # We use a small sample size to ensure the script runs quickly.
    # This will process the raw XYZ files, compute features, and cache them.
    debug_size = 50
    print(f"Loading dataloaders with debug_sample_size={debug_size}...")

    # The get_dataloaders function handles loading metadata, processing geometry files,
    # caching processed tensors, fitting/loading scalers, and creating DataLoaders.
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        debug_sample_size=debug_size
    )

    # Verify Data Loading
    print("Verifying DataLoader batch structure...")
    try:
        batch = next(iter(train_loader))
        # batch structure from CrystalCollate: (atomic_batch, batch_index, global_batch, target_batch, id_list)
        atomic_batch, batch_index, global_batch, target_batch, id_list = batch

        print(f"  Atomic Batch Shape: {atomic_batch.shape}")  # (Total_Atoms, 9)
        print(f"  Batch Index Shape: {batch_index.shape}")  # (Total_Atoms,)
        print(f"  Global Batch Shape: {global_batch.shape}")  # (B, 12)
        print(f"  Target Batch Shape: {target_batch.shape}")  # (B, 2)
        print(f"  ID List Length: {len(id_list)}")

        assert (
            atomic_batch.dim() == 2 and atomic_batch.shape[1] == 9
        ), "Atomic features dimension mismatch"
        assert (
            global_batch.dim() == 2 and global_batch.shape[1] == 12
        ), "Global features dimension mismatch"
        assert (
            target_batch.dim() == 2 and target_batch.shape[1] == 2
        ), "Target dimension mismatch"
        assert (
            len(id_list) == target_batch.shape[0]
        ), "Batch size mismatch between IDs and targets"

        print("DataLoader verification passed.")
    except StopIteration:
        print("Error: Train loader is empty.")
        return

    # 3. Model Instantiation
    print("Instantiating CSNWDS model...")
    model = CSNWDS()
    model.to(DEVICE)

    # Verify Model Forward Pass
    print("Verifying model forward pass...")
    with torch.no_grad():
        # Move batch to device
        atomic_batch = atomic_batch.to(DEVICE)
        batch_index = batch_index.to(DEVICE)
        global_batch = global_batch.to(DEVICE)

        output = model(atomic_batch, batch_index, global_batch)
        print(f"  Model Output Shape: {output.shape}")

        assert output.shape == (target_batch.shape[0], 2), "Model output shape mismatch"
    print("Model forward pass verification passed.")

    # 4. Training
    print("Initializing Trainer...")
    trainer = Trainer(model, train_loader, val_loader, device=DEVICE)

    # Run a short training loop (2 epochs) to demonstrate functionality
    print("Running training loop (2 epochs)...")
    trainer.fit(epochs=2, patience=1)

    # Check if model checkpoint was saved
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Model checkpoint found at {MODEL_SAVE_PATH}")
    else:
        print(
            "Warning: Model checkpoint not found (might not have improved in 2 epochs)."
        )

    # 5. Inference / Submission
    print("Generating submission...")
    # We use the trained model (current state) for demonstration
    generate_submission(model, test_loader, DEVICE, SUBMISSION_FILE_PATH)

    # Verify Submission File
    if os.path.exists(SUBMISSION_FILE_PATH):
        print(f"Submission file generated at {SUBMISSION_FILE_PATH}")
        df_sub = pd.read_csv(SUBMISSION_FILE_PATH)
        print(f"  Submission rows: {len(df_sub)}")
        print(f"  Submission columns: {df_sub.columns.tolist()}")

        # Check for NaNs
        if df_sub.isnull().any().any():
            print("Warning: NaNs found in submission file.")
        else:
            print("Submission file check passed (no NaNs).")

        # Check if IDs match test loader
        test_ids = []
        for batch in test_loader:
            test_ids.extend(batch[4])

        # With debug_sample_size, we only predict for those samples
        assert len(df_sub) == len(
            test_ids
        ), f"Submission length {len(df_sub)} != Test loader length {len(test_ids)}"
        print("Submission ID count matches test loader.")

    else:
        print("Error: Submission file was not created.")

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
