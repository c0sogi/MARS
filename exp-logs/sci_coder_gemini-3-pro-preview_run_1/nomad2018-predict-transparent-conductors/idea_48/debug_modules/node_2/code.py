import torch
import numpy as np
import pandas as pd
import os
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.data import get_loaders
from library.model import BA_ADS_Model
from library.train import train_model, generate_submission
from library.config import WORKING_DIR, ATOMIC_FEATURE_DIM, GLOBAL_FEATURE_DIM


def main():
    print("Starting demonstration of BA-ADS pipeline...")

    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Data Loading
    # We use debug_mode=True to load only 50 samples for speed.
    # load_cached_data=False ensures we demonstrate the processing logic
    # and don't rely on potentially pre-existing cache states for this demo run.
    # Note: The first run might take a minute to process features for the dataset
    # before slicing it for debug mode.
    print("\n[Step 1] Loading Data (Debug Mode)...")

    train_loader, val_loader, test_loader = get_loaders(
        batch_size=4,  # Small batch size for demo
        debug_mode=True,
        load_cached_data=False,
    )

    # Validation of Train Loader
    print("Validating Train Loader...")
    for batch in train_loader:
        batch_atomic, batch_index, batch_global, batch_targets, batch_ids = batch

        # Check dimensions
        # batch_atomic: (Total_Atoms, ATOMIC_FEATURE_DIM)
        assert batch_atomic.dim() == 2
        assert (
            batch_atomic.shape[1] == ATOMIC_FEATURE_DIM
        ), f"Expected atomic dim {ATOMIC_FEATURE_DIM}, got {batch_atomic.shape[1]}"

        # batch_index: (Total_Atoms,)
        assert batch_index.dim() == 1
        assert batch_index.shape[0] == batch_atomic.shape[0], "Index length mismatch"

        # batch_global: (Batch_Size, GLOBAL_FEATURE_DIM)
        assert batch_global.dim() == 2
        assert (
            batch_global.shape[1] == GLOBAL_FEATURE_DIM
        ), f"Expected global dim {GLOBAL_FEATURE_DIM}, got {batch_global.shape[1]}"

        # batch_targets: (Batch_Size, 2)
        assert batch_targets.dim() == 2
        assert batch_targets.shape[1] == 2, "Target dim mismatch"

        print(
            f"  Batch verified: {batch_ids.size(0)} crystals, {batch_atomic.size(0)} total atoms."
        )
        break  # Only check one batch

    # 2. Model Instantiation
    print("\n[Step 2] Instantiating Model...")
    model = BA_ADS_Model()
    # Basic check of model structure
    print(f"  Model class: {model.__class__.__name__}")

    # Validation of Model Forward Pass
    print("Validating Forward Pass...")
    # Use the batch fetched above
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    batch_atomic = batch_atomic.to(device)
    batch_index = batch_index.to(device)
    batch_global = batch_global.to(device)

    with torch.no_grad():
        output = model(batch_atomic, batch_index, batch_global)

    assert output.shape == (
        batch_ids.size(0),
        2,
    ), f"Expected output shape {(batch_ids.size(0), 2)}, got {output.shape}"
    print("  Forward pass successful. Output shape:", output.shape)

    # 3. Training Loop
    print("\n[Step 3] Running Training Loop (Demo)...")
    # We run for a very small number of epochs to demonstrate functionality
    model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=2,
        device=device,
    )
    print("  Training loop finished.")

    # 4. Inference / Submission
    print("\n[Step 4] Generating Submission...")
    submission_path = os.path.join(WORKING_DIR, "demo_submission.csv")

    # Generate submission using the trained model
    df_submission = generate_submission(
        model, test_loader, output_path=submission_path, device=device
    )

    # Validation of Submission
    print("Validating Submission...")
    assert os.path.exists(submission_path), "Submission file not created."
    # Debug mode uses 50 samples
    assert (
        len(df_submission) == 50
    ), f"Expected 50 predictions (debug mode), got {len(df_submission)}"

    expected_cols = ["id", "formation_energy_ev_natom", "bandgap_energy_ev"]
    assert (
        list(df_submission.columns) == expected_cols
    ), f"Columns mismatch. Got {df_submission.columns}"

    # Check for NaNs
    assert not df_submission.isnull().values.any(), "Submission contains NaNs"

    print(f"  Submission head:\n{df_submission.head()}")
    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
