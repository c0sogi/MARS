import os
import sys
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import SEED, DEVICE, WORKING_DIR, SUBMISSION_DIR, STATS_PATH
from library.data import get_train_val_datasets, collate_molecules
from library.model import MPIN
from library.utils import Standardizer
from library.engine import train_model


def run_demo():
    # 1. Setup & Configuration
    print("Initializing demo...")

    # Set seeds for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # Ensure output directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # 2. Data Loading
    # We use debug=True and a small debug_size to ensure the training loop is fast.
    # Note: The first run might take a few minutes to process/cache the molecular graphs
    # from the CSV files if they are not already cached.
    print("Loading datasets (this may trigger data processing)...")
    train_dataset, val_dataset = get_train_val_datasets(
        load_cached=True, debug=True, debug_size=200  # Small subset for demonstration
    )

    # Create DataLoaders
    # We use a small batch size appropriate for the small debug dataset
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_molecules,
        num_workers=0,  # Avoid multiprocessing overhead for small demo
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_molecules,
        num_workers=0,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    # 3. Model & Standardizer Initialization
    print("Initializing model and standardizer...")

    # Initialize Standardizer
    # The stats file is created during the processing of the training data
    standardizer = Standardizer(STATS_PATH)
    standardizer.load()

    # Initialize Model
    model = MPIN().to(DEVICE)

    # 4. Training
    print("Starting training...")
    # Train for a minimal number of epochs to demonstrate the loop
    best_log_mae = train_model(
        model,
        train_loader,
        val_loader,
        standardizer,
        device=DEVICE,
        epochs=2,  # Only 2 epochs for demo speed
        patience=2,
        lr=1e-3,
    )

    print(f"Training finished. Best Val LogMAE: {best_log_mae:.4f}")

    # 5. Inference & Submission Generation
    # For this demo, we will generate predictions on the validation set
    # to demonstrate the inference pipeline and output format.
    # (Processing the full test set would take longer than desired for this quick script)
    print("Generating predictions on validation set (demo for submission)...")

    model.eval()
    predictions = []
    ids = []

    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(DEVICE)

            # Predict
            preds_norm = model(batch)

            # Inverse transform to get physical values
            types = batch["coupling_type"]
            preds_phys = standardizer.inverse_transform(preds_norm, types)

            # Collect results
            predictions.append(preds_phys.cpu().numpy())
            ids.append(batch["coupling_id"].cpu().numpy())

    # Concatenate results
    all_preds = np.concatenate(predictions)
    all_ids = np.concatenate(ids)

    # Create submission DataFrame
    submission_df = pd.DataFrame({"id": all_ids, "scalar_coupling_constant": all_preds})

    # Sort by ID to match expected format
    submission_df.sort_values("id", inplace=True)

    # Save to file
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    print(f"Submission file saved to {submission_path}")
    print(f"Submission shape: {submission_df.shape}")

    # 6. Verification
    # Verify file existence and content format
    assert os.path.exists(submission_path), "Submission file was not created."

    check_df = pd.read_csv(submission_path)
    assert "id" in check_df.columns, "id column missing."
    assert (
        "scalar_coupling_constant" in check_df.columns
    ), "scalar_coupling_constant column missing."
    assert len(check_df) > 0, "Submission file is empty."

    print("Verification successful.")


if __name__ == "__main__":
    run_demo()
