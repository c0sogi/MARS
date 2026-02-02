import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library files
from library.utils import set_seed
from library.dataset import load_and_process_data, make_loader, IcebergDataset
from library.model import SPCNN
from library.train import run_training


def demo_pipeline():
    print("=== Starting Demo Pipeline ===")

    # 1. Setup
    # Define a separate working directory for this demonstration to verify outputs
    DEMO_WORK_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_WORK_DIR):
        shutil.rmtree(DEMO_WORK_DIR)
    os.makedirs(DEMO_WORK_DIR)

    # Set seed for reproducibility
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Processing Demonstration
    print("\n--- Testing Data Processing ---")
    # We force load_cached_data=False to verify the processing logic from scratch.
    # Note: dataset.py hardcodes CACHE_DIR = "./working/idea_55/", so this will write files there.
    train_data, val_data, test_data = load_and_process_data(load_cached_data=False)

    # Verify Data Shapes
    # X shape should be (N, 3, 75, 75)
    assert train_data["X"].ndim == 4, "Train X should be 4-dimensional"
    assert train_data["X"].shape[1:] == (
        3,
        75,
        75,
    ), "Image dimensions should be (3, 75, 75)"
    assert len(train_data["X"]) == len(
        train_data["y"]
    ), "Mismatch between X and y length"
    assert len(train_data["X"]) == len(
        train_data["angles"]
    ), "Mismatch between X and angles length"

    print(f"Train samples: {len(train_data['X'])}")
    print(f"Val samples: {len(val_data['X'])}")
    print(f"Test samples: {len(test_data['X'])}")
    print("Data processing and shape verification passed.")

    # 3. DataLoader Demonstration
    print("\n--- Testing Data Loaders ---")
    # Create loaders with a small batch size for inspection
    train_loader, val_loader, test_loader = make_loader(
        batch_size=8, num_workers=0, load_cached_data=True
    )

    # Fetch a single batch from the training loader
    batch = next(iter(train_loader))
    images = batch["image"]
    angles = batch["angle"]
    labels = batch["label"]
    ids = batch["id"]

    # Verify batch tensor shapes
    assert images.shape == (
        8,
        3,
        75,
        75,
    ), f"Unexpected image batch shape: {images.shape}"
    assert angles.shape == (8,), f"Unexpected angle batch shape: {angles.shape}"
    assert labels.shape == (8,), f"Unexpected label batch shape: {labels.shape}"
    assert len(ids) == 8, "Unexpected number of IDs in batch"
    print("DataLoader batch shape verification passed.")

    # 4. Model Demonstration
    print("\n--- Testing Model Architecture ---")
    model = SPCNN().to(device)

    # Perform a forward pass with the fetched batch
    images = images.to(device)
    angles = angles.to(device)

    logits = model(images, angles)

    # Output should be a flat vector of logits (Batch_Size,)
    assert logits.shape == (
        8,
    ), f"Model output shape mismatch. Expected (8,), got {logits.shape}"
    print("Model forward pass successful. Output shape matches batch size.")

    # 5. Training Loop Demonstration
    print("\n--- Testing Training Loop (Fast Run) ---")
    # Run the training pipeline for a minimal number of epochs (2) to ensure it completes quickly
    # We pass the DEMO_WORK_DIR so artifacts are saved there
    run_training(epochs=2, patience=1, batch_size=16, lr=1e-3, work_dir=DEMO_WORK_DIR)

    # 6. Submission Verification
    print("\n--- Verifying Submission Output ---")
    submission_path = os.path.join(DEMO_WORK_DIR, "submission", "submission.csv")

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not generated at {submission_path}")

    sub_df = pd.read_csv(submission_path)

    # Check for required columns
    assert "id" in sub_df.columns, "Submission missing 'id' column"
    assert "is_iceberg" in sub_df.columns, "Submission missing 'is_iceberg' column"

    # Check that the number of predictions matches the test set size
    assert len(sub_df) == len(
        test_data["X"]
    ), f"Submission length {len(sub_df)} does not match test set size {len(test_data['X'])}"

    # Check that probabilities are within the valid range [0, 1]
    # Note: The model outputs logits, but predict() applies sigmoid, so values must be in [0, 1]
    assert sub_df["is_iceberg"].min() >= 0.0, "Found probability < 0"
    assert sub_df["is_iceberg"].max() <= 1.0, "Found probability > 1"

    print("Submission file format and content verified.")
    print("\n=== Demo Pipeline Completed Successfully ===")


if __name__ == "__main__":
    demo_pipeline()
