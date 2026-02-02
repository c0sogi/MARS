import os
import shutil
import torch
import pandas as pd
import numpy as np
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model import MPDPCNN
from library.trainer import run_fold, generate_submission


def main():
    # --- Configuration ---
    DEMO_DIR = "./working/demo_run"
    CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

    # Use a small batch size and few epochs for speed
    BATCH_SIZE = 8
    EPOCHS = 2
    SEED = 42

    # Ensure clean slate for demo directory
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR)

    print(f"=== Starting Demo in {DEMO_DIR} ===")
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- 1. Data Loading Demo ---
    print("\n--- 1. Testing Data Loading ---")
    # We use the provided library function to get loaders
    # This will process json -> npy and cache it in CACHE_DIR
    train_loader, val_loader, test_loader = get_dataloaders(
        input_dir="./input",
        metadata_dir="./metadata",
        cache_dir=CACHE_DIR,
        batch_size=BATCH_SIZE,
        num_workers=0,  # Set to 0 for simple debugging/demo
        load_cached_data=True,
        seed=SEED,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Verify Batch Shapes
    batch = next(iter(train_loader))
    images = batch["image"]
    angles = batch["angle"]
    labels = batch["label"]
    ids = batch["id"]

    print(f"Image Batch Shape: {images.shape}")
    print(f"Angle Batch Shape: {angles.shape}")

    # Assertions
    assert images.shape == (
        BATCH_SIZE,
        3,
        75,
        75,
    ), f"Expected (B, 3, 75, 75), got {images.shape}"
    assert angles.shape == (BATCH_SIZE,), f"Expected (B,), got {angles.shape}"
    assert labels.shape == (BATCH_SIZE,), f"Expected (B,), got {labels.shape}"
    assert len(ids) == BATCH_SIZE, "ID list length mismatch"
    print("Data Loading Verification Passed.")

    # --- 2. Model Demo ---
    print("\n--- 2. Testing Model Architecture ---")
    model = MPDPCNN().to(device)

    # Move batch to device
    images = images.to(device)
    angles = angles.to(device)

    # Forward pass
    outputs = model(images, angles)

    print(f"Model Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        BATCH_SIZE,
        1,
    ), f"Expected output (B, 1), got {outputs.shape}"
    print("Model Architecture Verification Passed.")

    # --- 3. Training Loop Demo ---
    print("\n--- 3. Testing Training Loop (run_fold) ---")
    # We run just 1 fold (index 0) out of 5, for 2 epochs
    best_loss = run_fold(
        fold_idx=0,
        total_folds=5,
        epochs=EPOCHS,
        patience=2,
        batch_size=BATCH_SIZE,
        lr=1e-3,
        seed=SEED,
        input_dir="./input",
        metadata_dir="./metadata",
        working_dir=DEMO_DIR,  # Checkpoints will be saved here
        load_cached_data=True,
    )

    print(f"Training finished. Best Val Loss: {best_loss}")

    # Verify Checkpoint Creation
    expected_checkpoint = os.path.join(CHECKPOINT_DIR, "model_best_fold_0.pth")
    assert os.path.exists(
        expected_checkpoint
    ), f"Checkpoint not found at {expected_checkpoint}"
    assert isinstance(best_loss, float), "run_fold should return a float loss"
    print("Training Loop Verification Passed.")

    # --- 4. Inference Demo ---
    print("\n--- 4. Testing Inference (generate_submission) ---")
    # Generate submission using the model we just trained (Fold 0)
    generate_submission(
        fold_indices=[0],
        batch_size=BATCH_SIZE,
        input_dir="./input",
        metadata_dir="./metadata",
        working_dir=DEMO_DIR,
        output_dir=SUBMISSION_DIR,
        load_cached_data=True,
    )

    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    # Verify Submission Content
    df_sub = pd.read_csv(submission_path)
    print(f"Submission shape: {df_sub.shape}")
    print(df_sub.head())

    # Load test metadata to verify count
    df_test_meta = pd.read_csv("./metadata/test.csv")
    expected_rows = len(df_test_meta)

    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"
    assert list(df_sub.columns) == ["id", "is_iceberg"], "Incorrect submission columns"
    assert (
        df_sub["is_iceberg"].min() >= 0.0 and df_sub["is_iceberg"].max() <= 1.0
    ), "Probabilities out of bounds"

    print("Inference Verification Passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
