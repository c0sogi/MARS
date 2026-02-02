import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.utils import set_seed
from library.model import ResUNet
from library.dataset import get_dataloaders
from library.engine import train_engine
from library.inference import run_inference


def run_demonstration():
    # --- 1. Setup & Configuration ---
    print("--- 1. Setup & Configuration ---")
    WORKING_DIR = "./working/demo_run"
    CACHE_DIR = os.path.join(WORKING_DIR, "cache")
    MODEL_SAVE_PATH = os.path.join(WORKING_DIR, "model.pth")
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Clean up previous run if exists
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(42)

    # Detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 2. Data Loading ---
    print("\n--- 2. Data Loading ---")
    # We use a small batch size and specific cache directory
    train_loader, val_data, test_data = get_dataloaders(
        metadata_dir="./metadata",
        cache_dir=CACHE_DIR,
        input_dir="./input",
        batch_size=4,
        patch_size=128,
        patches_per_image=2,  # Reduced for speed
        num_workers=2,
        load_cached=True,
        seed=42,
    )

    # Verification
    print(f"Train Loader batches: {len(train_loader)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    assert len(train_loader) > 0, "Train loader is empty."
    assert len(val_data) > 0, "Validation data is empty."
    assert len(test_data) > 0, "Test data is empty."

    # Check batch shape
    sample_noisy, sample_clean = next(iter(train_loader))
    print(f"Sample batch shape: {sample_noisy.shape}")
    # Expected: (Batch_Size, 1, Patch_Size, Patch_Size)
    assert sample_noisy.shape == (
        4,
        1,
        128,
        128,
    ), f"Unexpected batch shape: {sample_noisy.shape}"
    assert sample_clean.shape == (
        4,
        1,
        128,
        128,
    ), f"Unexpected label shape: {sample_clean.shape}"

    # --- 3. Model Verification ---
    print("\n--- 3. Model Verification ---")
    model = ResUNet(in_channels=1, out_channels=1).to(device)

    # Dummy forward pass to check dimensions
    dummy_input = torch.randn(2, 1, 128, 128).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"Model output shape: {dummy_output.shape}")
    assert dummy_output.shape == (2, 1, 128, 128), "Model output shape mismatch."
    print("Model architecture verified.")

    # --- 4. Training Loop ---
    print("\n--- 4. Training Loop ---")
    # We limit max_train_batches and max_val_samples to ensure this runs very quickly
    best_rmse = train_engine(
        train_loader=train_loader,
        val_data=val_data,
        epochs=1,  # Single epoch for demo
        lr=1e-3,
        device=device,
        save_path=MODEL_SAVE_PATH,
        patience=1,
        max_train_batches=10,  # Limit to 10 batches
        max_val_samples=5,  # Limit validation to 5 images
        seed=42,
    )

    print(f"Training complete. Best RMSE: {best_rmse}")
    assert os.path.exists(MODEL_SAVE_PATH), "Model file was not saved."

    # --- 5. Inference ---
    print("\n--- 5. Inference ---")
    # Run inference on the test set (using TTA)
    # We will use the model we just saved
    run_inference(
        test_data=test_data,
        model_path=MODEL_SAVE_PATH,
        output_path=SUBMISSION_PATH,
        device=device,
        patch_size=128,
        overlap=32,
        use_tta=True,  # Enable TTA to demonstrate capability
    )

    assert os.path.exists(SUBMISSION_PATH), "Submission file was not generated."

    # --- 6. Submission Validation ---
    print("\n--- 6. Submission Validation ---")
    df_sub = pd.read_csv(SUBMISSION_PATH)

    print("First 5 rows of submission:")
    print(df_sub.head())

    # Check columns
    assert list(df_sub.columns) == ["id", "value"], "Submission columns are incorrect."

    # Check ID format (e.g., '110_1_1')
    sample_id = df_sub.iloc[0]["id"]
    parts = sample_id.split("_")
    assert len(parts) == 3, f"ID format incorrect: {sample_id}"

    # Check value range
    min_val = df_sub["value"].min()
    max_val = df_sub["value"].max()
    print(f"Value range: [{min_val}, {max_val}]")

    # Values should be roughly between 0 and 1.
    # Due to float precision/clipping, we check reasonable bounds.
    assert min_val >= 0, "Found negative pixel values."
    assert max_val <= 1, "Found pixel values > 1."

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    run_demonstration()
