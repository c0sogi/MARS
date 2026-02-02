import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model import WideRepVGG
from library.engine import train_classifier, generate_submission

# --- Configuration ---
SEED = 42
DEMO_EPOCHS = 2
BATCH_SIZE = 64
WORKING_DIR = "./working/demo_run"
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")
SUBMISSION_FILE = os.path.join(SUBMISSION_DIR, "submission.csv")


def main():
    print("Starting demonstration script...")

    # 1. Reproducibility
    set_seed(SEED)

    # Ensure working directories exist
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # 2. Data Loading & Verification
    print("\n--- Verifying Data Loading ---")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=2, load_cached_data=True
    )

    # Fetch a single batch to verify shapes
    images, labels = next(iter(train_loader))
    print(f"Train batch image shape: {images.shape}")
    print(f"Train batch label shape: {labels.shape}")

    # Assertions for data integrity
    # Images should be (Batch, 3, 32, 32)
    assert images.dim() == 4, "Images must be 4D tensors"
    assert images.shape[1] == 3, "Images must have 3 channels"
    assert images.shape[2] == 32 and images.shape[3] == 32, "Images must be 32x32"
    # Labels should be (Batch,)
    assert labels.dim() == 1, "Labels must be 1D tensors"
    assert (
        len(labels) == images.shape[0]
    ), "Batch size mismatch between images and labels"
    print("Data loading verification passed.")

    # 3. Model Architecture Verification
    print("\n--- Verifying Model Architecture ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WideRepVGG(num_classes=1, deploy=False).to(device)

    # Dummy forward pass
    dummy_input = torch.randn(2, 3, 32, 32).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model output shape: {output.shape}")
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"

    # Verify re-parameterization (switch_to_deploy)
    print("Testing switch_to_deploy...")
    model.switch_to_deploy()
    with torch.no_grad():
        deploy_output = model(dummy_input)

    assert deploy_output.shape == (2, 1), "Deploy mode output shape mismatch"
    # Note: Outputs might differ slightly due to floating point precision in fusion,
    # but logic execution is what we are testing here.
    print("Model architecture verification passed.")

    # 4. Training Loop Demonstration
    print(f"\n--- Running Training (Seed={SEED}, Epochs={DEMO_EPOCHS}) ---")
    # We use the engine's train_classifier which handles the loop, validation, and saving
    best_model_path = train_classifier(
        seed=SEED,
        epochs=DEMO_EPOCHS,
        patience=1,  # Strict patience for demo speed
        batch_size=BATCH_SIZE,
        save_dir=WORKING_DIR,
    )

    assert os.path.exists(
        best_model_path
    ), f"Model file was not saved at {best_model_path}"
    print(f"Training complete. Best model saved to: {best_model_path}")

    # 5. Inference & Submission
    print("\n--- Generating Submission ---")
    # We generate submission using the trained seed
    generate_submission(seeds=[SEED], save_dir=WORKING_DIR, output_file=SUBMISSION_FILE)

    assert os.path.exists(SUBMISSION_FILE), "Submission file was not created."
    print(f"Submission generated at: {SUBMISSION_FILE}")

    # 6. Validate Submission File
    print("\n--- Validating Submission File ---")
    df_sub = pd.read_csv(SUBMISSION_FILE)

    # Check columns
    required_cols = ["id", "has_cactus"]
    assert all(
        col in df_sub.columns for col in required_cols
    ), f"Missing columns. Found: {df_sub.columns}"

    # Check row count (Test set size is 3325 based on metadata/test_metadata.csv or sample_submission)
    # We can check against the test loader length * batch size roughly, or exact number if known.
    # The provided metadata info says sample_submission has 3325 rows.
    expected_rows = 3325
    # Note: If test_loader drops last or similar, count might vary, but standard submission requires all.
    # The provided dataset.py uses default DataLoader (drop_last=False).

    print(f"Submission shape: {df_sub.shape}")
    if len(df_sub) != expected_rows:
        print(
            f"Warning: Expected {expected_rows} rows, got {len(df_sub)}. Checking against test loader..."
        )
        # Verify against actual test dataset size
        assert len(df_sub) == len(
            test_loader.dataset
        ), f"Submission row count {len(df_sub)} does not match test dataset size {len(test_loader.dataset)}"

    # Check value range
    probs = df_sub["has_cactus"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    # Check for NaNs
    assert not df_sub.isnull().values.any(), "Submission contains NaN values"

    print("Submission file validation passed.")
    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
