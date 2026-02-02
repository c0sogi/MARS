import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.utils import set_seed
from library.data_loader import get_loaders, BirdDataset
from library.model import MILResNet18
from library.train_eval import run_kfold_training


def main():
    print("Starting Demonstration Script...")

    # Constants
    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working/demo_run"
    SUBMISSION_DIR = "./working/demo_submission"

    # Clean up previous run if exists
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    if os.path.exists(SUBMISSION_DIR):
        shutil.rmtree(SUBMISSION_DIR)

    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # 1. Set Seed
    print("\n--- Step 1: Setting Seed ---")
    set_seed(42)
    print("Seed set to 42.")

    # 2. Demonstrate Data Loading & Verify Shapes
    print("\n--- Step 2: Verifying Data Loader & Dataset ---")
    # We use a small batch size for demonstration
    train_loader, val_loader, test_loader = get_loaders(
        input_dir=INPUT_DIR,
        metadata_dir=METADATA_DIR,
        batch_size=4,
        num_workers=0,  # Use 0 workers to avoid multiprocessing overhead in demo
        image_size=224,
    )

    # Fetch one batch
    inputs, targets = next(iter(train_loader))

    # Expected Input Shape: (Batch, Tiles=3, Channels=1, H=224, W=224)
    # Expected Target Shape: (Batch, Num_Classes=19)
    print(f"Input batch shape: {inputs.shape}")
    print(f"Target batch shape: {targets.shape}")

    assert inputs.dim() == 5, f"Expected 5D input tensor, got {inputs.dim()}"
    assert inputs.shape[1] == 3, f"Expected 3 tiles, got {inputs.shape[1]}"
    assert inputs.shape[2] == 1, f"Expected 1 channel, got {inputs.shape[2]}"
    assert inputs.shape[3] == 224 and inputs.shape[4] == 224, "Image size mismatch"
    assert targets.shape[1] == 19, f"Expected 19 classes, got {targets.shape[1]}"

    print("Data Loader shapes verified.")

    # 3. Demonstrate Model Instantiation & Forward Pass
    print("\n--- Step 3: Verifying Model Architecture ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate model (pretrained=False for speed in this specific check,
    # though training function uses True)
    model = MILResNet18(num_classes=19, pretrained=False).to(device)

    # Move inputs to device
    inputs = inputs.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(inputs)

    print(f"Model output shape: {outputs.shape}")
    assert outputs.shape == (4, 19), f"Expected output (4, 19), got {outputs.shape}"

    print("Model forward pass successful.")

    # 4. Run Training Pipeline (Fast Mode)
    print("\n--- Step 4: Running Training Pipeline (Fast Mode) ---")
    # We run with minimal epochs and folds to demonstrate the pipeline functions correctly
    # without consuming too much time.

    run_kfold_training(
        input_dir=INPUT_DIR,
        metadata_dir=METADATA_DIR,
        working_dir=WORKING_DIR,
        submission_dir=SUBMISSION_DIR,
        num_folds=2,  # Only 2 folds for demo
        epochs=1,  # Only 1 epoch per fold
        batch_size=8,
        learning_rate=1e-3,
        weight_decay=1e-4,
        mixup_alpha=0.4,
        image_size=224,
        patience=1,
        seed=42,
    )

    print("Training pipeline execution completed.")

    # 5. Verify Submission File
    print("\n--- Step 5: Verifying Submission File ---")
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

    assert os.path.exists(submission_path), "Submission file was not created!"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file loaded. Rows: {len(df_sub)}")
    print(df_sub.head())

    # Check columns
    assert "Id" in df_sub.columns, "Missing 'Id' column"
    assert "Probability" in df_sub.columns, "Missing 'Probability' column"

    # Check row count: Test set size (64) * 19 classes = 1216 rows
    expected_rows = 64 * 19
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Check probability range
    probs = df_sub["Probability"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    print("Submission file format verified.")
    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()
