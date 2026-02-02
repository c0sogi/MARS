import sys
import os
import torch
import pandas as pd
import numpy as np
import shutil

# Ensure the current directory is in the path to import library modules
sys.path.append(".")

from library.config import Config, seed_everything
from library.dataset import process_metadata, DogDataset, get_transforms
from library.model import DogClassifier
from library.trainer import run_training_and_submission


def main():
    # ---------------------------------------------------------
    # 1. Setup and Configuration
    # ---------------------------------------------------------
    print("--- 1. Setting up Demo Configuration ---")

    # Set seed for reproducibility
    seed_everything(42)

    # Override Config attributes for a fast, isolated demo run
    Config.debug = True  # Triggers data subsetting in trainer.py
    Config.working_dir = "./working/demo_script_run"
    Config.submission_path = os.path.join(Config.working_dir, "submission_demo.csv")
    Config.batch_size = 8  # Small batch size for demo
    Config.num_workers = 2

    # Ensure the working directory exists
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)
    os.makedirs(Config.working_dir, exist_ok=True)

    print(f"Working Directory: {Config.working_dir}")
    print(f"Debug Mode: {Config.debug}")

    # ---------------------------------------------------------
    # 2. Data Pipeline Verification
    # ---------------------------------------------------------
    print("\n--- 2. Verifying Data Pipeline ---")

    # Generate metadata (force reload to ensure logic runs)
    train_df, val_df, test_df, classes = process_metadata(load_cached_data=False)

    print(f"Number of classes: {len(classes)}")
    assert len(classes) == 120, f"Expected 120 classes, found {len(classes)}"

    # Create a small dataset instance to verify transforms and loading
    class_to_idx = {c: i for i, c in enumerate(classes)}
    # Take first 5 samples
    sample_df = train_df.head(5).copy()

    dataset = DogDataset(
        sample_df,
        transform=get_transforms(mode="train"),
        mode="train",
        label_map=class_to_idx,
    )

    # Fetch one sample
    image, label = dataset[0]

    print(f"Sample Image Shape: {image.shape}")
    print(f"Sample Label: {label}")

    # Assertions
    assert image.shape == (
        3,
        256,
        256,
    ), f"Image shape mismatch. Expected (3, 256, 256), got {image.shape}"
    assert isinstance(label, torch.Tensor), "Label must be a torch.Tensor"
    assert 0 <= label.item() < 120, "Label index out of bounds"

    # ---------------------------------------------------------
    # 3. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n--- 3. Verifying Model Architecture ---")

    # Instantiate model (pretrained=False to avoid download overhead for this check)
    model = DogClassifier(num_classes=len(classes), pretrained=False)
    model.eval()

    # Create dummy input batch
    dummy_input = torch.randn(4, 3, 256, 256)

    # Forward pass
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"Output Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        4,
        120,
    ), f"Logits shape mismatch. Expected (4, 120), got {logits.shape}"

    # ---------------------------------------------------------
    # 4. Full Pipeline Execution
    # ---------------------------------------------------------
    print("\n--- 4. Executing Full Training & Inference Pipeline (Debug Mode) ---")

    # This function handles:
    # - Subsetting data (since Config.debug=True)
    # - Training 2 folds (Head Adaptation -> FineTuning -> SWA)
    # - Inference on Test Set
    # - Generating Submission CSV
    run_training_and_submission()

    # ---------------------------------------------------------
    # 5. Output Validation
    # ---------------------------------------------------------
    print("\n--- 5. Validating Submission Output ---")

    if not os.path.exists(Config.submission_path):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.submission_path}"
        )

    submission_df = pd.read_csv(Config.submission_path)
    print("Submission File Loaded.")
    print(submission_df.head(3))

    # In debug mode (trainer.py), test set is subset to 50 samples
    expected_rows = 50
    assert (
        len(submission_df) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(submission_df)}"

    # Check columns: 'id' + 120 breeds
    expected_cols = 121
    assert (
        len(submission_df.columns) == expected_cols
    ), f"Expected {expected_cols} columns, got {len(submission_df.columns)}"
    assert submission_df.columns[0] == "id", "First column should be 'id'"

    # Verify probabilities sum to 1
    # Drop 'id' column
    probs = submission_df.iloc[:, 1:].values
    row_sums = np.sum(probs, axis=1)

    print(f"Mean Probability Sum: {np.mean(row_sums):.6f}")

    # Assert sums are close to 1.0 (tolerance for float precision)
    assert np.allclose(row_sums, 1.0, atol=1e-4), "Probabilities do not sum to 1.0"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
