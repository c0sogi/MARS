import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader

# Monkey-patch tqdm to suppress progress bars from library files
# This must be done before importing library modules that use tqdm
import tqdm


def silent_tqdm(iterable, *args, **kwargs):
    return iterable


tqdm.tqdm = silent_tqdm

# Import provided library modules
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    SEED,
    NUM_CLASSES,
    WINDOW_SIZE,
)
from library.dataset import GestureDataset
from library.model import NRGSNet
from library.trainer import CustomLoss, train_epoch, validate_epoch
from library.inference import generate_submission


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup & Reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    demo_model_path = os.path.join(WORKING_DIR, "demo_model.pth")
    demo_submission_path = os.path.join(WORKING_DIR, "demo_submission.csv")

    # 2. Data Loading Verification
    print("\n[1/4] Verifying Data Loading...")

    # Load a small subset of training data
    # limit=20 ensures we process enough files to get valid windows
    train_dataset = GestureDataset(
        TRAIN_METADATA_PATH, is_train=True, load_cached_data=True, limit=20
    )

    if len(train_dataset) == 0:
        print(
            "Warning: Dataset is empty (possibly due to missing files in demo env). Skipping assertions."
        )
    else:
        print(f"Dataset size (windows): {len(train_dataset)}")

        # Fetch one sample
        features, labels = train_dataset[0]

        # Verify shapes
        # Expected Feature Dim: (20 joints * 9 kinematics) + 13 MFCC = 193
        expected_dim = 193

        print(f"Feature shape: {features.shape}")
        print(f"Label shape: {labels.shape}")

        assert features.dim() == 2, "Features should be (Time, Dim)"
        assert (
            features.shape[0] == WINDOW_SIZE
        ), f"Time dimension should be {WINDOW_SIZE}"
        assert (
            features.shape[1] == expected_dim
        ), f"Feature dimension should be {expected_dim}, got {features.shape[1]}"
        assert labels.shape[0] == WINDOW_SIZE, "Labels time dimension mismatch"

        print("Data loading verification passed.")

    # 3. Model Architecture Verification
    print("\n[2/4] Verifying Model Architecture...")

    model = NRGSNet().to(device)

    # Create dummy input: (Batch=2, Time=64, Dim=193)
    dummy_input = torch.randn(2, WINDOW_SIZE, 193).to(device)

    # Forward pass
    outputs = model(dummy_input)

    # Model returns tuple of 3 outputs (Deep Supervision)
    assert len(outputs) == 3, "Model should return 3 outputs for deep supervision"

    out_1, out_2, out_3 = outputs
    print(f"Output shapes: {out_1.shape}, {out_2.shape}, {out_3.shape}")

    # Verify shape: (Batch, Time, Num_Classes)
    expected_shape = (2, WINDOW_SIZE, NUM_CLASSES)
    assert out_1.shape == expected_shape
    assert out_2.shape == expected_shape
    assert out_3.shape == expected_shape

    print("Model architecture verification passed.")

    # 4. Training Loop Demonstration
    print("\n[3/4] Demonstrating Training Loop...")

    # Create DataLoaders for the subset
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Initialize Loss and Optimizer
    criterion = CustomLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Run 1 Training Epoch
    print("Running training epoch...")
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, device
    )
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # Run 1 Validation Epoch (using train dataset as validation for demo purposes)
    print("Running validation epoch...")
    val_loss, val_acc = validate_epoch(model, train_loader, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Save the model
    torch.save(model.state_dict(), demo_model_path)
    print(f"Model saved to {demo_model_path}")

    assert os.path.exists(demo_model_path), "Model file was not created"
    print("Training loop demonstration passed.")

    # 5. Inference Pipeline Demonstration
    print("\n[4/4] Demonstrating Inference Pipeline...")

    # Run inference on a small subset of test data
    # limit=5 to ensure speed
    try:
        generate_submission(
            model_path=demo_model_path,
            output_file=demo_submission_path,
            load_cached_data=True,
            limit=5,
        )

        # Verify submission file
        if os.path.exists(demo_submission_path):
            # Use standard I/O instead of pandas for ragged CSVs
            with open(demo_submission_path, "r") as f:
                lines = f.readlines()

            print(f"Submission generated with {len(lines)} rows.")

            # Check content format (SampleID, predictions)
            # Example line: Sample00300,1,2,3
            if len(lines) > 0:
                first_line = lines[0].strip()
                print(f"First line of submission: {first_line}")

            assert len(lines) > 0, "Submission file is empty"
            print("Inference pipeline demonstration passed.")
        else:
            print(
                "Warning: Submission file not created (possibly due to empty test subset)."
            )

    except Exception as e:
        print(f"Inference failed with error: {e}")
        # If test data is missing in the environment, we catch it here
        # but in a real run, this should work.
        raise e

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
