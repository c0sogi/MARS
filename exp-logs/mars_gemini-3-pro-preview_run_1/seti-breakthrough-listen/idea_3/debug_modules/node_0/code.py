import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config, seed_everything
from library.data import get_dataloaders
from library.model import SpatiotemporalResNet
from library.utils import calculate_roc_auc, AverageMeter
from library.train import train_model
from library.inference import predict


def run_demonstration():
    print("=== Starting Library Demonstration ===")

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # 1. Verify Configuration and Paths
    print("\n[1] Verifying Configuration...")
    assert os.path.exists(Config.INPUT_DIR), "Input directory not found."
    assert os.path.exists(Config.METADATA_DIR), "Metadata directory not found."
    # Ensure working directory is clean for this run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    print("Configuration verified.")

    # 2. Demonstrate Utility Functions
    print("\n[2] Testing Utility Functions...")
    # Test ROC AUC calculation
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    auc = calculate_roc_auc(y_true, y_pred)
    print(f"Calculated AUC: {auc}")
    assert 0 <= auc <= 1, "AUC score out of range."

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(val=10, n=2)
    meter.update(val=20, n=2)
    assert meter.avg == 15.0, f"AverageMeter failed. Expected 15.0, got {meter.avg}"
    print("Utility functions verified.")

    # 3. Demonstrate Data Loading
    print("\n[3] Testing Data Loading (Debug Mode)...")
    # Use a small batch size for demonstration
    demo_batch_size = 4
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True,
        batch_size=demo_batch_size,
        num_workers=2,  # Reduce workers for small demo
        debug_subset_size=100,  # Very small subset for speed
    )

    # Fetch one batch
    inputs, targets = next(iter(train_loader))

    print(f"Input batch shape: {inputs.shape}")
    print(f"Target batch shape: {targets.shape}")

    # Verify shapes: (Batch, Channels, Depth, Height, Width) -> (B, 1, 6, 273, 256)
    assert inputs.shape == (demo_batch_size, 1, 6, 273, 256), "Incorrect input shape."
    assert targets.shape == (demo_batch_size,), "Incorrect target shape."
    assert inputs.dtype == torch.float32, "Input tensor should be float32."
    print("Data loading verified.")

    # 4. Demonstrate Model Architecture
    print("\n[4] Testing Model Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpatiotemporalResNet(pretrained=False).to(device)

    # Move demo batch to device
    inputs = inputs.to(device)

    # Perform forward pass
    with torch.no_grad():
        outputs = model(inputs)

    print(f"Model output shape: {outputs.shape}")

    # Verify output shape: (Batch, 1) - logits
    assert outputs.shape == (demo_batch_size, 1), "Incorrect model output shape."
    print("Model architecture verified.")

    # 5. Demonstrate Training Loop
    print("\n[5] Running Training Loop (1 Epoch, Debug Mode)...")
    # We run train_model which handles the full loop including saving checkpoints.
    # We use debug=True to use the subset defined in Config (or overridden internally if modified).
    # Note: train_model uses Config.DEBUG_SUBSET_SIZE (2000) by default when debug=True.
    # We pass epochs=1 to ensure it finishes quickly.

    best_auc = train_model(debug=True, epochs=1)

    print(f"Training finished with Best AUC: {best_auc}")

    # Verify checkpoint creation
    checkpoint_path = Config.MODEL_SAVE_PATH
    assert os.path.exists(
        checkpoint_path
    ), f"Model checkpoint not found at {checkpoint_path}"
    print("Training loop and checkpointing verified.")

    # 6. Demonstrate Inference
    print("\n[6] Running Inference...")
    # Run prediction using the checkpoint generated in step 5
    predict(debug=True)

    # Verify submission file
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    # Load and check submission content
    sub_df = pd.read_csv(submission_path)
    print(f"Submission shape: {sub_df.shape}")
    print("First 3 rows:")
    print(sub_df.head(3))

    assert list(sub_df.columns) == ["id", "target"], "Submission columns mismatch."
    assert len(sub_df) > 0, "Submission file is empty."
    # In debug mode, get_dataloaders usually returns the full test set or a subset depending on implementation.
    # The provided library data.py loads the full test set metadata even in debug mode unless explicitly sliced.
    # However, let's just ensure it's not empty.

    print("Inference verified.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
