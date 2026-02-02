import os
import sys
import torch
import numpy as np
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.utils import AverageMeter, calculate_auc
from library.dataset import get_dataloaders
from library.model import CactusResNet
from library.train import run_training, run_inference


def main():
    print("=== Starting Cactus Identification Pipeline Demonstration ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid execution...")
    # Modify Config for speed: 1 epoch, small debug subset
    Config.NUM_EPOCHS = 1
    Config.DEBUG_SUBSET_SIZE = 100
    Config.BATCH_SIZE = 16
    Config.DEBUG = True  # Set default debug flag

    # Create necessary directories (working, submission)
    Config.create_directories()
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Submission Directory: {Config.SUBMISSION_DIR}")
    print("Configuration updated.")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n[2] Verifying utility functions...")

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(val=5.0, n=1)
    meter.update(val=15.0, n=1)
    assert (
        meter.avg == 10.0
    ), f"AverageMeter logic error: expected 10.0, got {meter.avg}"
    print("AverageMeter verified.")

    # Test calculate_auc
    y_true_dummy = np.array([0, 1, 0, 1])
    y_pred_dummy = np.array([0.1, 0.9, 0.2, 0.8])
    auc_score = calculate_auc(y_true_dummy, y_pred_dummy)
    assert 0.9 <= auc_score <= 1.0, f"calculate_auc logic error: got {auc_score}"
    print(f"calculate_auc verified (Score: {auc_score}).")

    # -------------------------------------------------------------------------
    # 3. Verify Data Loading & Dataset
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Data Loading...")
    # Use debug=True to load only the subset
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, batch_size=Config.BATCH_SIZE
    )

    # Fetch a single batch
    images, labels = next(iter(train_loader))
    print(f"Batch shapes - Images: {images.shape}, Labels: {labels.shape}")

    # Assertions
    assert images.shape == (Config.BATCH_SIZE, 3, 32, 32), "Image batch shape mismatch"
    assert labels.shape == (Config.BATCH_SIZE,), "Label batch shape mismatch"
    assert isinstance(images, torch.Tensor), "Images should be a Tensor"
    assert isinstance(labels, torch.Tensor), "Labels should be a Tensor"
    print("DataLoaders verified.")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")
    model = CactusResNet()

    # Move model to configured device (likely CPU or CUDA)
    model = model.to(Config.DEVICE)
    dummy_input = images.to(Config.DEVICE)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model output shape: {output.shape}")
    assert output.shape == (Config.BATCH_SIZE, 1), "Model output shape mismatch"
    print("Model architecture verified.")

    # -------------------------------------------------------------------------
    # 5. Execute Training Pipeline
    # -------------------------------------------------------------------------
    print("\n[5] Executing Training Pipeline (Debug Mode)...")
    # We run training with debug=True to use the small subset and finish quickly
    run_training(num_epochs=Config.NUM_EPOCHS, debug=True)

    # Verify checkpoint creation
    if not os.path.exists(Config.OUTPUT_MODEL_PATH):
        raise FileNotFoundError(
            f"Training failed to create model checkpoint at {Config.OUTPUT_MODEL_PATH}"
        )
    print("Training completed successfully. Checkpoint saved.")

    # -------------------------------------------------------------------------
    # 6. Execute Inference Pipeline
    # -------------------------------------------------------------------------
    print("\n[6] Executing Inference Pipeline...")
    # NOTE: We run inference with debug=False because the library's predict_and_submit
    # function expects the full metadata ID list to match the loader size.
    # Inference on ~3300 images is fast enough (< 1 min).
    run_inference(debug=False)

    # Verify submission file creation
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Inference failed to create submission file at {Config.SUBMISSION_PATH}"
        )
    print("Inference completed successfully.")

    # -------------------------------------------------------------------------
    # 7. Validate Submission Format
    # -------------------------------------------------------------------------
    print("\n[7] Validating Submission Format...")
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    print(df_sub.head(3))

    # Check columns
    expected_cols = ["id", "has_cactus"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Check values range
    preds = df_sub["has_cactus"]
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions out of probability range [0, 1]"

    # Check ID count
    assert len(df_sub) > 0, "Submission file is empty"

    print("Submission format validated.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
