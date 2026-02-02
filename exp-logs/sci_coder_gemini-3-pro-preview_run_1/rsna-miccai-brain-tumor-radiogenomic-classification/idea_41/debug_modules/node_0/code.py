import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import EfficientNetExpert
from library.train import run_expert_training
from library.inference import generate_submission


def print_step(msg):
    print(f"\n{'#'*60}")
    print(f"# {msg}")
    print(f"{'#'*60}")


if __name__ == "__main__":
    # 1. Setup & Configuration Override for Fast Demonstration
    print_step("1. Configuring Environment for Demo")

    # Set a specific working directory for this demo to avoid conflicts
    DEMO_WORK_DIR = "./working/demo_run"
    if os.path.exists(DEMO_WORK_DIR):
        shutil.rmtree(DEMO_WORK_DIR)
    os.makedirs(DEMO_WORK_DIR, exist_ok=True)

    # Override Config parameters for speed and resource constraints
    Config.WORK_DIR = DEMO_WORK_DIR
    Config.SUBMISSION_FILE = os.path.join(DEMO_WORK_DIR, "demo_submission.csv")

    # Enable Debug mode to use a tiny subset of data (e.g., 6 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 6

    # Reduce training complexity
    Config.BATCH_SIZE = 2
    Config.NUM_EPOCHS = 1
    Config.NUM_FOLDS = 2  # Run 2 folds to demonstrate CV logic
    Config.PATIENCE = 1

    # Only run one Expert (Center Plane) to save time
    Config.EXPERTS = {"Expert_B": 0}

    # Set seed for reproducibility
    set_seed(42)

    print(f"Working Directory: {Config.WORK_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # 2. Verify Data Loading Pipeline
    print_step("2. Verifying Data Loading Pipeline")

    # Get dataloaders for Fold 0 of Expert_B
    # load_cached_data=False forces re-computation of Centroids (good for testing logic)
    train_loader, val_loader = get_dataloaders(
        expert_name="Expert_B", fold_idx=0, load_cached_data=False
    )

    # Fetch one batch
    images, labels = next(iter(train_loader))

    print(f"Batch Images Shape: {images.shape}")
    print(f"Batch Labels Shape: {labels.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), f"Incorrect image shape. Expected {(Config.BATCH_SIZE, 3, 224, 224)}, got {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect label shape. Expected {(Config.BATCH_SIZE,)}, got {labels.shape}"
    assert images.dtype == torch.float32, "Images should be float32"
    assert labels.dtype == torch.float32, "Labels should be float32"

    # Check Normalization (Min-Max scaling should result in values between 0 and 1)
    assert (
        images.min() >= 0.0 and images.max() <= 1.0
    ), "Images are not properly normalized to [0, 1]"

    print("Data Loading Verification Passed.")

    # 3. Verify Model Architecture
    print_step("3. Verifying Model Architecture")

    model = EfficientNetExpert(
        pretrained=False
    )  # No need to download weights for shape check
    model.to(Config.DEVICE)

    # Forward pass with the batch from step 2
    with torch.no_grad():
        images = images.to(Config.DEVICE)
        logits = model(images)

    print(f"Output Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {logits.shape}"

    print("Model Architecture Verification Passed.")

    # 4. Execute Training Loop
    print_step("4. Executing Training Loop (Reduced Epochs/Folds)")

    # This runs the full training logic defined in library.train
    # It iterates over Config.EXPERTS and Config.NUM_FOLDS
    run_expert_training(load_cached_data=False)

    # Verify artifacts
    expected_model_path = os.path.join(Config.WORK_DIR, "best_model_Expert_B_fold0.pth")
    if not os.path.exists(expected_model_path):
        raise FileNotFoundError(
            f"Training failed to produce model file at {expected_model_path}"
        )

    print(f"Training completed successfully. Model saved at: {expected_model_path}")

    # 5. Execute Inference & Submission Generation
    print_step("5. Executing Inference and Submission Generation")

    # This runs the full inference logic defined in library.inference
    generate_submission(load_cached_data=False)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError(
            f"Inference failed to produce submission file at {Config.SUBMISSION_FILE}"
        )

    # Load and check content
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print("Submission File Head:")
    print(df_sub.head())

    # Assertions on Submission
    assert "BraTS21ID" in df_sub.columns, "Submission missing BraTS21ID column"
    assert "MGMT_value" in df_sub.columns, "Submission missing MGMT_value column"
    assert len(df_sub) > 0, "Submission file is empty"

    # Check probability range
    preds = df_sub["MGMT_value"].values
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions out of probability range [0, 1]"

    print("Inference Verification Passed.")

    print_step("Demo Completed Successfully")
