import os
import sys
import torch
import numpy as np
import pandas as pd

# 1. Import Config and patch it for a fast demo run
from library.config import Config

# Define demo-specific paths and settings
DEMO_DIR = "./working/demo_execution"
os.makedirs(DEMO_DIR, exist_ok=True)

print(f"Setting up demo configuration in {DEMO_DIR}...")
Config.WORKING_DIR = DEMO_DIR
Config.CACHE_DIR = DEMO_DIR
Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pth")
Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")
Config.NUM_EPOCHS = 1
Config.BATCH_SIZE = 16
Config.DEBUG = True  # Encourages lighter processing if implemented in library

# 2. Import remaining library modules
# Note: Imports are done after patching Config so that if any default args
# are evaluated at import time, they might pick up changes (though explicit passing is safer).
from library.utils import seed_everything, get_device
from library.data import get_loaders
from library.model import GLiClassModel
from library.train import run_training
from library.inference import predict_and_submit

if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # ==========================================
    # Step 1: Data Loading & Verification
    # ==========================================
    print("\n=== Step 1: Initializing Data Loaders ===")
    # This will trigger data processing and caching to ./working/demo_execution
    train_loader, val_loader = get_loaders()

    print("Verifying Train Loader...")
    # Fetch one batch to verify shapes and content
    images, targets, ids = next(iter(train_loader))

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), f"Incorrect image shape: {images.shape}. Expected ({Config.BATCH_SIZE}, 3, 224, 224)"
    assert targets.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect target shape: {targets.shape}. Expected ({Config.BATCH_SIZE},)"
    assert ids.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect ID shape: {ids.shape}. Expected ({Config.BATCH_SIZE},)"

    # Check Normalization (MinMax [0, 1])
    assert (
        images.min() >= 0.0 and images.max() <= 1.0
    ), "Images do not appear to be normalized to [0, 1]"

    print("Data Loader verification successful.")

    # ==========================================
    # Step 2: Model Initialization & Forward Pass
    # ==========================================
    print("\n=== Step 2: Model Instantiation ===")
    model = GLiClassModel(
        backbone=Config.BACKBONE,
        pretrained=Config.PRETRAINED,
        num_classes=Config.NUM_CLASSES,
        in_chans=Config.IN_CHANNELS,
    )
    model = model.to(device)

    print("Verifying Forward Pass...")
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"

    print("Model verification successful.")

    # ==========================================
    # Step 3: Training Loop
    # ==========================================
    print("\n=== Step 3: Running Training (1 Epoch) ===")
    # Explicitly pass num_epochs to ensure the patched value is used
    run_training(num_epochs=Config.NUM_EPOCHS)

    # Verify model checkpoint was created
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(
            f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"
        )

    print(f"Training complete. Model saved to {Config.MODEL_SAVE_PATH}")

    # ==========================================
    # Step 4: Inference & Submission
    # ==========================================
    print("\n=== Step 4: Inference and Submission ===")
    # Run inference on the test set
    predict_and_submit(
        model_path=Config.MODEL_SAVE_PATH, output_path=Config.SUBMISSION_PATH
    )

    # Verify Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check Columns
    required_cols = ["BraTS21ID", "MGMT_value"]
    for col in required_cols:
        assert col in df_sub.columns, f"Submission missing column: {col}"

    # Check Values
    if not df_sub.empty:
        probs = df_sub["MGMT_value"]
        assert (
            probs.min() >= 0.0 and probs.max() <= 1.0
        ), "Predicted probabilities are out of range [0, 1]"

        # Check ID format (should be integers as per sample_submission, or at least numeric)
        assert pd.api.types.is_numeric_dtype(
            df_sub["BraTS21ID"]
        ), "BraTS21ID column should be numeric"

    print(f"Submission verification successful. Rows: {len(df_sub)}")
    print("\n=== Demo Execution Completed Successfully ===")
