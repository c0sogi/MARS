import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders, HotelDataset, get_transforms
from library.model import HotelRecognitionModel
from library.engine import train_model, generate_submission


def run_demo():
    print("=== Starting Hotel ID Task Demonstration ===")

    # --------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed/Demo
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Enable Debug mode to use a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Use 100 images for train/val/test

    # Reduce training duration
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 8

    # Reduce workers to avoid overhead on small data
    Config.NUM_WORKERS = 0

    # Ensure clean slate for working directory
    # We remove the specific model file if it exists to verify saving works
    if os.path.exists(Config.MODEL_SAVE_PATH):
        try:
            os.remove(Config.MODEL_SAVE_PATH)
        except OSError:
            pass

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print(f"Batch Size: {Config.BATCH_SIZE}")

    # --------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # --------------------------------------------------------------------------
    print("\n[2] Verifying Data Pipeline...")

    # Force load_cached_data=False to ensure we generate encoding for our debug subset
    # This ensures the label encoder matches the random subset sampled in this run
    train_loader, val_loader, test_loader, idx_to_hotel = get_dataloaders(
        load_cached_data=False
    )

    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches: {len(val_loader)}")

    # Fetch one batch to verify shapes
    images, labels = next(iter(train_loader))

    print(f"Sample Batch - Images Shape: {images.shape}")
    print(f"Sample Batch - Labels Shape: {labels.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect Image Shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect Label Shape"
    assert images.dtype == torch.float32, "Images should be float32"
    assert labels.dtype == torch.long, "Labels should be long (int64)"

    print("Data Pipeline verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Model Logic Verification
    # --------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = HotelRecognitionModel(n_classes=Config.NUM_CLASSES)
    model.to(device)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(device)

    # Test Inference Mode (No labels) -> Expect Embeddings
    with torch.no_grad():
        embeddings = model(dummy_input, labels=None)

    print(f"Embeddings Shape: {embeddings.shape}")
    assert embeddings.shape == (2, Config.EMBEDDING_DIM), "Embedding shape mismatch"

    # Test Training Mode (With labels) -> Expect Logits
    dummy_labels = torch.tensor([0, 1]).to(device)
    logits = model(dummy_input, labels=dummy_labels)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (2, Config.NUM_CLASSES), "Logits shape mismatch"

    print("Model logic verified successfully.")

    # --------------------------------------------------------------------------
    # 4. Training Loop Execution
    # --------------------------------------------------------------------------
    print("\n[4] Executing Training Loop (Engine)...")

    # This will run for 1 epoch on the debug dataset
    # It handles model init, optimizer, loop, and saving
    train_model()

    # Verify model was saved
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved!"
    print(f"Model successfully saved to {Config.MODEL_SAVE_PATH}")

    # --------------------------------------------------------------------------
    # 5. Inference and Submission
    # --------------------------------------------------------------------------
    print("\n[5] Generating Submission (Engine)...")

    # This loads the saved model and predicts on the test set (debug subset)
    generate_submission()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not generated!"

    # Check content format
    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission Shape: {sub_df.shape}")
    print("Submission Head:")
    print(sub_df.head(3))

    # Assertions on submission
    assert (
        "image" in sub_df.columns and "hotel_id" in sub_df.columns
    ), "Missing columns in submission"
    assert len(sub_df) > 0, "Submission is empty"

    # Check prediction format (space delimited)
    example_pred = sub_df.iloc[0]["hotel_id"]
    assert isinstance(example_pred, str), "Prediction should be a string"
    # Ensure we have TOP_K predictions
    assert (
        len(example_pred.split()) == Config.TOP_K
    ), f"Prediction should contain {Config.TOP_K} IDs"

    print("Submission verified successfully.")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()
