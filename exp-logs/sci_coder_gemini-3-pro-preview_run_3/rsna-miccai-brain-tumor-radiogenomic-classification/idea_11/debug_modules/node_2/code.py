import os
import torch
import pandas as pd
import numpy as np
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import ModalityGroupedEfficientNet
from library.train import run_training
from library.inference import predict_submission


def main():
    # 1. Setup and Configuration
    seed_everything(42)
    device = get_device()
    print(f"Running demonstration on device: {device}")

    # Define parameters for a quick demonstration run
    DEMO_BATCH_SIZE = 4
    DEMO_EPOCHS = 1
    DEMO_DEBUG_LIMIT = 16  # Only use 16 samples to ensure speed
    SUBMISSION_PATH = "./working/demo_submission.csv"

    print("\n" + "=" * 40)
    print(" 1. Testing Data Loader & Shapes")
    print("=" * 40)

    # Initialize data loaders with debug limit
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=DEMO_BATCH_SIZE,
        load_cached_data=True,  # Will use existing cache if available in ./working/idea_11
        debug_limit=DEMO_DEBUG_LIMIT,
    )

    # Fetch one batch to verify shapes
    images, targets = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    # Assertions for Data Loader
    # Expected shape: (Batch, Channels=128, Height=256, Width=256)
    # Channels = 4 modalities * 32 slices = 128
    assert images.shape == (
        DEMO_BATCH_SIZE,
        128,
        256,
        256,
    ), f"Incorrect image shape. Expected {(DEMO_BATCH_SIZE, 128, 256, 256)}, got {images.shape}"
    assert targets.shape == (
        DEMO_BATCH_SIZE,
    ), f"Incorrect target shape. Expected {(DEMO_BATCH_SIZE,)}, got {targets.shape}"

    print("Data Loader verification successful.")

    print("\n" + "=" * 40)
    print(" 2. Testing Model Architecture")
    print("=" * 40)

    # Instantiate model
    model = ModalityGroupedEfficientNet()
    model.to(device)

    # Perform a forward pass with the batch fetched earlier
    images = images.to(device)
    with torch.no_grad():
        logits = model(images)

    print(f"Model Output Logits Shape: {logits.shape}")

    # Assertions for Model
    # Expected output: (Batch, 1)
    assert logits.shape == (
        DEMO_BATCH_SIZE,
        1,
    ), f"Incorrect output shape. Expected {(DEMO_BATCH_SIZE, 1)}, got {logits.shape}"

    print("Model architecture verification successful.")

    print("\n" + "=" * 40)
    print(" 3. Running Training Loop (Demo)")
    print("=" * 40)

    # Run training for 1 epoch on the subset
    best_auc = run_training(
        epochs=DEMO_EPOCHS,
        batch_size=DEMO_BATCH_SIZE,
        lr=1e-4,
        patience=1,
        load_cached_data=True,
        debug_limit=DEMO_DEBUG_LIMIT,
    )

    print(f"Training demo finished with Best AUC: {best_auc}")

    # Verify model checkpoint was saved
    model_path = "./working/idea_11/best_model.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    print("Training loop verification successful.")

    print("\n" + "=" * 40)
    print(" 4. Running Inference")
    print("=" * 40)

    # Generate submission using the trained model
    submission_df = predict_submission(
        model_path=model_path,
        output_path=SUBMISSION_PATH,
        batch_size=DEMO_BATCH_SIZE,
        load_cached_data=True,
        device=device,
        debug_limit=DEMO_DEBUG_LIMIT,
    )

    # Verify Submission File
    if not os.path.exists(SUBMISSION_PATH):
        raise FileNotFoundError(f"Submission file not found at {SUBMISSION_PATH}")

    # Verify Columns
    expected_cols = ["BraTS21ID", "MGMT_value"]
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(submission_df.columns)}"

    # Verify ID types (should be int as per sample_submission logic in inference.py)
    assert pd.api.types.is_integer_dtype(
        submission_df["BraTS21ID"]
    ), "BraTS21ID column should be integer type."

    # Verify Probability Range
    probs = submission_df["MGMT_value"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Predictions contain probabilities outside [0, 1] range."

    print("Inference verification successful.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
