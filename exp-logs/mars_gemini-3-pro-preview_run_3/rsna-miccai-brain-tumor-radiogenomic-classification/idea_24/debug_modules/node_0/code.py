import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config, seed_everything
from library.data_loader import get_dataloaders
from library.model import RMSHDNet
from library.train import run_training
from library.predict import generate_submission


def main():
    print("Starting Library Usage Demonstration...")

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    # Override Config for a fast, self-contained demonstration
    print("\n[1] Configuring environment for demo...")

    # Enable Debug mode to process only a small subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 6  # Process only 6 subjects per split

    # Set training parameters for speed
    Config.BATCH_SIZE = 2
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # Set paths to a temporary demo directory to avoid conflicts
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    Config.CACHE_DIR = demo_dir
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "best_model.pth")

    # Seed for reproducibility
    seed_everything(42)
    print("Configuration updated for speed and isolation.")

    # ==========================================
    # 2. Data Loader Verification
    # ==========================================
    print("\n[2] Verifying Data Loading...")

    # Generate dataloaders (force processing by setting load_cached_data=False initially)
    # Note: In a real run, we'd use True, but here we want to verify the processing logic.
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=False
    )

    # Verify Train Loader
    print(f"Train Loader batches: {len(train_loader)}")
    assert len(train_loader) > 0, "Train loader is empty!"

    # Fetch one batch
    images, targets = next(iter(train_loader))

    # Verify Shapes
    # Expected: (Batch, Channels=128, Height=224, Width=224)
    print(f"Input Batch Shape: {images.shape}")
    print(f"Target Batch Shape: {targets.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        128,
        224,
        224,
    ), f"Incorrect input shape. Expected {(Config.BATCH_SIZE, 128, 224, 224)}, got {images.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect target shape. Expected {(Config.BATCH_SIZE,)}, got {targets.shape}"

    print("Data Loader verification passed.")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n[3] Verifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RMSHDNet().to(device)

    # Move sample batch to device
    images = images.to(device)

    # Forward pass
    logits = model(images)

    print(f"Output Logits Shape: {logits.shape}")

    # Verify Output Shape: (Batch, 1)
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Incorrect model output shape. Expected {(Config.BATCH_SIZE, 1)}, got {logits.shape}"

    print("Model architecture verification passed.")

    # ==========================================
    # 4. Training Pipeline Execution
    # ==========================================
    print("\n[4] Executing Training Pipeline...")

    # Run training (uses the modified Config)
    # This will train for 1 epoch on the debug subset
    run_training(epochs=Config.EPOCHS)

    # Verify model was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model file was not saved at {Config.MODEL_SAVE_PATH}"

    print("Training pipeline executed successfully.")

    # ==========================================
    # 5. Inference & Submission
    # ==========================================
    print("\n[5] Generating Submission...")

    # Run inference
    # This uses the saved model and the test cache (which was generated during run_training's data setup)
    generate_submission()

    # Verify submission file
    submission_path = "./submission/submission.csv"
    assert os.path.exists(submission_path), "Submission file not found!"

    df_sub = pd.read_csv(submission_path)
    print("Submission File Head:")
    print(df_sub.head())

    # Verify submission content
    assert (
        "BraTS21ID" in df_sub.columns and "MGMT_value" in df_sub.columns
    ), "Submission file missing required columns."

    # Since we used DEBUG_SAMPLE_SIZE=6, the submission should have roughly 6 rows
    # (or fewer if the test set is smaller, but here test set is 59 so it will be clipped to 6)
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} predictions, found {len(df_sub)}"

    # Verify probability range
    assert (
        df_sub["MGMT_value"].min() >= 0.0 and df_sub["MGMT_value"].max() <= 1.0
    ), "Predictions are out of probability range [0, 1]"

    print("Inference and submission generation passed.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
