import os
import shutil
import torch
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed
from library.data_loader import get_dataloaders
from library.model import DCHN
from library.train_eval import Trainer


def run_demonstration():
    print("Initializing Demonstration...")

    # 1. Configuration Override for Speed and Demonstration
    # We modify the Config class directly to use a temporary working directory
    # and limit the compute time (fewer epochs, smaller batch size, debug subset).
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.PROCESSED_DATA_PATH = os.path.join(Config.WORKING_DIR, "demo_data.npz")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.DEBUG_SUBSET_SIZE = 20  # Only use 20 samples for this demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Run setup to create directories and set seeds
    Config.setup()
    set_seed(Config.SEED)

    print(f"\n[Step 1] Configuration set up. Working dir: {Config.WORKING_DIR}")

    # 2. Data Loading Demonstration
    print("\n[Step 2] Loading Data (Debug Mode)...")
    # We force processing from scratch to ensure the pipeline works,
    # but use debug=True to slice the datasets to DEBUG_SUBSET_SIZE.
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, batch_size=Config.BATCH_SIZE, debug=True
    )

    # Validation: Check DataLoaders
    print("Validating DataLoaders...")
    assert len(train_loader) > 0, "Train loader should not be empty"
    assert len(val_loader) > 0, "Val loader should not be empty"
    assert len(test_loader) > 0, "Test loader should not be empty"

    # Fetch a single batch to verify shapes
    sample_batch = next(iter(train_loader))
    images = sample_batch["image"]
    inc_angles = sample_batch["inc_angle"]
    labels = sample_batch["label"]

    print(f"  Batch Image Shape: {images.shape}")
    print(f"  Batch Angle Shape: {inc_angles.shape}")

    # Expected: (Batch, 3, 75, 75)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        75,
        75,
    ), f"Expected image shape {(Config.BATCH_SIZE, 3, 75, 75)}, got {images.shape}"
    # Expected: (Batch,)
    assert inc_angles.shape == (
        Config.BATCH_SIZE,
    ), f"Expected angle shape {(Config.BATCH_SIZE,)}, got {inc_angles.shape}"

    print("Data Loading verified successfully.")

    # 3. Model Architecture Demonstration
    print("\n[Step 3] Verifying Model Architecture...")
    model = DCHN()
    model.eval()  # Set to eval mode for deterministic check

    # Move inputs to CPU (default for this demo check)
    dummy_img = images.clone()
    dummy_angle = inc_angles.clone()

    with torch.no_grad():
        output = model(dummy_img, dummy_angle)

    print(f"  Model Output Shape: {output.shape}")

    # Expected: (Batch, 1) because NUM_CLASSES=1 and it's binary classification
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected output shape {(Config.BATCH_SIZE, 1)}, got {output.shape}"

    # Check value range (Sigmoid output should be between 0 and 1)
    assert (
        output.min() >= 0.0 and output.max() <= 1.0
    ), "Model outputs should be probabilities between 0 and 1"

    print("Model architecture verified successfully.")

    # 4. Training Loop Demonstration
    print("\n[Step 4] Running Training Loop (Trainer)...")
    trainer = Trainer()

    # Run fit (Training + Validation)
    # This will run for Config.NUM_EPOCHS (2) on the small subset
    trainer.fit(train_loader, val_loader)

    # Verify model checkpoint creation
    assert os.path.exists(
        Config.MODEL_PATH
    ), f"Model checkpoint not found at {Config.MODEL_PATH}"

    print(f"Training finished. Model saved to {Config.MODEL_PATH}")

    # 5. Inference/Prediction Demonstration
    print("\n[Step 5] Running Prediction...")
    trainer.predict(test_loader)

    # Verify submission file creation
    assert os.path.exists(
        Config.SUBMISSION_FILE
    ), f"Submission file not found at {Config.SUBMISSION_FILE}"

    # Verify submission content format
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission file loaded. Shape: {df_sub.shape}")
    print(df_sub.head())

    expected_cols = ["id", "is_iceberg"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Expected columns {expected_cols}, got {list(df_sub.columns)}"

    # Check if probabilities are valid
    assert (
        df_sub["is_iceberg"].between(0, 1).all()
    ), "Submission contains probabilities outside [0, 1]"

    # Check if IDs match the debug subset size (or close to it depending on batching)
    # Since we used debug=True, the test set is also truncated.
    assert (
        len(df_sub) <= Config.DEBUG_SUBSET_SIZE
    ), f"Submission length {len(df_sub)} exceeds debug limit {Config.DEBUG_SUBSET_SIZE}"

    print("Prediction pipeline verified successfully.")
    print("\nDemonstration Complete. All systems operational.")


if __name__ == "__main__":
    run_demonstration()
