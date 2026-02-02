import os
import shutil
import torch
import pandas as pd
import numpy as np

# Import provided library modules
from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model import WhaleEfficientNet
from library.train import run_training
from library.predict import generate_predictions


def main():
    print("=== 1. Setup and Configuration ===")
    # Override Config for a fast, isolated demo run
    Config.PROJECT_NAME = "demo_run"
    Config.WORKING_DIR = os.path.join("./working", Config.PROJECT_NAME)

    # Clean up any previous demo run to ensure reproducibility
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Redirect paths to the demo working directory
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.TRAIN_DATA_CACHE = os.path.join(Config.WORKING_DIR, "train_data.npy")
    Config.TRAIN_LABELS_CACHE = os.path.join(Config.WORKING_DIR, "train_labels.npy")
    Config.VAL_DATA_CACHE = os.path.join(Config.WORKING_DIR, "val_data.npy")
    Config.VAL_LABELS_CACHE = os.path.join(Config.WORKING_DIR, "val_labels.npy")
    Config.TEST_DATA_CACHE = os.path.join(Config.WORKING_DIR, "test_data.npy")
    Config.TEST_CLIPS_CACHE = os.path.join(Config.WORKING_DIR, "test_clips.npy")
    Config.SUBMISSION_DIR = Config.WORKING_DIR
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Set Hyperparameters for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 20  # Use only 20 samples per split
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.PRETRAINED = False  # Skip downloading weights
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.PATIENCE = 1

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration updated for demo mode.")

    print("\n=== 2. Verify Data Loading and Preprocessing ===")
    # Load data (this will process audio files and save to cache)
    # load_cached_data=False forces processing from scratch
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Check if loaders have data
    assert len(train_loader) > 0, "Train loader is empty."
    assert len(val_loader) > 0, "Validation loader is empty."
    assert len(test_loader) > 0, "Test loader is empty."

    # Fetch one batch to verify shapes
    images, labels = next(iter(train_loader))

    print(f"Batch Images Shape: {images.shape}")
    print(f"Batch Labels Shape: {labels.shape}")

    # Verify Image Dimensions: (Batch, Channels, Freq, Time)
    # Channels should be 3 (Log-Mel, Delta, Delta-Delta)
    # Freq should be N_MELS (128)
    assert images.shape[0] == Config.BATCH_SIZE
    assert images.shape[1] == 3
    assert images.shape[2] == Config.N_MELS
    # Time dimension depends on SR, Duration, and Hop Length
    # 2000Hz * 2s = 4000 samples. 4000 / 64 (hop) approx 63 frames.
    assert images.shape[3] > 0

    print("Data loading verification passed.")

    print("\n=== 3. Verify Model Architecture ===")
    # Initialize model
    model = WhaleEfficientNet(pretrained=False)
    model.eval()

    # Run forward pass on the batch fetched earlier
    with torch.no_grad():
        outputs = model(images)

    print(f"Model Output Shape: {outputs.shape}")

    # Verify Output
    assert outputs.shape == (Config.BATCH_SIZE, 1), "Output shape mismatch."
    assert not torch.isnan(outputs).any(), "Model produced NaN values."

    print("Model architecture verification passed.")

    print("\n=== 4. Run Training Pipeline ===")
    # Execute the training loop
    # This will use the cached data generated in step 2
    run_training()

    # Verify that the model checkpoint was saved
    assert os.path.exists(
        Config.MODEL_PATH
    ), f"Model checkpoint not found at {Config.MODEL_PATH}"
    print("Training pipeline finished successfully.")

    print("\n=== 5. Run Prediction Pipeline ===")
    # Generate predictions on the test set
    submission_df = generate_predictions(load_cached_data=True)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission CSV not found."

    # Verify Content
    print(f"Submission Shape: {submission_df.shape}")
    print(f"Columns: {submission_df.columns.tolist()}")

    assert "clip" in submission_df.columns
    assert "probability" in submission_df.columns
    # Should match DEBUG_SAMPLES (20)
    assert (
        len(submission_df) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} predictions, got {len(submission_df)}"

    print("Prediction pipeline finished successfully.")
    print("\n=== All Tasks Completed ===")


if __name__ == "__main__":
    main()
