import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library
from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    INPUT_ROOT,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
)
from library.utils import set_seed
from library.dataset import get_dataloaders
from library.model import TimePreservingEfficientNet
from library.trainer import Trainer


def clean_working_directory():
    """
    Cleans up the working directory to ensure we generate fresh debug data
    instead of loading potentially existing full datasets from cache.
    """
    if os.path.exists(WORKING_DIR):
        # We only remove the .npy files to avoid deleting directories if not empty
        for f in os.listdir(WORKING_DIR):
            if f.endswith(".npy") or f.endswith(".pth"):
                os.remove(os.path.join(WORKING_DIR, f))
    else:
        os.makedirs(WORKING_DIR, exist_ok=True)


def demo_data_loading():
    """
    Demonstrates loading the data using the debug mode.
    Verifies the structure of the batches.
    """
    print("\n=== 1. Data Loading Demo ===")

    # Force debug=True to load only a small subset (100 samples)
    # load_cached_data=False ensures we process the raw audio files for this demo
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        debug=True, load_cached_data=False
    )

    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches: {len(val_loader)}")
    print(f"Test Loader Batches: {len(test_loader)}")

    # Fetch one batch to verify shapes
    data, target = next(iter(train_loader))

    # Expected shape: (Batch, 1, n_mels, time)
    # Note: The dataset implementation in library/dataset.py unsqueezes dim 0.
    # n_mels = 128 (from config)
    # time is variable or padded. The load_and_cache_data pads/crops to 100 frames.
    print(f"Input Batch Shape: {data.shape}")
    print(f"Target Batch Shape: {target.shape}")

    # Assertions
    assert data.dim() == 4, "Data should be 4D (Batch, Channel, Freq, Time)"
    assert data.size(1) == 1, "Channel dimension should be 1"
    assert data.size(2) == 128, "Frequency dimension (n_mels) should be 128"
    assert target.dim() == 1, "Target should be 1D"

    return train_loader, val_loader, test_loader, test_ids


def demo_model_architecture():
    """
    Demonstrates model instantiation and performs a forward pass
    with dummy data to verify output dimensions.
    """
    print("\n=== 2. Model Architecture Demo ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TimePreservingEfficientNet().to(device)
    model.eval()

    # Create a dummy input matching the expected shape: (Batch, 1, Freq, Time)
    # Batch=2, Channel=1, Freq=128, Time=100
    dummy_input = torch.randn(2, 1, 128, 100).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Dummy Input Shape: {dummy_input.shape}")
    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.dim() == 2, "Output should be 2D (Batch, Num_Classes)"
    assert output.size(0) == 2, "Output batch size should match input"
    assert output.size(1) == 1, "Output classes should be 1 (Binary)"

    print("Model forward pass successful.")
    return model


def demo_training_and_inference(train_loader, val_loader, test_loader, test_ids):
    """
    Demonstrates the training loop and inference using the Trainer class.
    Runs for 1 epoch to ensure speed.
    """
    print("\n=== 3. Training and Inference Demo ===")

    trainer = Trainer(train_loader, val_loader, test_loader, test_ids)

    # Run training for 1 epoch
    print("Starting training (1 epoch)...")
    trainer.fit(epochs=1)

    # Verify model checkpoint was saved
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "best_model.pth was not saved."
    print(f"Checkpoint verified at: {best_model_path}")

    # Run prediction
    print("Starting prediction...")
    trainer.predict()

    # Verify submission file
    assert os.path.exists(SUBMISSION_PATH), "submission.csv was not generated."
    print(f"Submission verified at: {SUBMISSION_PATH}")

    # Validate submission content
    df_sub = pd.read_csv(SUBMISSION_PATH)
    print("Submission Head:")
    print(df_sub.head())

    assert list(df_sub.columns) == [
        "clip",
        "probability",
    ], "Submission columns mismatch"
    assert len(df_sub) == len(test_ids), "Submission length mismatch"
    assert df_sub["probability"].dtype == float, "Probability column should be float"

    print("Training and Inference pipeline completed successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(42)

    # Clean up working directory to force fresh data processing for the demo
    clean_working_directory()

    # 1. Load Data
    train_loader, val_loader, test_loader, test_ids = demo_data_loading()

    # 2. Verify Model
    demo_model_architecture()

    # 3. Train and Predict
    demo_training_and_inference(train_loader, val_loader, test_loader, test_ids)

    print("\nAll demonstrations passed successfully.")
