import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed
from library.dataset import get_dataloaders, mixup_data
from library.model import TimePreservingEfficientNetBiGRU
from library.trainer import Trainer


def run_demonstration():
    print("=== Starting Whale Detection Pipeline Demonstration ===")

    # 1. Configure for Fast Demonstration
    print("\n[Step 1] Configuring environment for rapid execution...")

    # Override Config for speed and debugging
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 100  # Small subset for speed
    Config.EPOCHS = 1  # Single epoch
    Config.BATCH_SIZE = 16  # Small batch size
    Config.WORKING_DIR = "./working/demo_run"  # Isolated working directory
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure working directory exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration updated: DEBUG=True, EPOCHS=1, SAMPLES=100")

    # 2. Data Loading and Verification
    print("\n[Step 2] Initializing DataLoaders and verifying data shapes...")

    # Force reload to generate debug cache
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify Train Loader
    try:
        data_batch, label_batch = next(iter(train_loader))
        print(f"Train Batch Data Shape: {data_batch.shape}")
        print(f"Train Batch Label Shape: {label_batch.shape}")

        # Assertions
        assert data_batch.dim() == 4, "Data should be 4D (B, C, F, T)"
        assert data_batch.shape[1] == 1, "Should have 1 channel"
        assert (
            data_batch.shape[2] == Config.N_MELS
        ), f"Freq dim should be {Config.N_MELS}"
        # Time dim is approx 250 (4000 samples / 16 hop)
        assert (
            240 < data_batch.shape[3] < 260
        ), f"Time dim unexpected: {data_batch.shape[3]}"
        assert label_batch.dim() == 1, "Labels should be 1D"

        print("Data Loader verification passed.")
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # Verify Mixup
    print("Verifying Mixup Augmentation...")
    mixed_x, y_a, y_b, lam = mixup_data(
        data_batch, label_batch, alpha=0.4, device="cpu"
    )
    assert mixed_x.shape == data_batch.shape, "Mixup output shape mismatch"
    assert y_a.shape == label_batch.shape, "Mixup target A shape mismatch"
    print("Mixup verification passed.")

    # 3. Model Architecture Verification
    print("\n[Step 3] Instantiating Model and verifying forward pass...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimePreservingEfficientNetBiGRU().to(device)

    # Create dummy input based on actual batch shape
    dummy_input = torch.randn(2, 1, Config.N_MELS, data_batch.shape[3]).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (2, 1), "Model output should be (Batch, 1)"
    print("Model architecture verification passed.")

    # 4. Training Loop Execution
    print("\n[Step 4] Executing Training Loop (1 Epoch)...")

    trainer = Trainer()

    # Run training
    # This uses the modified Config.EPOCHS = 1
    trainer.train(train_loader, val_loader)

    # Verify artifacts
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not saved."
    print(f"Training complete. Checkpoint saved at {best_model_path}")

    # 5. Inference and Submission
    print("\n[Step 5] Generating Predictions on Test Set...")

    df_submission = trainer.predict(test_loader)

    # Verify Submission
    print(f"Submission shape: {df_submission.shape}")
    print(f"Submission columns: {df_submission.columns.tolist()}")

    expected_test_samples = min(Config.DEBUG_SAMPLES, len(pd.read_csv(Config.TEST_CSV)))
    assert (
        len(df_submission) == expected_test_samples
    ), f"Expected {expected_test_samples} predictions, got {len(df_submission)}"

    assert (
        "clip" in df_submission.columns and "probability" in df_submission.columns
    ), "Submission missing required columns"

    # Check probability range
    probs = df_submission["probability"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities out of range [0, 1]"

    print("Inference verification passed.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()
