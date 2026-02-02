import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data import process_eeg_signal, process_spectrogram
from library.model import BandAdaptiveNet
from library.train import run_training
from library.inference import predict


def demo_pipeline():
    print("Initializing Demo Pipeline...")

    # ==========================================
    # 1. Configuration Overrides for Demo
    # ==========================================
    print("\n[Step 1] Configuring environment for fast execution...")

    # Set a specific working directory for this demo
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config values
    Config.WORKING_DIR = demo_dir
    Config.DEBUG = True  # Use small subset of data
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 2  # Minimal workers
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")

    # Update cache paths to use the demo directory
    Config.CACHE_FILES = {
        k: os.path.join(demo_dir, os.path.basename(v))
        for k, v in Config.CACHE_FILES.items()
    }

    seed_everything(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")

    # ==========================================
    # 2. Verify Data Processing Logic
    # ==========================================
    print("\n[Step 2] Verifying Data Processing Functions...")

    # Mock Raw Data
    # EEG: (Samples, Channels) -> (10000, 19)
    mock_eeg_raw = np.random.randn(10000, 19).astype(np.float32)
    # Spec: (Time, Freq) -> (400, 400) - approximation of raw input
    mock_spec_raw = np.random.randn(400, 400).astype(np.float32)

    # Test EEG Processing
    # Expected Output: (57, 128, 256) -> (19 channels * 3 bands, H, W)
    processed_eeg = process_eeg_signal(mock_eeg_raw, output_size=Config.IMG_SIZE_A)
    print(f"Processed EEG shape: {processed_eeg.shape}")

    if processed_eeg.shape != (57, 128, 256):
        raise AssertionError(
            f"EEG Processing failed. Expected (57, 128, 256), got {processed_eeg.shape}"
        )

    # Test Spectrogram Processing
    # Expected Output: (4, 256, 256) -> (4 regions, H, W)
    processed_spec = process_spectrogram(mock_spec_raw, output_size=Config.IMG_SIZE_B)
    print(f"Processed Spec shape: {processed_spec.shape}")

    if processed_spec.shape != (4, 256, 256):
        raise AssertionError(
            f"Spectrogram Processing failed. Expected (4, 256, 256), got {processed_spec.shape}"
        )

    print("Data processing logic verified.")

    # ==========================================
    # 3. Verify Model Architecture
    # ==========================================
    print("\n[Step 3] Verifying Model Architecture...")

    model = BandAdaptiveNet()
    model.eval()

    # Create dummy batch
    # Batch size = 2
    dummy_eeg = torch.randn(2, 57, 128, 256)
    dummy_spec = torch.randn(2, 4, 256, 256)

    with torch.no_grad():
        output = model(dummy_eeg, dummy_spec)

    print(f"Model Output shape: {output.shape}")

    if output.shape != (2, 6):
        raise AssertionError(
            f"Model output shape mismatch. Expected (2, 6), got {output.shape}"
        )

    if torch.isnan(output).any():
        raise AssertionError("Model produced NaN values.")

    print("Model architecture verified.")

    # ==========================================
    # 4. Run Training Loop (Integration)
    # ==========================================
    print("\n[Step 4] Running Training Loop (Debug Mode)...")

    # This will load data (cached to demo_dir), create datasets, and train for 1 epoch
    run_training(debug=True)

    # Verify artifacts
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Training failed to save model at {model_path}")

    print(f"Training completed successfully. Model saved at {model_path}")

    # ==========================================
    # 5. Run Inference (Integration)
    # ==========================================
    print("\n[Step 5] Running Inference...")

    # This will load the best model and predict on the test set (truncated)
    submission_df = predict(debug=True)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Inference failed to save submission at {Config.SUBMISSION_PATH}"
        )

    # Verify columns
    expected_cols = [
        "eeg_id",
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    if list(submission_df.columns) != expected_cols:
        raise AssertionError(
            f"Submission columns mismatch.\nExpected: {expected_cols}\nGot: {list(submission_df.columns)}"
        )

    # Verify values (probabilities sum to ~1)
    # We check the first row
    row_sum = submission_df.iloc[0, 1:].sum()
    print(f"First row probability sum: {row_sum:.4f}")

    if not (0.9 < row_sum < 1.1):
        raise AssertionError(f"Probabilities do not sum to 1. Sum: {row_sum}")

    print(
        f"Inference completed successfully. Submission saved to {Config.SUBMISSION_PATH}"
    )
    print("\nAll demonstrations passed successfully.")


if __name__ == "__main__":
    try:
        demo_pipeline()
    except Exception as e:
        print(f"\n[ERROR] Demo failed with exception: {e}")
        sys.exit(1)
