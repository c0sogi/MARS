import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, get_device
from library.features import load_and_process_audio, compute_multires_spectrogram
from library.sk_resnet import get_model
from library.train import run_training, generate_submission


def main():
    # ==========================================
    # 1. Configuration for Fast Demo Run
    # ==========================================
    print("--- Setting up Demo Configuration ---")

    # Use a specific directory for this demo to avoid overwriting production runs
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Optimize for speed: Small batch, 1 epoch, minimal samples, no multiprocessing overhead
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0
    Config.MAX_TRAIN_SAMPLES = 20  # Limit dataset to 20 samples for quick execution

    # Initialize directories and seed
    Config.setup()
    set_seed(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Verify Feature Extraction Logic
    # ==========================================
    print("\n--- Verifying Feature Extraction ---")

    # Load train metadata to get a valid file path
    train_csv_path = os.path.join(Config.METADATA_DIR, "train.csv")
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Metadata not found at {train_csv_path}")

    df_train = pd.read_csv(train_csv_path)
    sample_filepath = df_train.iloc[0]["filepath"]
    print(f"Testing with file: {sample_filepath}")

    # Test 1: Audio Loading and Processing
    waveform = load_and_process_audio(sample_filepath)
    expected_samples = Config.NUM_SAMPLES

    print(f"Waveform Shape: {waveform.shape}")
    assert waveform.shape == (
        1,
        expected_samples,
    ), f"Waveform shape mismatch. Expected (1, {expected_samples}), got {waveform.shape}"

    # Test 2: Multi-Resolution Spectrogram Computation
    spec = compute_multires_spectrogram(waveform)
    # Expected shape: (3 channels, 64 mels, 101 time steps)
    # 101 comes from (16000 samples / 160 hop) + 1, handled in features.py
    expected_spec_shape = (3, Config.N_MELS, 101)

    print(f"Spectrogram Shape: {spec.shape}")
    assert (
        spec.shape == expected_spec_shape
    ), f"Spectrogram shape mismatch. Expected {expected_spec_shape}, got {spec.shape}"

    print("Feature extraction verified successfully.")

    # ==========================================
    # 3. Verify Model Architecture
    # ==========================================
    print("\n--- Verifying Model Architecture ---")

    model = get_model().to(device)
    model.eval()

    # Create a dummy batch of input data
    # Batch size: 2, Channels: 3, Freq: 64, Time: 101
    dummy_input = torch.randn(2, 3, Config.N_MELS, 101).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Expect output shape: (Batch Size, Num Classes)
    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Model output mismatch. Expected (2, {Config.NUM_CLASSES}), got {output.shape}"

    print("Model architecture verified successfully.")

    # ==========================================
    # 4. Verify Training Pipeline
    # ==========================================
    print("\n--- Running Training Pipeline (Demo) ---")

    # run_training handles:
    # 1. Loading metadata and subsampling (based on Config.MAX_TRAIN_SAMPLES)
    # 2. Caching features to disk
    # 3. Creating DataLoaders
    # 4. Training for Config.EPOCHS
    # 5. Saving the best model to Config.MODEL_PATH
    # 6. Returning the test_loader

    test_loader = run_training(
        epochs=Config.EPOCHS, patience=1, max_samples=Config.MAX_TRAIN_SAMPLES
    )

    # Check if model file was created
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Training failed: Model file not found at {Config.MODEL_PATH}"
        )

    print("Training pipeline completed and model saved.")

    # ==========================================
    # 5. Verify Submission Generation
    # ==========================================
    print("\n--- Generating Submission ---")

    generate_submission(test_loader)

    # Check if submission file was created
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Inference failed: Submission file not found at {Config.SUBMISSION_PATH}"
        )

    # Validate submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Head:\n{df_sub.head()}")

    assert list(df_sub.columns) == [
        "fname",
        "label",
    ], "Submission columns are incorrect."

    # Since we subsampled the test set via MAX_TRAIN_SAMPLES in get_dataloaders,
    # the submission length should match that limit (or the total test size if smaller).
    expected_len = min(
        len(pd.read_csv(os.path.join(Config.METADATA_DIR, "test.csv"))),
        Config.MAX_TRAIN_SAMPLES,
    )
    assert (
        len(df_sub) == expected_len
    ), f"Submission length mismatch. Expected {expected_len}, got {len(df_sub)}"

    print("\n=== All Tests and Demonstrations Passed Successfully ===")


if __name__ == "__main__":
    main()
