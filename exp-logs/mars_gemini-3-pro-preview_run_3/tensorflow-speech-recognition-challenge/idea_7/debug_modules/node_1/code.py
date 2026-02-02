import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Add current directory to path to ensure library imports work
sys.path.append(os.getcwd())

# Import library components
from library.config import Config
from library.utils import set_seed, setup_logger
from library.audio_processor import load_audio, generate_multires_spectrogram
from library.dataset import (
    CachedSpeechDataset,
    SpecAugment,
    cache_dataset,
    get_dataloaders,
    get_test_dataloader,
)
from library.network import MR_SK_CRNN
from library.training_loop import train_model, predict_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Speech Command Recognition Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Only use 20 samples for cache/train
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Use main process to avoid overhead in demo

    # Ensure working directories are clean/ready
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.setup_directories()

    set_seed(Config.SEED)
    print("    Configuration updated: DEBUG=True, EPOCHS=1, BATCH=4")

    # ---------------------------------------------------------
    # 2. Audio Processing Demonstration
    # ---------------------------------------------------------
    print("\n[2] Testing Audio Processor...")

    # Read metadata to find a valid file
    df_train = pd.read_csv(Config.TRAIN_METADATA)
    sample_row = df_train.iloc[0]
    sample_filepath = sample_row["filepath"]
    print(f"    Loading sample file: {sample_filepath}")

    # Test load_audio
    waveform = load_audio(sample_filepath)
    print(f"    Waveform shape: {waveform.shape}")

    # Assertions
    assert isinstance(waveform, torch.Tensor), "Waveform must be a torch.Tensor"
    assert waveform.shape == (
        1,
        Config.NUM_SAMPLES,
    ), f"Expected shape (1, {Config.NUM_SAMPLES}), got {waveform.shape}"

    # Test generate_multires_spectrogram
    spec = generate_multires_spectrogram(waveform)
    print(f"    Spectrogram shape: {spec.shape}")

    # Assertions
    # Shape should be (3, 64, Time). Time depends on hop length.
    # 16000 samples / 160 hop + 1 (centered) = 101 frames
    expected_time_steps = 101
    assert isinstance(spec, np.ndarray), "Spectrogram must be a numpy array"
    assert spec.shape == (
        3,
        Config.N_MELS,
        expected_time_steps,
    ), f"Expected shape (3, {Config.N_MELS}, {expected_time_steps}), got {spec.shape}"

    print("    Audio Processor tests passed.")

    # ---------------------------------------------------------
    # 3. Dataset & Augmentation Demonstration
    # ---------------------------------------------------------
    print("\n[3] Testing Dataset and Augmentation...")

    # Manually trigger caching for a few files
    print("    Caching subset of data...")
    data_list = cache_dataset(
        Config.TRAIN_METADATA, Config.CACHE_DIR, load_cached_data=False, debug=True
    )

    assert len(data_list) > 0, "Data list should not be empty"
    assert (
        len(data_list) <= Config.DEBUG_SAMPLE_SIZE
    ), "Should respect debug sample size"

    # Instantiate Dataset
    augmenter = SpecAugment(freq_mask_param=5, time_mask_param=10)
    dataset = CachedSpeechDataset(data_list, transform=augmenter)

    # Fetch one item
    item_spec, item_label = dataset[0]
    print(f"    Dataset item shape: {item_spec.shape}, Label: {item_label}")

    # Assertions
    assert isinstance(item_spec, torch.Tensor), "Dataset output spec must be Tensor"
    assert isinstance(item_label, torch.Tensor), "Dataset output label must be Tensor"
    assert item_spec.shape == (
        3,
        Config.N_MELS,
        expected_time_steps,
    ), "Incorrect spec shape from dataset"

    print("    Dataset tests passed.")

    # ---------------------------------------------------------
    # 4. Network Architecture Demonstration
    # ---------------------------------------------------------
    print("\n[4] Testing Network Architecture (MR_SK_CRNN)...")

    device = torch.device("cpu")  # Use CPU for simple shape check
    model = MR_SK_CRNN().to(device)
    model.eval()

    # Create dummy batch
    dummy_input = torch.randn(2, 3, Config.N_MELS, expected_time_steps).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Model output shape: {output.shape}")

    # Assertions
    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Expected output shape (2, {Config.NUM_CLASSES}), got {output.shape}"

    print("    Network architecture tests passed.")

    # ---------------------------------------------------------
    # 5. Full Training & Inference Loop
    # ---------------------------------------------------------
    print("\n[5] Running Training Loop (Debug Mode)...")

    # Run training (this handles dataloaders, training, validation, saving)
    train_model(debug=True)

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved."
    print("    Training complete. Model saved.")

    print("\n[6] Running Inference (Debug Mode)...")

    # Run inference
    predict_submission(debug=True)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission rows: {len(df_sub)}")
    print("    First 3 rows:")
    print(df_sub.head(3))

    assert (
        "fname" in df_sub.columns and "label" in df_sub.columns
    ), "Submission columns missing"
    assert len(df_sub) > 0, "Submission is empty"

    print("    Inference complete.")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    main()
