import os
import torch
import pandas as pd
import numpy as np
import sys

# Add the current directory to path to ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed
from library.dataset import (
    SpeechCommandDataset,
    get_dataloaders,
    load_background_noises,
)
from library.transforms import GPUAudioPreprocess
from library.model import FrequencyAttentiveResNeStCRNN
from library.train import run_training


def verify_dataset_and_loaders(device):
    print("\n=== Verifying Dataset and DataLoaders ===")

    # 1. Instantiate Dataset in Debug mode
    # This limits the number of samples to ensure speed
    ds_train = SpeechCommandDataset(split="train", debug=True, max_samples=100)

    # Validate length
    print(f"Debug Train Dataset Length: {len(ds_train)}")
    assert len(ds_train) > 0, "Dataset should not be empty."

    # Validate Item Structure
    waveform, label = ds_train[0]
    print(f"Sample Waveform Shape: {waveform.shape}")
    print(f"Sample Label ID: {label}")

    # Expected length is Sample Rate * Duration (16000 * 1.0)
    expected_len = int(Config.SAMPLE_RATE * Config.DURATION)
    assert (
        waveform.shape[0] == expected_len
    ), f"Waveform length mismatch. Expected {expected_len}, got {waveform.shape[0]}"
    assert isinstance(label, torch.Tensor), "Label should be a tensor."

    # 2. Verify DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False,  # Force reload to test logic
        debug=True,
        max_samples=64,  # Small sample for speed
    )

    # Fetch one batch
    batch_waves, batch_labels = next(iter(train_loader))
    print(f"Batch Waveforms Shape: {batch_waves.shape}")
    print(f"Batch Labels Shape: {batch_labels.shape}")

    assert batch_waves.dim() == 2, "Batch waveforms should be (Batch, Time)"
    assert batch_waves.size(1) == expected_len, "Batch time dimension incorrect."
    assert batch_labels.size(0) == batch_waves.size(
        0
    ), "Batch size mismatch between data and labels."

    # 3. Verify Background Noise Loading
    noise_data = load_background_noises(load_cached_data=False)
    if noise_data is not None:
        print(f"Background Noise Tensor Shape: {noise_data.shape}")
        assert noise_data.dim() == 1, "Noise data should be a 1D tensor."
    else:
        print(
            "No background noise found (this is acceptable if dir is missing in demo env)."
        )


def verify_preprocessing_and_model(device):
    print("\n=== Verifying Preprocessing and Model Architecture ===")

    # 1. Initialize Components
    preprocessor = GPUAudioPreprocess(device=device).to(device)
    model = FrequencyAttentiveResNeStCRNN().to(device)

    model.eval()
    preprocessor.eval()

    # 2. Create Dummy Input
    # Batch of 4, 1 second of audio at 16kHz
    batch_size = 4
    seq_len = int(Config.SAMPLE_RATE * Config.DURATION)
    dummy_audio = torch.randn(batch_size, seq_len).to(device)

    # 3. Test Preprocessing
    # Should output (Batch, 3, 224, 224)
    with torch.no_grad():
        spectrograms = preprocessor(dummy_audio, training=False)

    print(f"Generated Spectrogram Shape: {spectrograms.shape}")

    assert spectrograms.shape == (
        batch_size,
        3,
        224,
        224,
    ), f"Preprocessor output shape mismatch. Expected {(batch_size, 3, 224, 224)}, got {spectrograms.shape}"

    # Check for NaNs
    assert not torch.isnan(spectrograms).any(), "Spectrograms contain NaNs."

    # 4. Test Model Forward Pass
    with torch.no_grad():
        logits = model(spectrograms)

    print(f"Model Logits Shape: {logits.shape}")

    assert logits.shape == (
        batch_size,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(batch_size, Config.NUM_CLASSES)}, got {logits.shape}"


def verify_full_pipeline():
    print("\n=== Verifying Full Training Pipeline ===")

    # Run the provided training loop in debug mode
    # This handles Config setup, data loading, training, validation, and inference
    best_acc = run_training(debug=True, epochs=1, max_samples=200)

    print(f"Pipeline finished with Best Validation Accuracy: {best_acc}")

    # Verify Submission File
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    df_sub = pd.read_csv(submission_path)
    print(f"Submission File Loaded. Shape: {df_sub.shape}")
    print("Head:")
    print(df_sub.head())

    # Check columns
    assert (
        "fname" in df_sub.columns and "label" in df_sub.columns
    ), "Submission file missing required columns 'fname' and 'label'"

    # Check label validity
    valid_labels = set(Config.LABELS)
    pred_labels = set(df_sub["label"].unique())
    invalid_preds = pred_labels - valid_labels
    assert (
        len(invalid_preds) == 0
    ), f"Found invalid labels in submission: {invalid_preds}"


if __name__ == "__main__":
    # 1. Global Setup
    Config.setup()
    set_seed(Config.SEED)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Component Verification
    verify_dataset_and_loaders(device)
    verify_preprocessing_and_model(device)

    # 3. Integration Verification
    verify_full_pipeline()

    print("\n=== All Demonstrations and Verifications Passed Successfully ===")
