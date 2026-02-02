import os
import torch
import pandas as pd
import numpy as np
import random
import shutil

# Import from the provided library files
from library.config import AUDIO_CONFIG, TRAIN_CONFIG, LABEL_CONFIG, PATH_CONFIG
from library.dataset import get_dataloaders, SpeechDataset
from library.transforms import LogMelSpectrogram, WaveformAugment, SpecAugment
from library.model import DilatedEfficientNetB2
from library.trainer import Trainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_demo():
    print("Initializing Demo...")
    set_seed(42)

    # -------------------------------------------------------------------------
    # 1. Configure for Speed (Override Defaults)
    # -------------------------------------------------------------------------
    print("\n[1] Configuring for fast demonstration run...")

    # Reduce training duration
    TRAIN_CONFIG.epochs = 2
    TRAIN_CONFIG.swa_start_epoch = 1  # Trigger SWA logic immediately after epoch 1
    TRAIN_CONFIG.batch_size = 16  # Small batch size for debug
    TRAIN_CONFIG.num_workers = 2

    # Update paths to ensure we write to writable ./working directory
    PATH_CONFIG.submission_path = "./working/submission.csv"
    PATH_CONFIG.cache_dir = "./working/demo_cache"
    TRAIN_CONFIG.checkpoint_dir = "./working/demo_checkpoints"

    # Update model paths based on new checkpoint dir
    TRAIN_CONFIG.best_model_path = os.path.join(
        TRAIN_CONFIG.checkpoint_dir, "best_model.pth"
    )
    TRAIN_CONFIG.swa_model_path = os.path.join(
        TRAIN_CONFIG.checkpoint_dir, "swa_model.pth"
    )

    os.makedirs(PATH_CONFIG.cache_dir, exist_ok=True)
    os.makedirs(TRAIN_CONFIG.checkpoint_dir, exist_ok=True)

    print(f"  Epochs: {TRAIN_CONFIG.epochs}")
    print(f"  Batch Size: {TRAIN_CONFIG.batch_size}")
    print(f"  Device: {TRAIN_CONFIG.device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading...")

    # Use debug=True to load only 100 samples per split
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=True, load_cached_data=False
    )

    # Fetch one batch
    waveforms, labels = next(iter(train_loader))

    print(f"  Waveform Batch Shape: {waveforms.shape}")
    print(f"  Labels Batch Shape: {labels.shape}")

    # Assertions
    expected_time_steps = int(AUDIO_CONFIG.sample_rate * AUDIO_CONFIG.duration)
    assert waveforms.shape == (
        TRAIN_CONFIG.batch_size,
        expected_time_steps,
    ), f"Expected waveform shape ({TRAIN_CONFIG.batch_size}, {expected_time_steps}), got {waveforms.shape}"
    assert labels.shape == (TRAIN_CONFIG.batch_size,), "Labels shape mismatch"
    assert labels.max() < LABEL_CONFIG.num_classes, "Label index out of bounds"
    print("  Data Loading Verified.")

    # -------------------------------------------------------------------------
    # 3. Transforms Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Transforms...")
    device = torch.device(TRAIN_CONFIG.device)
    waveforms = waveforms.to(device)

    # Instantiate Transforms
    log_mel = LogMelSpectrogram(AUDIO_CONFIG).to(device)
    wave_aug = WaveformAugment(p=1.0).to(device)  # Force augment
    spec_aug = SpecAugment(p=1.0).to(device)  # Force augment

    # 3.1 Waveform Augmentation
    aug_waveforms = wave_aug(waveforms)
    assert aug_waveforms.shape == waveforms.shape, "Waveform Augmentation changed shape"
    assert not torch.allclose(
        waveforms, aug_waveforms
    ), "Waveform Augmentation did not modify data"

    # 3.2 Spectrogram Generation
    specs = log_mel(aug_waveforms)
    print(f"  Spectrogram Shape: {specs.shape}")

    # Expected: (Batch, 1, n_mels, time_steps)
    # time_steps = (16000 * 1.0) // 160 + 1 = 101
    expected_spec_shape = (
        TRAIN_CONFIG.batch_size,
        1,
        AUDIO_CONFIG.n_mels,
        AUDIO_CONFIG.time_steps,
    )
    assert (
        specs.shape == expected_spec_shape
    ), f"Expected spec shape {expected_spec_shape}, got {specs.shape}"

    # 3.3 Spectrogram Augmentation
    aug_specs = spec_aug(specs)
    assert aug_specs.shape == specs.shape, "Spec Augmentation changed shape"
    # Note: SpecAugment might mask zeros with zeros, but usually changes something.
    print("  Transforms Verified.")

    # -------------------------------------------------------------------------
    # 4. Model Verification
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    model = DilatedEfficientNetB2(num_classes=LABEL_CONFIG.num_classes).to(device)

    # Forward pass
    outputs = model(specs)
    print(f"  Model Output Shape: {outputs.shape}")

    assert outputs.shape == (
        TRAIN_CONFIG.batch_size,
        LABEL_CONFIG.num_classes,
    ), f"Expected output shape ({TRAIN_CONFIG.batch_size}, {LABEL_CONFIG.num_classes}), got {outputs.shape}"

    print("  Model Verified.")

    # -------------------------------------------------------------------------
    # 5. Trainer Execution (Fit & Predict)
    # -------------------------------------------------------------------------
    print("\n[5] Running Trainer (Training Loop & Submission)...")

    # Initialize Trainer
    trainer = Trainer(train_loader, val_loader, test_loader)

    # Run Training
    # This will run for 2 epochs (as configured above)
    # Epoch 1: Standard training
    # Epoch 2: SWA collection
    # Then: BN Update and Submission Generation
    trainer.fit()

    # -------------------------------------------------------------------------
    # 6. Submission Verification
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Submission...")

    if not os.path.exists(PATH_CONFIG.submission_path):
        raise FileNotFoundError(
            f"Submission file not found at {PATH_CONFIG.submission_path}"
        )

    df_sub = pd.read_csv(PATH_CONFIG.submission_path)
    print(f"  Submission Head:\n{df_sub.head()}")
    print(f"  Submission Shape: {df_sub.shape}")

    # Verify columns
    assert list(df_sub.columns) == ["fname", "label"], "Submission columns mismatch"

    # Verify row count matches test loader (debug mode = 100 samples)
    # The debug loader subsets the dataset to indices 0-99.
    expected_rows = 100
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df_sub)}"

    # Verify labels are valid submission labels
    valid_submission_labels = set(
        LABEL_CONFIG.target_labels
        + [LABEL_CONFIG.silence_label, LABEL_CONFIG.unknown_label]
    )
    unique_preds = set(df_sub["label"].unique())
    invalid_preds = unique_preds - valid_submission_labels
    assert not invalid_preds, f"Found invalid labels in submission: {invalid_preds}"

    print("  Submission Verified.")
    print("\nDemo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
