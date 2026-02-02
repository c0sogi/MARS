import os
import sys
import torch
import pandas as pd
import numpy as np


# ==========================================
# 1. Patch tqdm to suppress progress bars
# ==========================================
# The provided library files use tqdm. To comply with the requirement
# "Do not print progress bars", we patch tqdm before importing the library.
class MockTqdm:
    def __init__(self, iterable=None, *args, **kwargs):
        self.iterable = iterable if iterable is not None else []

    def __iter__(self):
        return iter(self.iterable)

    def set_description(self, desc):
        pass

    def update(self, n=1):
        pass

    def close(self):
        pass


# Mock the module
import tqdm

tqdm.tqdm = MockTqdm

# ==========================================
# 2. Import Library Modules
# ==========================================
# We import Config first to modify it before other modules might use it
from library.config import Config

# Modify Config for fast execution / debugging
Config.DEBUG = True
Config.DEBUG_SAMPLE_SIZE = 100  # Use only 100 samples for demonstration
Config.NUM_EPOCHS = 1  # Train for only 1 epoch
Config.BATCH_SIZE = 16  # Smaller batch size
Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

from library.utils import set_seed
from library.dataset import SpeechDataset, get_dataloaders
from library.model import ConvNeXtAudio
from library.train import run_training

# ==========================================
# 3. Main Execution
# ==========================================
if __name__ == "__main__":
    print("Starting Speech Command Recognition Demo...")

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # ---------------------------------------------------------
    # Step 1: Verify Dataset Logic
    # ---------------------------------------------------------
    print("\n[Step 1] Verifying Dataset...")
    train_dataset = SpeechDataset(Config.TRAIN_METADATA, mode="train")

    # Assert we are in debug mode and size is correct
    assert (
        len(train_dataset) <= Config.DEBUG_SAMPLE_SIZE
    ), f"Dataset size {len(train_dataset)} exceeds debug limit."

    # Get a single sample
    spec, label_id, fname = train_dataset[0]

    # Check Spectrogram Shape: (1, n_mels, time_frames)
    # n_mels = 128. Time frames approx 101 for 1 sec audio with hop 160.
    print(f"  Sample shape: {spec.shape}")
    print(f"  Label ID: {label_id}")

    assert spec.dim() == 3, "Spectrogram must be a 3D tensor (C, F, T)."
    assert spec.shape[0] == 1, "Channel dimension should be 1."
    assert spec.shape[1] == Config.N_MELS, f"Freq dimension should be {Config.N_MELS}."
    assert isinstance(label_id, (int, np.integer)), "Label ID must be an integer."

    print("  Dataset verification passed.")

    # ---------------------------------------------------------
    # Step 2: Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[Step 2] Verifying Model...")
    model = ConvNeXtAudio(num_classes=Config.NUM_CLASSES, pretrained=False)
    model.to(Config.DEVICE)
    model.eval()

    # Create dummy input batch
    dummy_input = torch.randn(2, 1, Config.N_MELS, spec.shape[2]).to(Config.DEVICE)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")

    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected (2, {Config.NUM_CLASSES}), got {output.shape}"

    print("  Model verification passed.")

    # ---------------------------------------------------------
    # Step 3: Run Training & Inference Pipeline
    # ---------------------------------------------------------
    print("\n[Step 3] Running Training Pipeline (Fast Mode)...")

    # This function handles DataLoaders, Trainer initialization,
    # fitting (training loop), and submission generation.
    run_training(
        epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=Config.LEARNING_RATE,
        patience=1,
    )

    print("  Training pipeline finished.")

    # ---------------------------------------------------------
    # Step 4: Verify Outputs
    # ---------------------------------------------------------
    print("\n[Step 4] Verifying Outputs...")

    # Check 1: Best Model Checkpoint
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"  Checkpoint found at: {Config.BEST_MODEL_PATH}")
        checkpoint = torch.load(Config.BEST_MODEL_PATH, map_location="cpu")
        assert "model_state_dict" in checkpoint, "Checkpoint missing model_state_dict."
        assert "epoch" in checkpoint, "Checkpoint missing epoch info."
    else:
        # It's possible no improvement happened in 1 epoch, but usually one is saved.
        # If not, we check if the code ran without error.
        print(
            "  Notice: No checkpoint saved (validation accuracy might not have improved)."
        )

    # Check 2: Submission File
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"  Submission file found at: {Config.SUBMISSION_PATH}")
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)

        # Check columns
        assert "fname" in df_sub.columns, "Submission missing 'fname' column."
        assert "label" in df_sub.columns, "Submission missing 'label' column."

        # Check length (should match test set size, constrained by DEBUG_SAMPLE_SIZE)
        # Note: Test set is also sampled in SpeechDataset if Config.DEBUG is True
        expected_len = min(6473, Config.DEBUG_SAMPLE_SIZE)
        # Actually, get_dataloaders creates test_dataset which respects Config.DEBUG
        # Let's verify the length matches the actual file on disk
        print(f"  Submission rows: {len(df_sub)}")
        assert len(df_sub) > 0, "Submission file is empty."

        # Check label validity
        valid_labels = set(Config.LABELS)
        pred_labels = set(df_sub["label"].unique())
        invalid_preds = pred_labels - valid_labels
        assert not invalid_preds, f"Submission contains invalid labels: {invalid_preds}"

        print("  Submission verification passed.")
    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    print("\nAll demonstrations and verifications completed successfully.")
