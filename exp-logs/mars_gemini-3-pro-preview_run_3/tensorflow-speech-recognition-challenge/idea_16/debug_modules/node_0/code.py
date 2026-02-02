import os
import torch
import pandas as pd
import numpy as np
import sys

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.trainer import Trainer


def run_demo():
    print("=== Setting up Configuration for Speed/Demo ===")
    # Override Config parameters to ensure the script runs quickly (approx 1-2 mins)
    # and doesn't require internet access (disable pretrained weights download).
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Small subset for demo
    Config.EPOCHS = 2  # Run 2 epochs to test scheduler step
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.PRETRAINED = False  # Skip downloading weights for speed/offline safety
    Config.PATIENCE = 2  # Short patience

    # Ensure working directory exists
    Config.setup()

    # Set seed for reproducibility
    set_seed(Config.SEED)

    print("\n=== Initializing Trainer ===")
    # The Trainer handles DataLoaders, Model, and Processor initialization
    trainer = Trainer()

    print("\n=== Verifying Components Logic ===")
    # 1. Verify DataLoader
    print("Fetching a batch from train_loader...")
    try:
        waveforms, targets = next(iter(trainer.train_loader))
        print(f"  Waveform shape: {waveforms.shape}")
        print(f"  Targets shape: {targets.shape}")

        # Assertions
        assert waveforms.shape[0] == Config.BATCH_SIZE
        assert waveforms.shape[1] == Config.NUM_SAMPLES
        assert targets.shape[0] == Config.BATCH_SIZE
    except StopIteration:
        raise Exception("Train loader is empty!")

    # 2. Verify Audio Processor (Augmentation + Spectrograms)
    print("Running batch through GPUAudioProcessor...")
    # Move to device
    waveforms = waveforms.to(trainer.device)

    # Ensure training mode (enables augmentation logic, though we just check shape)
    trainer.processor.train()
    features = trainer.processor(waveforms)

    print(f"  Features shape: {features.shape}")
    # Expected: (Batch, 3, 224, 224) -> 3 channels for multi-resolution
    assert features.shape == (
        Config.BATCH_SIZE,
        3,
        224,
        224,
    ), f"Expected {(Config.BATCH_SIZE, 3, 224, 224)}, got {features.shape}"

    # 3. Verify Model Forward Pass
    print("Running features through FreqAttnResNeStCRNN...")
    trainer.model.train()
    logits = trainer.model(features)

    print(f"  Logits shape: {logits.shape}")
    # Expected: (Batch, NUM_CLASSES)
    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {logits.shape}"

    print("Component verification passed.")

    print("\n=== Executing Training Loop (fit) ===")
    # This runs train_one_epoch and validate for Config.EPOCHS
    trainer.fit()

    # Check if best model was saved
    if os.path.exists(trainer.best_model_path):
        print(f"Best model saved at: {trainer.best_model_path}")
    else:
        # It's possible no improvement happened if initialized randomly and trained for 1 epoch,
        # but the file should usually exist if validation runs.
        print("Warning: best_model.pth not found (might be due to very short run).")

    print("\n=== Executing Inference (predict) ===")
    # This generates the submission file
    trainer.predict()

    print("\n=== Validating Submission File ===")
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission loaded. Shape: {sub_df.shape}")
        print("First 5 rows:")
        print(sub_df.head())

        # Validation
        assert "fname" in sub_df.columns
        assert "label" in sub_df.columns
        assert len(sub_df) > 0

        # Check if labels are valid
        valid_labels = set(Config.ALL_LABELS)
        predicted_labels = set(sub_df["label"].unique())
        invalid_preds = predicted_labels - valid_labels
        if invalid_preds:
            raise ValueError(f"Submission contains invalid labels: {invalid_preds}")

        print("Submission format valid.")
    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
