import os
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import (
    INPUT_ROOT,
    WORKING_DIR,
    SUBMISSION_PATH,
    NUM_SAMPLES,
    NUM_CLASSES,
    SAMPLE_RATE,
)
from library.utils import set_seed
from library.data_manager import load_dataset_to_memory
from library.augmentations import GPUBackgroundNoiseMixer
from library.model import AudioEfficientNetV2
from library.engine import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration
    # ---------------------------------------------------------
    print("\n[1] Setting up environment...")
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print(f"    Input Root: {INPUT_ROOT}")
    print(f"    Working Dir: {WORKING_DIR}")

    # 2. Data Loading (Debug Mode)
    # ---------------------------------------------------------
    print("\n[2] Loading Dataset (Debug Mode)...")
    # We use a small debug_size to ensure the script runs quickly (e.g., < 5 mins)
    # This loads a subset of the data into memory.
    debug_size = 128
    data_dict = load_dataset_to_memory(load_cached_data=True, debug_size=debug_size)

    # Validate Data Dictionary
    required_keys = [
        "train_waveforms",
        "train_labels",
        "val_waveforms",
        "val_labels",
        "test_waveforms",
        "test_labels",
        "background_noise",
    ]
    for key in required_keys:
        if key not in data_dict:
            raise KeyError(f"Data dictionary missing key: {key}")

    # Validate Shapes
    print("    Validating data shapes...")
    assert data_dict["train_waveforms"].shape == (
        debug_size,
        NUM_SAMPLES,
    ), f"Train waveforms shape mismatch: {data_dict['train_waveforms'].shape}"
    assert data_dict["train_labels"].shape == (
        debug_size,
    ), f"Train labels shape mismatch: {data_dict['train_labels'].shape}"

    print(f"    Loaded {len(data_dict['train_waveforms'])} training samples.")
    print(f"    Loaded {len(data_dict['background_noise'])} background noise clips.")

    # 3. Augmentation Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying GPU Augmentations...")
    # Initialize the mixer
    mixer = GPUBackgroundNoiseMixer(data_dict["background_noise"], device=device)

    # Create a dummy batch of silence/low amplitude noise
    batch_size = 8
    dummy_wavs = torch.randn(batch_size, NUM_SAMPLES, device=device) * 0.01

    # Run augmentation (must be in training mode)
    mixer.train()
    augmented_wavs = mixer(dummy_wavs)

    # Checks
    assert (
        augmented_wavs.shape == dummy_wavs.shape
    ), "Augmentation changed output shape."
    # Compare device types to handle cases where one has an index (cuda:0) and the other doesn't (cuda)
    assert (
        augmented_wavs.device.type == device.type
    ), "Augmentation output on wrong device."
    assert (
        augmented_wavs.min() >= -1.0 and augmented_wavs.max() <= 1.0
    ), "Augmentation output not clamped to [-1, 1]."

    # Check that signal changed (since we mix noise)
    # Note: With NOISE_PROB=0.5, some might not change, but with batch=8, likelihood is high.
    # We force the check by observing if *any* difference exists across the batch.
    if not torch.allclose(dummy_wavs, augmented_wavs):
        print("    Augmentation successfully modified the waveforms.")
    else:
        print(
            "    Note: Augmentation did not modify waveforms (random chance or empty noise buffer)."
        )

    # 4. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")
    model = AudioEfficientNetV2(num_classes=NUM_CLASSES).to(device)
    model.eval()

    # Forward pass with dummy data
    with torch.no_grad():
        logits = model(dummy_wavs)

    # Check output
    assert logits.shape == (
        batch_size,
        NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(batch_size, NUM_CLASSES)}, got {logits.shape}"

    print("    Model forward pass successful. Output shape:", logits.shape)

    # 5. Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n[5] Running Training Loop (Short Run)...")

    # Initialize Trainer
    trainer = Trainer(data_dict, device=device)

    # Run fit for a minimal number of epochs to demonstrate functionality
    # The Trainer handles moving data to GPU, weighted sampling, EMA, etc.
    trainer.fit(epochs=2, patience=2)

    # Check if best model checkpoint was created
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"    Checkpoint created at: {best_model_path}")
    else:
        raise FileNotFoundError("Best model checkpoint was not created.")

    # 6. Submission Generation
    # ---------------------------------------------------------
    print("\n[6] Generating Submission...")
    trainer.predict_submission()

    if os.path.exists(SUBMISSION_PATH):
        df_sub = pd.read_csv(SUBMISSION_PATH)
        print(f"    Submission file created at: {SUBMISSION_PATH}")
        print(f"    Submission shape: {df_sub.shape}")

        # Validate submission format
        assert (
            "fname" in df_sub.columns and "label" in df_sub.columns
        ), "Submission missing required columns."
        assert len(df_sub) == len(
            pd.read_csv(os.path.join("./metadata", "test.csv"))
        ), "Submission row count does not match test set size."
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
