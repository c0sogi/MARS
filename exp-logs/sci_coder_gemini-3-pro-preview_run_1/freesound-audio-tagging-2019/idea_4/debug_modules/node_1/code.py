import os
import torch
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, calculate_lwlrap, load_checkpoint
from library.dataset import AudioDataset
from library.model import AudioClassifier
from library.trainer import Trainer

# Suppress potential warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demonstration():
    print("=== Starting Audio Tagging Task Demonstration ===")

    # 1. Setup and Configuration Overrides
    # We override Config parameters to ensure the demo runs quickly (within seconds/minutes)
    # and uses a small memory footprint.
    print("\n[1] Configuring environment for rapid demonstration...")
    set_seed(42)

    # Patching Config for speed
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.DEBUG = True  # Enable debug mode to use a subset of data
    Config.DEBUG_SUBSET_SIZE = 16  # Use only 16 samples
    Config.NUM_WORKERS = 0  # Disable multiprocessing to avoid overhead in this script
    Config.EARLY_STOPPING_PATIENCE = 1

    print(f"    Epochs: {Config.EPOCHS}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Subset Size: {Config.DEBUG_SUBSET_SIZE}")

    # 2. Verify Metric Logic (LWLRAP)
    print("\n[2] Verifying Metric Calculation (LWLRAP)...")
    # Create synthetic data
    # 2 samples, 3 classes
    # Sample 0: True=[1, 0, 1] (Classes 0 and 2 present)
    # Sample 1: True=[0, 1, 0] (Class 1 present)
    y_true = np.array([[1, 0, 1], [0, 1, 0]])

    # Perfect predictions for Sample 0 (High scores for 0 and 2)
    # Perfect predictions for Sample 1 (High score for 1)
    y_score = np.array([[0.9, 0.1, 0.8], [0.1, 0.9, 0.2]])

    # Expected Score: 1.0 (Perfect ranking)
    score = calculate_lwlrap(y_true, y_score)
    print(f"    Calculated LWLRAP: {score:.4f}")

    if not np.isclose(score, 1.0):
        raise AssertionError(f"Metric verification failed. Expected 1.0, got {score}")
    print("    Metric verification passed.")

    # 3. Verify Dataset Loading and Processing
    print("\n[3] Verifying AudioDataset...")
    # Initialize dataset in debug mode (uses 'train' split)
    dataset = AudioDataset(split="train", debug=True)

    # Check length
    ds_len = len(dataset)
    print(f"    Dataset length: {ds_len}")
    if ds_len != Config.DEBUG_SUBSET_SIZE:
        raise AssertionError(
            f"Dataset length mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {ds_len}"
        )

    # Check item structure
    spec, target = dataset[0]
    print(f"    Spectrogram Shape: {spec.shape}")  # Expected: (1, n_mels, time)
    print(f"    Target Shape: {target.shape}")  # Expected: (num_classes,)

    # Assertions
    expected_channels = 3 if Config.USE_INPUT_REPETITION else 1
    if spec.dim() != 3 or spec.shape[0] != expected_channels:
        raise AssertionError(
            f"Spectrogram has incorrect dimensions: {spec.shape}. Expected ({expected_channels}, F, T)."
        )
    if spec.shape[1] != Config.N_MELS:
        raise AssertionError(
            f"Spectrogram Mel bins mismatch. Expected {Config.N_MELS}, got {spec.shape[1]}."
        )
    if target.shape[0] != Config.NUM_CLASSES:
        raise AssertionError(
            f"Target class count mismatch. Expected {Config.NUM_CLASSES}, got {target.shape[0]}."
        )

    print("    Dataset verification passed.")

    # 4. Verify Model Architecture
    print("\n[4] Verifying AudioClassifier Model...")
    model = AudioClassifier()
    model.eval()

    # Create a dummy input tensor matching the dataset output
    # Shape: (Batch, Channels, Freq, Time)
    # Time dimension is variable in dataset due to cropping/padding, let's assume 200 frames
    channels = 3 if Config.USE_INPUT_REPETITION else 1
    dummy_input = torch.randn(2, channels, Config.N_MELS, 200)

    print(f"    Input Shape: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Output Shape: {output.shape}")

    # Assertions
    if output.shape != (2, Config.NUM_CLASSES):
        raise AssertionError(
            f"Model output shape mismatch. Expected (2, {Config.NUM_CLASSES}), got {output.shape}"
        )

    print("    Model verification passed.")

    # 5. Verify Training Pipeline
    print("\n[5] Verifying Trainer and Training Loop...")
    trainer = Trainer()

    print("    Starting training (1 Epoch)...")
    # This will train for 1 epoch on the debug subset and validate
    trainer.train(debug=True)

    # Verify that the model checkpoint was saved
    checkpoint_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("    Training failed to save 'best_model.pth'.")

    print(f"    Checkpoint successfully saved at: {checkpoint_path}")
    print("    Training loop verification passed.")

    # 6. Verify Checkpoint Loading
    print("\n[6] Verifying Checkpoint Loading...")
    new_model = AudioClassifier()
    # Move model to configured device (likely CPU or CUDA)
    new_model.to(Config.DEVICE)

    # Load the checkpoint we just saved
    best_val_score = load_checkpoint(new_model, filename="best_model.pth")

    print(f"    Loaded model with validation score: {best_val_score:.6f}")

    # Since we trained on random data/weights for 1 epoch, score is likely low,
    # but it should be a valid float.
    if not isinstance(best_val_score, float):
        raise AssertionError("    Failed to retrieve a valid score from checkpoint.")

    print("    Checkpoint loading verification passed.")

    print("\n=== Demonstration Complete: All Systems Operational ===")


if __name__ == "__main__":
    run_demonstration()
