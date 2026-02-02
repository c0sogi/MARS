import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import Config
from library.dataset import SpeechCommandsDataset
from library.model import EfficientNetV2Audio
from library.train import Trainer
from library.utils import set_seed


def run_demo():
    print("=== Starting Speech Command Recognition Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Setup for Fast Execution
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Override Config for speed and debugging
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 64  # Small subset for quick processing
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Redirect outputs to a demo directory
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.CHECKPOINT_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Ensure reproducibility
    set_seed(Config.SEED)

    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Subset Size: {Config.DEBUG_SUBSET_SIZE}")
    print(f"    Epochs: {Config.EPOCHS}")
    print(f"    Working Dir: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Dataset Verification
    # ---------------------------------------------------------
    print("\n[2] Verifying SpeechCommandsDataset...")

    # Initialize dataset
    train_ds = SpeechCommandsDataset(subset="train")

    # Check length
    assert (
        len(train_ds) == Config.DEBUG_SUBSET_SIZE
    ), f"Dataset length mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {len(train_ds)}"

    # Fetch one sample
    spec, label = train_ds[0]

    # Verify Spectrogram Shape: (1, n_mels, time)
    # Time dimension depends on sample rate (16000), hop length (160) -> 16000/160 + 1 = 101 frames
    expected_freq = Config.N_MELS
    expected_time = (Config.NUM_SAMPLES // Config.HOP_LENGTH) + 1

    print(f"    Sample shape: {spec.shape}")
    print(f"    Label index: {label}")

    assert spec.dim() == 3, "Spectrogram must be 3D tensor (C, F, T)"
    assert spec.shape[0] == 1, "Channel dimension must be 1"
    assert (
        spec.shape[1] == expected_freq
    ), f"Freq dimension mismatch. Got {spec.shape[1]}, expected {expected_freq}"
    # Allow small variance in time dimension due to padding/cropping logic
    assert (
        abs(spec.shape[2] - expected_time) <= 2
    ), f"Time dimension mismatch. Got {spec.shape[2]}, expected approx {expected_time}"
    assert isinstance(label, int), "Label must be an integer"
    assert 0 <= label < Config.NUM_CLASSES, "Label index out of bounds"

    print("    Dataset verification passed.")

    # ---------------------------------------------------------
    # 3. Model Verification
    # ---------------------------------------------------------
    print("\n[3] Verifying EfficientNetV2Audio Model...")

    model = EfficientNetV2Audio(num_classes=Config.NUM_CLASSES)
    model.to(Config.DEVICE)
    model.eval()

    # Create dummy input batch
    batch_size = 2
    dummy_input = torch.randn(batch_size, 1, expected_freq, spec.shape[2]).to(
        Config.DEVICE
    )

    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Input shape: {dummy_input.shape}")
    print(f"    Output shape: {output.shape}")

    assert output.shape == (
        batch_size,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(batch_size, Config.NUM_CLASSES)}, got {output.shape}"

    print("    Model verification passed.")

    # ---------------------------------------------------------
    # 4. Training Pipeline Execution
    # ---------------------------------------------------------
    print("\n[4] Executing Training Pipeline (Trainer)...")

    trainer = Trainer()

    # Run training (fit)
    # This will run for 1 epoch on the small subset
    trainer.fit()

    # Check if checkpoint was saved (if validation improved, which is likely from 0.0)
    # Note: If validation doesn't improve (rare with random init vs 0.0), file might not exist.
    # However, with 0.0 init best_score, any acc > 0 saves it.
    if os.path.exists(Config.CHECKPOINT_PATH):
        print(f"    Checkpoint saved at: {Config.CHECKPOINT_PATH}")
    else:
        print("    No checkpoint saved (Validation accuracy did not improve over 0.0).")

    # Generate submission
    trainer.generate_submission()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created!"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission file created with {len(df_sub)} rows.")

    # Check submission format
    assert (
        "fname" in df_sub.columns and "label" in df_sub.columns
    ), "Submission columns missing"
    assert len(df_sub) > 0, "Submission file is empty"

    print("    Pipeline execution successful.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
