import os
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, kl_divergence_loss
from library.data import get_dataloaders
from library.model import DualStreamNetwork
from library.train import run_training
from library.inference import run_inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    print("--- Starting Library Usage Demonstration ---")

    # 1. Configuration & Setup
    # Patch Config for a fast demo run
    print("\n[1] Configuring environment for demo...")
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Small subset for speed
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.OUTPUT_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = os.path.join(Config.OUTPUT_DIR, "submission.csv")

    # Ensure clean slate
    if os.path.exists(Config.OUTPUT_DIR):
        shutil.rmtree(Config.OUTPUT_DIR)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Set seeds
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Output Directory: {Config.OUTPUT_DIR}")

    # 2. Data Loading Verification
    print("\n[2] Verifying Data Loading...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_batch_size=Config.BATCH_SIZE,
        val_batch_size=Config.BATCH_SIZE,
        debug=True,
        debug_subset_size=Config.DEBUG_SUBSET_SIZE,
    )

    # Fetch one batch
    inputs, targets = next(iter(train_loader))
    eeg_batch, spec_batch = inputs

    # Move to device for consistency with model checks
    eeg_batch = eeg_batch.to(device)
    spec_batch = spec_batch.to(device)
    targets = targets.to(device)

    print(f"    EEG Batch Shape: {eeg_batch.shape}")
    print(f"    Spec Batch Shape: {spec_batch.shape}")
    print(f"    Targets Shape: {targets.shape}")

    # Assertions
    # EEG: (Batch, Channels=20, Time=5000)
    assert eeg_batch.shape == (
        Config.BATCH_SIZE,
        Config.EEG_CHANNELS,
        Config.EEG_SEQ_LEN,
    ), f"Mismatch in EEG shape: {eeg_batch.shape}"

    # Spec: (Batch, Channels=3, Height=512, Width=512)
    assert spec_batch.shape == (
        Config.BATCH_SIZE,
        3,
        Config.SPEC_SIZE[0],
        Config.SPEC_SIZE[1],
    ), f"Mismatch in Spectrogram shape: {spec_batch.shape}"

    # Targets: (Batch, Classes=6)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Mismatch in Targets shape: {targets.shape}"

    print("    Data loading verification passed.")

    # 3. Model Instantiation & Forward Pass
    print("\n[3] Verifying Model Architecture...")
    model = DualStreamNetwork().to(device)

    # Forward pass
    logits = model((eeg_batch, spec_batch))
    print(f"    Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch."
    assert not torch.isnan(logits).any(), "Model produced NaN logits."

    # Loss Calculation
    loss = kl_divergence_loss(logits, targets)
    print(f"    Calculated Loss: {loss.item():.4f}")

    assert loss.item() >= 0, "KL Divergence loss should be non-negative."
    print("    Model verification passed.")

    # 4. Training Loop Demonstration
    print("\n[4] Running Training Loop (1 Epoch)...")
    # run_training handles the loop, saving, and validation
    trained_model = run_training(debug=True, epochs=Config.EPOCHS)

    best_model_path = os.path.join(Config.OUTPUT_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Training failed to save best_model.pth"
    print("    Training complete. Checkpoint saved.")

    # 5. Inference Demonstration
    print("\n[5] Running Inference...")
    submission_df = run_inference(
        checkpoint_path=best_model_path,
        save_path=Config.SUBMISSION_PATH,
        debug=True,
        batch_size=Config.BATCH_SIZE,
    )

    # Verification
    print(f"    Submission Shape: {submission_df.shape}")

    # Check columns
    expected_cols = ["eeg_id"] + Config.CLASS_NAMES
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(submission_df.columns)}"

    # Check row count matches test loader dataset size
    # Note: In debug mode, run_inference re-instantiates loader with debug sampling.
    # We need to match that logic to verify count.
    assert len(submission_df) == len(
        test_loader.dataset
    ), f"Prediction count mismatch. Expected {len(test_loader.dataset)}, got {len(submission_df)}"

    # Check probability sum (should be approx 1.0)
    prob_sums = submission_df[Config.CLASS_NAMES].sum(axis=1)
    assert np.allclose(prob_sums, 1.0, atol=1e-4), "Probabilities do not sum to 1.0"

    print("    Inference verification passed.")
    print("\n--- Demonstration Complete ---")
