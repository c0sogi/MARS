import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, get_device
from library.dataset import get_dataloaders
from library.model import get_model
from library.trainer import Trainer
from library.inference import generate_submission


def run_demo():
    print("=== Starting Demonstration of Speech Command Recognition Pipeline ===")

    # ==========================================
    # 1. Configuration Override for Speed
    # ==========================================
    print("\n[1] Configuring environment for rapid demonstration...")

    # Modify Config attributes to run a fast, lightweight test
    Config.epochs = 1
    Config.batch_size = 8
    Config.debug = True
    Config.debug_sample_size = 32  # Small subset for speed
    Config.num_workers = 0  # Avoid multiprocessing overhead for small test

    # Setup specific working directory for this demo
    Config.working_dir = "./working/demo_run"
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)
    os.makedirs(Config.working_dir, exist_ok=True)

    Config.best_model_path = os.path.join(Config.working_dir, "best_model.pth")
    Config.submission_path = os.path.join(Config.working_dir, "demo_submission.csv")

    set_seed(Config.seed)
    device = get_device()
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.working_dir}")
    print(f"    Debug Mode: {Config.debug}")

    # ==========================================
    # 2. Data Loading Verification
    # ==========================================
    print("\n[2] Verifying Data Loading and Processing...")

    train_loader, val_loader, test_loader = get_dataloaders(debug=Config.debug)

    # Fetch one batch from train loader
    images, labels, fnames = next(iter(train_loader))

    print(f"    Batch Images Shape: {images.shape}")
    print(f"    Batch Labels Shape: {labels.shape}")

    # Assertions
    # Expected shape: (Batch, 1, n_mels, time_steps)
    # Time steps depend on hop_length and duration.
    # For 16000sr, 1.0s, hop 160 -> ~101 frames
    assert images.dim() == 4, f"Expected 4D input (B, C, F, T), got {images.shape}"
    assert (
        images.shape[0] == Config.batch_size
    ), f"Expected batch size {Config.batch_size}, got {images.shape[0]}"
    assert images.shape[1] == 1, f"Expected 1 channel, got {images.shape[1]}"
    assert (
        images.shape[2] == Config.n_mels
    ), f"Expected {Config.n_mels} mels, got {images.shape[2]}"

    assert labels.dim() == 1, "Labels should be 1D tensor"
    assert labels.shape[0] == Config.batch_size, "Labels batch size mismatch"

    print("    Data Loading verification passed.")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n[3] Verifying Model Architecture...")

    model = get_model()
    model.to(device)
    model.eval()

    # Create dummy input matching the batch shape
    dummy_input = torch.randn_like(images).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape[0] == Config.batch_size, "Output batch size mismatch"
    assert (
        output.shape[1] == Config.num_classes
    ), f"Expected {Config.num_classes} classes, got {output.shape[1]}"

    print("    Model verification passed.")

    # ==========================================
    # 4. Training Loop Execution
    # ==========================================
    print("\n[4] Executing Training Loop (1 Epoch)...")

    trainer = Trainer()

    # Run training
    # This uses the modified Config settings (1 epoch, debug data)
    trainer.fit(debug=Config.debug)

    # Verify checkpoint creation
    assert os.path.exists(
        Config.best_model_path
    ), f"Model checkpoint not found at {Config.best_model_path}"
    print(f"    Checkpoint successfully saved to {Config.best_model_path}")

    # ==========================================
    # 5. Inference and Submission
    # ==========================================
    print("\n[5] Running Inference and Generating Submission...")

    generate_submission(debug=Config.debug)

    # Verify submission file
    assert os.path.exists(
        Config.submission_path
    ), f"Submission file not found at {Config.submission_path}"

    df_sub = pd.read_csv(Config.submission_path)
    print(f"    Submission shape: {df_sub.shape}")
    print("    Submission Head:")
    print(df_sub.head())

    # Assertions on submission format
    expected_columns = ["fname", "label"]
    assert (
        list(df_sub.columns) == expected_columns
    ), f"Expected columns {expected_columns}, got {list(df_sub.columns)}"
    assert len(df_sub) > 0, "Submission file is empty"

    # Check if labels are valid strings (not indices)
    valid_labels = set(
        [
            "yes",
            "no",
            "up",
            "down",
            "left",
            "right",
            "on",
            "off",
            "stop",
            "go",
            "silence",
            "unknown",
        ]
    )
    sample_label = df_sub.iloc[0]["label"]
    assert (
        sample_label in valid_labels
    ), f"Invalid label found in submission: {sample_label}"

    print("    Inference verification passed.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
