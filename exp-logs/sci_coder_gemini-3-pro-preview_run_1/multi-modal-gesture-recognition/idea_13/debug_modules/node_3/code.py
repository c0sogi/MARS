import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config, set_seed
from library.data_loader import get_dataloaders
from library.model import IICGRN
from library.engine import Trainer
from library.utils import compute_levenshtein


def run_demo():
    # ==========================================
    # 1. Setup & Configuration Overrides
    # ==========================================
    print(">>> Setting up demo configuration...")

    # Set seed for reproducibility
    set_seed(42)

    # Override Config to use a separate demo directory and run fast
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache_demo")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.STATS_PATH = os.path.join(Config.WORKING_DIR, "stats.npz")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    # Reduce compute load for demo
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.HIDDEN_SIZE = 64  # Smaller model for speed

    # Manually create these directories since Config code ran at import time with old paths
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Clean up stats if they exist to force re-computation on the debug subset
    if os.path.exists(Config.STATS_PATH):
        os.remove(Config.STATS_PATH)

    print(f"Working Directory: {Config.WORKING_DIR}")

    # ==========================================
    # 2. Data Loading Verification
    # ==========================================
    print("\n>>> Loading Data (Debug Mode)...")

    # debug=True loads a very small subset (20 train, 10 val, 10 test)
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Check dataset sizes
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Verify Batch Structure
    batch = next(iter(train_loader))

    # Assertions to verify data integrity
    assert "skeleton" in batch, "Batch missing skeleton data"
    assert "audio" in batch, "Batch missing audio data"
    assert "dense_labels" in batch, "Batch missing dense labels"

    # Check shapes: (Batch, Time, Features)
    # Skeleton: Features=60
    assert batch["skeleton"].dim() == 3
    assert batch["skeleton"].shape[2] == 60
    # Audio: Features=13 (N_MFCC)
    assert batch["audio"].dim() == 3
    assert batch["audio"].shape[2] == 13
    # Dense Labels: (Batch, Time)
    assert batch["dense_labels"].dim() == 2

    print("Data Loader verification passed.")

    # ==========================================
    # 3. Model Logic Verification
    # ==========================================
    print("\n>>> Verifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IICGRN().to(device)

    # Move batch to device
    skel_input = batch["skeleton"].to(device)
    audio_input = batch["audio"].to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        logits = model(skel_input, audio_input)

    # Check output shape: (Batch, Time, NumClasses)
    # NumClasses = 21 (20 gestures + 1 background)
    assert logits.shape[0] == skel_input.shape[0]
    assert logits.shape[1] == skel_input.shape[1]
    assert logits.shape[2] == 21

    # Check for NaNs
    assert not torch.isnan(logits).any(), "Model output contains NaNs"

    print("Model forward pass verification passed.")

    # ==========================================
    # 4. Training Loop Demonstration
    # ==========================================
    print("\n>>> Starting Training Loop (1 Epoch)...")

    trainer = Trainer(device_str=str(device))

    # Run training
    # This uses the subset data, so it should be very fast
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created."

    print("Training loop finished and checkpoint verified.")

    # ==========================================
    # 5. Inference & Submission
    # ==========================================
    print("\n>>> Running Inference on Test Set...")

    trainer.predict(test_loader)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Check content of submission
    with open(Config.SUBMISSION_PATH, "r") as f:
        lines = f.readlines()
    # Format: SessionID, Label1, Label2...
    # Since we used debug mode, we expect 10 rows (head(10) in data_loader)
    assert len(lines) == 10, f"Expected 10 predictions, found {len(lines)}"

    print("Inference finished and submission verified.")

    # ==========================================
    # 6. Metric Logic Verification
    # ==========================================
    print("\n>>> Verifying Metric Logic (Levenshtein)...")

    # Case 1: Exact match
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    score = compute_levenshtein([seq1], [seq2])
    assert score == 0.0, f"Expected 0.0, got {score}"

    # Case 2: One insertion
    seq1 = [1, 2]
    seq2 = [
        1,
        2,
        3,
    ]  # Target has 3, pred has 2. Distance = 1 (deletion from target or insertion to pred)
    # Function returns Total Distance / Total Target Length
    # Distance = 1, Length = 3 -> 0.333...
    score = compute_levenshtein([seq1], [seq2])
    assert abs(score - (1 / 3)) < 1e-6, f"Expected 0.333, got {score}"

    # Case 3: Empty prediction
    seq1 = []
    seq2 = [5, 5]
    # Distance = 2, Length = 2 -> 1.0
    score = compute_levenshtein([seq1], [seq2])
    assert score == 1.0, f"Expected 1.0, got {score}"

    print("Metric verification passed.")

    print("\n>>> All demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
