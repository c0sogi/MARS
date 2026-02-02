import os
import sys
import torch
import numpy as np
import pandas as pd

# Import from the provided library
from library.config import Config
from library.utils import set_seed, compute_levenshtein
from library.data_loader import get_data_loaders
from library.model import SAMPNet
from library.train import run_training
from library.inference import generate_predictions


def demo_metric_logic():
    """
    Verifies the Levenshtein distance calculation logic.
    Metric = Sum(Levenshtein) / Total_Ground_Truth_Gestures
    """
    print("1. Verifying Metric Logic...")

    # Case 1: Perfect match
    t1 = [1, 2, 3]
    p1 = [1, 2, 3]
    # Distance = 0, Length = 3. Error = 0.0
    assert compute_levenshtein([p1], [t1]) == 0.0

    # Case 2: One substitution
    t2 = [1, 2, 3]
    p2 = [1, 5, 3]
    # Distance = 1, Length = 3. Error = 1/3
    err = compute_levenshtein([p2], [t2])
    assert abs(err - (1.0 / 3.0)) < 1e-6

    # Case 3: Empty prediction (Deletion)
    t3 = [1, 2]
    p3 = []
    # Distance = 2, Length = 2. Error = 1.0
    assert compute_levenshtein([p3], [t3]) == 1.0

    print("   Metric logic verified.")


def demo_pipeline():
    """
    Demonstrates the full pipeline: Config -> Data -> Model -> Train -> Inference.
    Optimized for speed using debug mode and reduced hyperparameters.
    """

    # -------------------------------------------------------------------------
    # 2. Configuration Override for Fast Execution
    # -------------------------------------------------------------------------
    print("\n2. Configuring for Demo Execution...")

    # Override Config defaults to run quickly
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Use a specific directory for demo outputs to avoid overwriting real work
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Update paths based on new dirs
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission_demo.csv")

    # Initialize environment (creates dirs, sets seeds)
    Config.setup()
    set_seed(Config.SEED)

    print(f"   Epochs: {Config.EPOCHS}")
    print(f"   Batch Size: {Config.BATCH_SIZE}")
    print(f"   Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 3. Data Loading & Shape Verification
    # -------------------------------------------------------------------------
    print("\n3. Verifying Data Loading...")

    # debug=True loads a small subset (first 32 samples)
    train_loader, val_loader, test_loader = get_data_loaders(debug=True)

    # Fetch a single batch to inspect
    batch = next(iter(train_loader))
    skel, audio, labels, aux_targets, lengths = batch

    # Validate Shapes
    # Skeleton: (B, T, 60)
    assert skel.ndim == 3
    assert skel.shape[0] == Config.BATCH_SIZE
    assert skel.shape[2] == Config.SKELETON_INPUT_DIM

    # Audio: (B, T, 13)
    assert audio.ndim == 3
    assert audio.shape[0] == Config.BATCH_SIZE
    assert audio.shape[2] == Config.AUDIO_INPUT_DIM

    # Labels: (B, T)
    assert labels.ndim == 2
    assert labels.shape[0] == Config.BATCH_SIZE

    # Aux Targets: (B, NumClasses)
    assert aux_targets.ndim == 2
    assert aux_targets.shape[1] == Config.NUM_CLASSES

    print(f"   Batch Shapes Verified: Skel {skel.shape}, Audio {audio.shape}")

    # -------------------------------------------------------------------------
    # 4. Model Instantiation & Forward Pass Verification
    # -------------------------------------------------------------------------
    print("\n4. Verifying Model Architecture...")

    model = SAMPNet().to(Config.DEVICE)

    # Move batch to device
    skel = skel.to(Config.DEVICE)
    audio = audio.to(Config.DEVICE)

    # Forward Pass
    logits, aux_preds = model(skel, audio, lengths)

    # Validate Output Shapes
    # Logits: (B, T, NumClasses)
    assert logits.shape == (Config.BATCH_SIZE, skel.shape[1], Config.NUM_CLASSES)

    # Aux Preds: (B, NumClasses)
    assert aux_preds.shape == (Config.BATCH_SIZE, Config.NUM_CLASSES)

    print("   Model Forward Pass Successful.")

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n5. Running Training Loop (Debug Mode)...")

    # run_training handles the loop, validation, and saving best model
    # We use debug=True to use the subset loaders we defined implicitly via Config
    # Note: run_training calls get_data_loaders internally, which respects our Config overrides
    run_training(debug=True)

    # Verify checkpoint creation
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not created."
    print("   Training complete. Checkpoint verified.")

    # -------------------------------------------------------------------------
    # 6. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n6. Running Inference (Debug Mode)...")

    generate_predictions(debug=True)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Check content format
    # Cite debug_lesson_2: Avoid Using Pandas for Ragged or Variable-Length Row Data
    with open(Config.SUBMISSION_PATH, "r") as f:
        lines = f.readlines()

    assert len(lines) > 0, "Submission file is empty."
    first_line = lines[0]
    assert "," in first_line, "Submission file format incorrect (missing comma)."

    print(f"   Inference complete. Submission saved to {Config.SUBMISSION_PATH}")
    print(f"   First line of submission: {first_line.strip()}")


if __name__ == "__main__":
    demo_metric_logic()
    demo_pipeline()
    print("\nAll demonstrations passed successfully.")
