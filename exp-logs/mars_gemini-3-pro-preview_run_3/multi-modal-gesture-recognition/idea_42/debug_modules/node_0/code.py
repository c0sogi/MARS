import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Ensure the current directory is in the path to import library modules
sys.path.append(".")

from library.config import Config
from library.utils import compute_levenshtein, decode_predictions
from library.dataset import GestureDataset
from library.model import LSMCN
from library.trainer import Trainer


def run_demo():
    print("=== Starting Demonstration Script ===")

    # ==========================================
    # 1. Configuration Overrides for Speed
    # ==========================================
    print("\n[1] Overriding Configuration for Fast Demonstration...")
    # Reduce epochs to 1 for quick validation
    Config.NUM_EPOCHS = 1
    # Reduce batch size to ensure it runs on any hardware quickly
    Config.BATCH_SIZE = 8
    # Reduce hidden dimension to speed up model initialization and forward pass
    Config.HIDDEN_DIM = 64
    # Ensure we use a fresh working directory for this run if needed,
    # but we'll stick to the default to use existing cache if available.
    print(f"    NUM_EPOCHS set to: {Config.NUM_EPOCHS}")
    print(f"    BATCH_SIZE set to: {Config.BATCH_SIZE}")
    print(f"    HIDDEN_DIM set to: {Config.HIDDEN_DIM}")

    # ==========================================
    # 2. Verify Utility Functions
    # ==========================================
    print("\n[2] Verifying Utility Functions...")

    # Test Levenshtein Distance
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist_eq = compute_levenshtein(seq1, seq2)
    assert dist_eq == 0, f"Expected distance 0 for identical sequences, got {dist_eq}"

    seq3 = [1, 2, 4]
    dist_diff = compute_levenshtein(seq1, seq3)
    assert dist_diff == 1, f"Expected distance 1 for one substitution, got {dist_diff}"

    seq4 = [1, 2]
    dist_del = compute_levenshtein(seq1, seq4)
    assert dist_del == 1, f"Expected distance 1 for deletion, got {dist_del}"

    print("    Levenshtein distance logic verified.")

    # Test Decode Predictions
    # Create dummy probabilities: (Time=10, Classes=21)
    # Sequence: 5 frames of class 1, 5 frames of class 2
    # Min duration is 5
    dummy_probs = torch.zeros(10, Config.NUM_CLASSES)
    dummy_probs[0:5, 1] = 1.0  # Class 1
    dummy_probs[5:10, 2] = 1.0  # Class 2

    decoded = decode_predictions(dummy_probs, min_duration=5)
    assert decoded == [1, 2], f"Expected [1, 2], got {decoded}"

    # Test filtering (short duration)
    dummy_probs_short = torch.zeros(10, Config.NUM_CLASSES)
    dummy_probs_short[0:4, 1] = 1.0  # Class 1 (4 frames < 5)
    dummy_probs_short[4:10, 2] = 1.0  # Class 2 (6 frames >= 5)

    decoded_short = decode_predictions(dummy_probs_short, min_duration=5)
    assert decoded_short == [2], f"Expected [2] (filtered class 1), got {decoded_short}"

    print("    Prediction decoding logic verified.")

    # ==========================================
    # 3. Verify Dataset Loading
    # ==========================================
    print("\n[3] Verifying Dataset Loading...")

    # Instantiate Training Dataset
    # This will trigger cache creation if not present
    train_ds = GestureDataset(split="train")
    print(f"    Training Dataset Size: {len(train_ds)} windows")

    if len(train_ds) > 0:
        x, y, sid = train_ds[0]
        print(f"    Sample 0 Shapes - Input: {x.shape}, Label: {y.shape}, ID: {sid}")

        # Validation checks
        assert x.dim() == 2, "Input should be 2D (Time, Features)"
        assert (
            x.shape[1] == Config.INPUT_DIM
        ), f"Feature dim mismatch. Expected {Config.INPUT_DIM}, got {x.shape[1]}"
        assert y.dim() == 1, "Label should be 1D (Time,)"
        assert (
            x.shape[0] == y.shape[0]
        ), "Time dimension mismatch between input and label"

        print("    Dataset shapes verified.")
    else:
        print("    Warning: Dataset is empty. Skipping shape verification.")

    # ==========================================
    # 4. Verify Model Architecture
    # ==========================================
    print("\n[4] Verifying Model Architecture...")

    model = LSMCN()
    model.eval()

    # Create dummy input batch: (Batch=2, Time=64, Features=INPUT_DIM)
    dummy_input = torch.randn(2, 64, Config.INPUT_DIM)

    with torch.no_grad():
        p1, p2, p3 = model(dummy_input)

    print(f"    Model Output Shapes: P1={p1.shape}, P2={p2.shape}, P3={p3.shape}")

    assert p1.shape == (2, 64, Config.NUM_CLASSES), "Stage 1 output shape mismatch"
    assert p2.shape == (2, 64, Config.NUM_CLASSES), "Stage 2 output shape mismatch"
    assert p3.shape == (2, 64, Config.NUM_CLASSES), "Stage 3 output shape mismatch"

    # Check probability properties (sum to 1)
    sum_probs = p3[0, 0, :].sum().item()
    assert np.isclose(
        sum_probs, 1.0, atol=1e-5
    ), f"Probabilities do not sum to 1: {sum_probs}"

    print("    Model forward pass verified.")

    # ==========================================
    # 5. Run Training Loop (Simulated)
    # ==========================================
    print("\n[5] Running Training Loop (1 Epoch)...")

    trainer = Trainer()

    # Ensure the model is on the correct device
    print(f"    Training on device: {trainer.device}")

    # Run training
    trainer.train(num_epochs=Config.NUM_EPOCHS)

    # Check if best model was saved
    if os.path.exists(trainer.best_model_path):
        print(f"    Success: Best model saved at {trainer.best_model_path}")
    else:
        # If validation score didn't improve (unlikely with inf init), or 0 epochs ran
        print(
            "    Note: Best model file not found (might be due to no validation improvement or logic)."
        )

    # ==========================================
    # 6. Run Inference / Prediction
    # ==========================================
    print("\n[6] Running Inference on Test Set...")

    trainer.predict_test()

    # Check submission file
    if os.path.exists(Config.SUBMISSION_FILE):
        print(f"    Success: Submission file created at {Config.SUBMISSION_FILE}")

        # Validate content format
        with open(Config.SUBMISSION_FILE, "r") as f:
            lines = f.readlines()
            print(f"    Submission File Header/First Line: {lines[0].strip()}")

            # Check basic format: SessionID,label,label...
            parts = lines[0].strip().split(",")
            assert len(parts) >= 1, "Submission line should at least have SessionID"
            assert (
                "Session" in parts[0] or "Sample" in parts[0]
            ), "First part should be SessionID"

        print("    Submission format verified.")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    # Set fixed seed for the entire run
    Config.seed_everything(42)
    run_demo()
