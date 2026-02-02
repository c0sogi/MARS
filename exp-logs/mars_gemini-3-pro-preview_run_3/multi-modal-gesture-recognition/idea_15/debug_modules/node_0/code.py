import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import (
    set_seed,
    rle_encode,
    levenshtein_distance,
    compute_normalized_levenshtein,
)
from library.data_loader import get_dataloaders, get_test_loader, GestureDataset
from library.model import RSKARN
from library.trainer import Trainer
from library.inference import generate_predictions


def main():
    print("=== RSK-ARN Library Demo Execution ===\n")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("[1] Setting up configuration for demo...")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SIZE = 10  # Use only 10 samples for the demo
    Config.NUM_EPOCHS = 1  # Train for just 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = Config.WORKING_DIR
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Create directories
    Config.setup_directories()

    # Set seed for reproducibility
    set_seed(42)
    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG}")
    print("    Configuration setup complete.\n")

    # ---------------------------------------------------------
    # 2. Verify Utilities
    # ---------------------------------------------------------
    print("[2] Verifying utility functions...")

    # Test RLE Encode
    # Sequence: 0 (BG), 1, 1, 1, 0, 2, 2, 0, 3 -> Expected: [1, 2, 3]
    raw_preds = [0, 1, 1, 1, 0, 2, 2, 0, 3]
    encoded = rle_encode(raw_preds)
    assert encoded == [1, 2, 3], f"RLE Encode failed. Expected [1, 2, 3], got {encoded}"
    print("    rle_encode: Passed")

    # Test Levenshtein Distance
    seq1 = [1, 2, 3]
    seq2 = [1, 3]  # Deletion of '2' -> Distance 1
    dist = levenshtein_distance(seq1, seq2)
    assert dist == 1, f"Levenshtein distance failed. Expected 1, got {dist}"
    print("    levenshtein_distance: Passed")

    # Test Normalized Metric
    norm_dist = compute_normalized_levenshtein([seq1], [seq2])
    # Distance 1, Truth Length 2 -> 0.5
    assert norm_dist == 0.5, f"Normalized metric failed. Expected 0.5, got {norm_dist}"
    print("    compute_normalized_levenshtein: Passed\n")

    # ---------------------------------------------------------
    # 3. Data Loading
    # ---------------------------------------------------------
    print("[3] Testing Data Loading Pipeline...")

    # Instantiate DataLoaders
    # This triggers cache generation if not present
    train_loader, val_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    )

    # Verify Train Loader
    try:
        data, target, indices = next(iter(train_loader))
        print(f"    Train Batch Shape: X={data.shape}, Y={target.shape}")

        # Check dimensions
        # X: (Batch, Window, Input_Dim)
        assert data.shape == (
            Config.BATCH_SIZE,
            Config.WINDOW_SIZE,
            Config.INPUT_DIM,
        ), f"Input shape mismatch. Expected {(Config.BATCH_SIZE, Config.WINDOW_SIZE, Config.INPUT_DIM)}, got {data.shape}"

        # Y: (Batch, Window) - Frame-wise labels
        assert target.shape == (
            Config.BATCH_SIZE,
            Config.WINDOW_SIZE,
        ), f"Target shape mismatch. Expected {(Config.BATCH_SIZE, Config.WINDOW_SIZE)}, got {target.shape}"

        print("    Data loading and augmentation: Passed")
    except StopIteration:
        raise RuntimeError("Train loader is empty! Check dataset paths or debug size.")

    # ---------------------------------------------------------
    # 4. Model Initialization & Forward Pass
    # ---------------------------------------------------------
    print("\n[4] Testing Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RSKARN().to(device)

    # Create dummy input
    dummy_input = torch.randn(2, Config.WINDOW_SIZE, Config.INPUT_DIM).to(device)

    # Forward pass
    outputs = model(dummy_input)

    # Verify outputs
    expected_keys = ["logits_1", "logits_2", "logits_3", "probs_3"]
    for key in expected_keys:
        assert key in outputs, f"Model output missing key: {key}"

    # Check shape of final logits: (Batch, Window, Num_Classes)
    logits_shape = outputs["logits_3"].shape
    expected_shape = (2, Config.WINDOW_SIZE, Config.NUM_CLASSES)
    assert (
        logits_shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {logits_shape}"

    print(f"    Model forward pass successful. Output shape: {logits_shape}")

    # ---------------------------------------------------------
    # 5. Training Loop
    # ---------------------------------------------------------
    print("\n[5] Executing Training Loop (1 Epoch)...")

    trainer = Trainer(model, train_loader, val_loader, learning_rate=1e-3)

    # Run fit
    trainer.fit(epochs=Config.NUM_EPOCHS, patience=1)

    # Verify checkpoint creation
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not saved!"
    print(f"    Training complete. Checkpoint saved at: {best_model_path}")

    # ---------------------------------------------------------
    # 6. Inference Pipeline
    # ---------------------------------------------------------
    print("\n[6] Running Inference Pipeline...")

    # Run generation
    generate_predictions(
        checkpoint_path=best_model_path,
        batch_size=Config.BATCH_SIZE,
        debug=Config.DEBUG,
    )

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not generated!"

    # Check content format
    df_sub = pd.read_csv(Config.SUBMISSION_FILE, header=None)
    print(f"    Submission generated with {len(df_sub)} rows.")

    if len(df_sub) > 0:
        sample_row = df_sub.iloc[0]
        # First column should be SessionID (string), subsequent are gestures (ints/floats)
        session_id = sample_row[0]
        assert isinstance(
            session_id, str
        ), "First column of submission should be SessionID string."
        print(f"    Sample prediction: {session_id} -> {sample_row.values[1:]}")

    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    main()
