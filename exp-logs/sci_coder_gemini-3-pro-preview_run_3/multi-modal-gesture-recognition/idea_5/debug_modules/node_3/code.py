import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil
import json
import time

# Import provided libraries
from library.config import Config
from library.utils import decode_predictions, compute_levenshtein
from library.models import GestureNet, DualStreamEncoder, RefinementTCN
from library.losses import CombinedLoss, SmoothingLoss
from library.dataset import process_dataset, GestureDataset, get_datasets
from library.trainer import Trainer


def setup_demo_environment():
    """
    Overrides Config parameters for a quick demo run and creates mini-metadata files.
    """
    print("=== Setting up Demo Environment ===")

    # 1. Override Config parameters for speed
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.OUTPUT_DIR = os.path.join(Config.WORKING_DIR, "outputs")
    Config.SUBMISSION_DIR = Config.WORKING_DIR

    Config.MODEL_SAVE_PATH = os.path.join(Config.OUTPUT_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Ensure directories exist
    Config.setup_directories()

    # Invalidate stale cache files to force data reprocessing (Cite debug_lesson_4)
    # This prevents loading empty datasets created by previous failed runs.
    if os.path.exists(Config.WORKING_DIR):
        for filename in os.listdir(Config.WORKING_DIR):
            if filename.endswith(".npz"):
                os.remove(os.path.join(Config.WORKING_DIR, filename))
                print(f"Removed stale cache: {filename}")

    # 2. Create Mini Metadata (Subset of real data)
    # We read the original metadata and take the top 5 samples for train/val/test

    def create_mini_csv(source_path, dest_name, n=5):
        if not os.path.exists(source_path):
            print(f"Warning: Source {source_path} not found. Creating dummy.")
            return

        df = pd.read_csv(source_path)
        # Take top n
        df_mini = df.head(n)
        dest_path = os.path.join(Config.WORKING_DIR, dest_name)
        df_mini.to_csv(dest_path, index=False)
        print(f"Created mini metadata: {dest_path} with {len(df_mini)} samples.")
        return dest_path

    # Update Config paths to point to mini CSVs
    Config.TRAIN_CSV = create_mini_csv(
        os.path.join("./metadata", "train.csv"), "mini_train.csv", n=6
    )
    Config.VAL_CSV = create_mini_csv(
        os.path.join("./metadata", "val.csv"), "mini_val.csv", n=6
    )
    Config.TEST_CSV = create_mini_csv(
        os.path.join("./metadata", "test.csv"), "mini_test.csv", n=6
    )


def test_utils():
    print("\n=== Testing Utils ===")

    # Test decode_predictions
    # Scenario: Background(0) -> Class 1 -> Class 1 -> Class 1 -> Class 1 -> Class 1 -> Background(0)
    # Threshold is 5. Length of Class 1 is 5. Should be detected.
    probs = np.zeros((7, Config.NUM_CLASSES))
    probs[0, 0] = 1.0  # BG
    probs[1:6, 1] = 1.0  # Class 1 (5 frames)
    probs[6, 0] = 1.0  # BG

    decoded = decode_predictions(probs, threshold=5)
    print(f"Decoded sequence: {decoded}")
    assert decoded == [1], f"Expected [1], got {decoded}"

    # Test filtering (length < threshold)
    probs_short = np.zeros((5, Config.NUM_CLASSES))
    probs_short[1:4, 2] = 1.0  # Class 2 (3 frames)
    decoded_short = decode_predictions(probs_short, threshold=5)
    print(f"Decoded short sequence: {decoded_short}")
    assert decoded_short == [], f"Expected [], got {decoded_short}"

    # Test compute_levenshtein
    seq1 = [1, 2, 3]
    seq2 = [1, 2]  # Deletion of 1
    # Dist = 1, Total Len = 2. Score = 0.5
    score = compute_levenshtein([seq1], [seq2])
    print(f"Levenshtein Score: {score}")
    assert score == 0.5, f"Expected 0.5, got {score}"

    print("Utils verification passed.")


def test_models():
    print("\n=== Testing Models ===")

    batch_size = 2
    seq_len = 32
    static_dim = 73
    dynamic_dim = 120

    # Instantiate Model
    model = GestureNet().to(Config.DEVICE)
    model.eval()

    # Create dummy inputs
    static_x = torch.randn(batch_size, seq_len, static_dim).to(Config.DEVICE)
    dynamic_x = torch.randn(batch_size, seq_len, dynamic_dim).to(Config.DEVICE)

    # Forward pass
    with torch.no_grad():
        s1_logits, s2_logits = model(static_x, dynamic_x)

    print(f"Stage 1 Logits Shape: {s1_logits.shape}")
    print(f"Stage 2 Logits Shape: {s2_logits.shape}")

    # Assertions
    assert s1_logits.shape == (batch_size, seq_len, Config.NUM_CLASSES)
    assert s2_logits.shape == (batch_size, seq_len, Config.NUM_CLASSES)

    print("Model verification passed.")


def test_losses():
    print("\n=== Testing Losses ===")

    criterion = CombinedLoss().to(Config.DEVICE)

    batch_size = 2
    seq_len = 16
    num_classes = Config.NUM_CLASSES

    # Dummy Logits: (B, C, T) as expected by Loss forward (permuted inside trainer usually)
    # But CombinedLoss expects (B, C, T)
    s1_logits = torch.randn(
        batch_size, num_classes, seq_len, device=Config.DEVICE, requires_grad=True
    )
    s2_logits = torch.randn(
        batch_size, num_classes, seq_len, device=Config.DEVICE, requires_grad=True
    )

    # Dummy Targets: (B, T)
    targets = torch.randint(0, num_classes, (batch_size, seq_len)).to(Config.DEVICE)

    # Compute Loss
    loss, ce1, ce2, smooth = criterion(s1_logits, s2_logits, targets)

    print(f"Total Loss: {loss.item():.4f}")
    print(f"CE1: {ce1.item():.4f}, CE2: {ce2.item():.4f}, Smooth: {smooth.item():.4f}")

    # Assertions
    assert loss.item() > 0
    assert not torch.isnan(loss)

    # Test Backward
    loss.backward()
    print("Backward pass successful.")

    print("Loss verification passed.")


def run_integration_pipeline():
    print("\n=== Running Integration Pipeline (Trainer) ===")

    # Initialize Trainer
    trainer = Trainer()

    # 1. Train
    print("Starting Training...")
    try:
        trainer.fit()
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e

    # Verify Model Saved
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Model checkpoint found at {Config.MODEL_SAVE_PATH}")
    else:
        raise FileNotFoundError("Model checkpoint was not saved.")

    # 2. Predict
    print("Starting Prediction...")
    try:
        trainer.predict()
    except Exception as e:
        print(f"Prediction failed with error: {e}")
        raise e

    # Verify Submission
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file found at {Config.SUBMISSION_PATH}")
        # Print first few lines
        with open(Config.SUBMISSION_PATH, "r") as f:
            print("Submission head:")
            for i, line in enumerate(f):
                print(line.strip())
                if i >= 2:
                    break
    else:
        raise FileNotFoundError("Submission file was not saved.")

    print("Integration pipeline completed successfully.")


if __name__ == "__main__":
    # Set seeds
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    # 1. Setup
    setup_demo_environment()

    # 2. Unit Tests
    test_utils()
    test_models()
    test_losses()

    # 3. Integration Test
    # Note: This will process data, train for 2 epochs, and predict.
    # It relies on the mini-datasets created in setup.
    run_integration_pipeline()
