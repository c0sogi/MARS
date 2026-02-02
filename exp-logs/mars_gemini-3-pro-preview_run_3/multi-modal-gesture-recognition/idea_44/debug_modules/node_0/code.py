import os
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import (
    compute_levenshtein,
    run_length_encoding,
    filter_short_segments,
    decode_predictions_to_labels,
    calculate_score,
)
from library.dataset import GestureDataset
from library.model import RCMCN
from library.train import Trainer
from library.predict import generate_submission


def main():
    print("=== Starting Demo Execution ===")

    # ==========================================
    # 1. Setup & Configuration Override
    # ==========================================
    DEMO_DIR = "./working/demo_execution"

    # Clean previous run
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)

    # Create directories
    os.makedirs(DEMO_DIR, exist_ok=True)
    META_DIR = os.path.join(DEMO_DIR, "metadata")
    CACHE_DIR = os.path.join(DEMO_DIR, "cache")  # Note: Dataset uses Config.CACHE_DIR
    CKPT_DIR = os.path.join(DEMO_DIR, "checkpoints")
    SUB_DIR = os.path.join(DEMO_DIR, "submission")

    os.makedirs(META_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(SUB_DIR, exist_ok=True)

    print(f"Working directory set to: {DEMO_DIR}")

    # Create Mini Metadata (Subset of real data)
    # We read the original metadata and take top 5 rows
    orig_train_csv = "./metadata/train.csv"
    orig_val_csv = "./metadata/val.csv"
    orig_test_csv = "./metadata/test.csv"

    # Helper to create mini csv
    def create_mini_csv(src, dst, n=5):
        if os.path.exists(src):
            df = pd.read_csv(src)
            df_mini = df.head(n).copy()
            df_mini.to_csv(dst, index=False)
            return len(df_mini)
        return 0

    n_train = create_mini_csv(orig_train_csv, os.path.join(META_DIR, "train.csv"), n=6)
    n_val = create_mini_csv(orig_val_csv, os.path.join(META_DIR, "val.csv"), n=4)
    n_test = create_mini_csv(orig_test_csv, os.path.join(META_DIR, "test.csv"), n=4)

    print(f"Created mini datasets: Train={n_train}, Val={n_val}, Test={n_test}")

    # Override Config
    print("Overriding Config parameters for demo...")
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = CACHE_DIR  # Dataset uses this
    Config.CHECKPOINT_DIR = CKPT_DIR
    Config.SUBMISSION_DIR = SUB_DIR

    Config.TRAIN_CSV = os.path.join(META_DIR, "train.csv")
    Config.VAL_CSV = os.path.join(META_DIR, "val.csv")
    Config.TEST_CSV = os.path.join(META_DIR, "test.csv")

    Config.BEST_MODEL_PATH = os.path.join(CKPT_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(SUB_DIR, "submission.csv")

    Config.EPOCHS = 2
    Config.BATCH_SIZE = 2
    Config.setup_dirs()  # Re-ensure dirs exist based on new paths
    Config.set_seed(42)

    # ==========================================
    # 2. Verify Utility Functions
    # ==========================================
    print("\n=== Verifying Utils ===")

    # Test Levenshtein
    seq1 = [1, 2, 3]
    seq2 = [1, 2]
    dist = compute_levenshtein(seq1, seq2)
    assert dist == 1, f"Levenshtein failed: expected 1, got {dist}"
    print("Levenshtein check passed.")

    # Test RLE
    # 0, 0, 1, 1, 1, 2, 2, 0
    preds = np.array([0, 0, 1, 1, 1, 2, 2, 0])
    segments = run_length_encoding(preds)
    # Expected: [{'label':0, ...}, {'label':1, ...}, {'label':2, ...}, {'label':0, ...}]
    assert len(segments) == 4, f"RLE failed: expected 4 segments, got {len(segments)}"
    assert (
        segments[1]["label"] == 1 and segments[1]["end"] - segments[1]["start"] + 1 == 3
    ), "RLE segment 1 incorrect"
    print("RLE check passed.")

    # Test Filter Short Segments (Min duration 5 in Config, let's test logic)
    # Create segments manually
    test_segs = [
        {"label": 1, "start": 0, "end": 10},  # Len 11 -> Keep
        {"label": 2, "start": 11, "end": 12},  # Len 2 -> Drop (if min=5)
    ]
    filtered = filter_short_segments(test_segs, min_duration=5)
    assert len(filtered) == 1, "Filter short segments failed"
    assert filtered[0]["label"] == 1, "Wrong segment kept"
    print("Segment filtering check passed.")

    # ==========================================
    # 3. Verify Dataset & DataLoader
    # ==========================================
    print("\n=== Verifying Dataset ===")

    # Initialize Dataset (Train)
    # This will trigger caching
    train_ds = GestureDataset(split="train", load_cached_data=False, transform=True)

    # Check basic properties
    assert len(train_ds) > 0, "Train dataset is empty"
    print(f"Train dataset size (windows): {len(train_ds)}")

    # Check item structure
    features, labels = train_ds[0]
    # Features: (WindowSize, InputDim)
    # Labels: (WindowSize,)
    print(f"Sample Feature Shape: {features.shape}")
    print(f"Sample Label Shape: {labels.shape}")

    assert features.shape[0] == Config.WINDOW_SIZE, "Feature time dimension mismatch"
    assert (
        features.shape[1] == Config.INPUT_DIM
    ), f"Feature input dimension mismatch. Expected {Config.INPUT_DIM}, got {features.shape[1]}"
    assert labels.shape[0] == Config.WINDOW_SIZE, "Label time dimension mismatch"

    # Initialize DataLoader
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    batch_feat, batch_lbl = next(iter(train_loader))
    assert batch_feat.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    print("DataLoader check passed.")

    # ==========================================
    # 4. Verify Model Architecture
    # ==========================================
    print("\n=== Verifying Model ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RCMCN().to(device)

    # Dummy input: (Batch, Time, InputDim)
    # Time can be variable, but for training it's WindowSize
    dummy_input = torch.randn(2, Config.WINDOW_SIZE, Config.INPUT_DIM).to(device)

    # Forward pass
    out1, out2, out3 = model(dummy_input)

    # Check shapes: (Batch, Time, NumClasses)
    expected_shape = (2, Config.WINDOW_SIZE, Config.NUM_CLASSES)
    assert out1.shape == expected_shape, f"Stage 1 output shape mismatch: {out1.shape}"
    assert out3.shape == expected_shape, f"Stage 3 output shape mismatch: {out3.shape}"

    print("Model forward pass check passed.")

    # ==========================================
    # 5. Verify Training Loop
    # ==========================================
    print("\n=== Verifying Training Loop ===")

    # Setup Val Loader
    val_ds = GestureDataset(split="val", load_cached_data=False, transform=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    trainer = Trainer(model, train_loader, val_loader, device)

    # Run fit
    # We set epochs=2 in Config override
    print(f"Training for {Config.EPOCHS} epochs on mini dataset...")
    trainer.fit(Config.EPOCHS)

    # Check if model saved
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint not found after training"
    print("Training loop completed and model saved.")

    # ==========================================
    # 6. Verify Prediction & Submission
    # ==========================================
    print("\n=== Verifying Prediction ===")

    # The generate_submission function loads Config.TEST_CSV and Config.BEST_MODEL_PATH
    # We have already set these.

    # Note: generate_submission instantiates GestureDataset('test').
    # We need to make sure it doesn't fail on cache.
    # We'll pass load_cached_data=False to force processing of the mini test csv.

    generate_submission(load_cached_data=False)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not generated"

    # Verify content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH, header=None)
    # Should have n_test rows
    assert (
        len(sub_df) == n_test
    ), f"Submission row count mismatch. Expected {n_test}, got {len(sub_df)}"

    print(f"Submission generated at {Config.SUBMISSION_PATH}")
    print("First few rows of submission:")
    print(sub_df.head())

    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    main()
