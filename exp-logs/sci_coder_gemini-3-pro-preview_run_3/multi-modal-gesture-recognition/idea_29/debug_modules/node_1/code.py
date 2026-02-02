import os
import sys
import torch
import numpy as np
import random
import shutil
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import (
    levenshtein_distance,
    rle_encode,
    filter_short_segments,
    TruncatedMSELoss,
)
from library.data_loader import GestureDataset, get_dataloaders
from library.model import NGKRN
from library.trainer import Trainer


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def test_utils():
    print("=== Testing Utils ===")

    # 1. Levenshtein Distance
    seq1 = [1, 2, 3]
    seq2 = [1, 3]
    dist = levenshtein_distance(seq1, seq2)
    assert dist == 1, f"Levenshtein distance failed. Expected 1, got {dist}"
    print("Levenshtein distance: OK")

    # 2. RLE Encode
    # Should collapse duplicates and remove 0 (background)
    raw_preds = [0, 0, 1, 1, 1, 0, 2, 2, 0, 3]
    encoded = rle_encode(raw_preds)
    expected = [1, 2, 3]
    assert encoded == expected, f"RLE Encode failed. Expected {expected}, got {encoded}"
    print("RLE Encode: OK")

    # 3. Filter Short Segments
    # Min duration 3.
    # [1, 1, 1, 2, 2, 1, 1, 1] -> The '2, 2' (len 2) is short.
    # Logic in utils replaces with prev label (1) or next.
    # In utils.py: "replacement = segments[i-1]['label']" (if i>0)
    raw_seg = np.array([1, 1, 1, 2, 2, 1, 1, 1])
    filtered = filter_short_segments(raw_seg, min_duration=3)
    # The middle segment '2' is length 2. Previous label is 1. Should become 1.
    expected_seg = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(
        filtered, expected_seg, err_msg="Filter Short Segments failed"
    )
    print("Filter Short Segments: OK")

    # 4. Truncated MSE Loss
    # log_probs: (Batch, Time, Classes)
    # We simulate a jump greater than threshold
    criterion = TruncatedMSELoss(threshold=1.0)
    # Time 0: 0.0, Time 1: 2.0. Diff is 2.0. Sq is 4.0. Threshold sq is 1.0.
    # Loss should be 1.0
    t_input = torch.tensor([[[0.0], [2.0]]], dtype=torch.float32)  # (1, 2, 1)
    loss = criterion(t_input)
    assert (
        abs(loss.item() - 1.0) < 1e-5
    ), f"TruncatedMSELoss failed. Expected 1.0, got {loss.item()}"
    print("TruncatedMSELoss: OK")


def test_data_loader():
    print("\n=== Testing Data Loader ===")

    # Use debug mode to load only a few samples
    ds_train = GestureDataset("train", debug=True)
    print(f"Train Dataset Length (Windows): {len(ds_train)}")

    if len(ds_train) > 0:
        item = ds_train[0]
        feats = item["features"]
        lbls = item["labels"]

        # Check shapes
        # Features: (WindowSize, InputDim) -> (64, 193)
        assert feats.shape == (
            Config.WINDOW_SIZE,
            Config.INPUT_DIM,
        ), f"Feature shape mismatch. Expected {(Config.WINDOW_SIZE, Config.INPUT_DIM)}, got {feats.shape}"
        assert lbls.shape == (
            Config.WINDOW_SIZE,
        ), f"Label shape mismatch. Expected {(Config.WINDOW_SIZE,)}, got {lbls.shape}"
        print("Train Item Shapes: OK")

    ds_val = GestureDataset("val", debug=True)
    print(f"Val Dataset Length (Samples): {len(ds_val)}")
    if len(ds_val) > 0:
        item = ds_val[0]
        # Val returns full sequence, shape depends on T
        feats = item["features"]
        print(f"Val Item Feature Shape: {feats.shape}")
        assert feats.shape[1] == Config.INPUT_DIM, "Val feature dimension mismatch"
        print("Val Item Dimensions: OK")


def test_model():
    print("\n=== Testing Model Architecture ===")

    model = NGKRN()
    model.eval()

    # Create dummy input: (Batch, Time, InputDim)
    B, T, D = 2, 64, Config.INPUT_DIM
    dummy_input = torch.randn(B, T, D)

    with torch.no_grad():
        p1, p2, p3 = model(dummy_input)

    # Check outputs
    # All stages should output (Batch, Time, NumClasses)
    expected_shape = (B, T, Config.NUM_CLASSES)

    assert p1.shape == expected_shape, f"Stage 1 shape mismatch: {p1.shape}"
    assert p2.shape == expected_shape, f"Stage 2 shape mismatch: {p2.shape}"
    assert p3.shape == expected_shape, f"Stage 3 shape mismatch: {p3.shape}"

    # Check probability properties (sum to 1)
    sum_probs = p3.sum(dim=2)
    assert torch.allclose(
        sum_probs, torch.ones_like(sum_probs), atol=1e-5
    ), "Output probabilities do not sum to 1"

    print("Model Forward Pass: OK")


def run_trainer_demo():
    print("\n=== Running Trainer Demo ===")

    # Configure for speed
    Config.NUM_EPOCHS = 1
    Config.DEBUG_SAMPLE_SIZE = 6
    Config.BATCH_SIZE = 2
    Config.WORKING_DIR = "./working/demo_run"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_FILE = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Ensure directories exist
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Initialize Trainer with debug=True
    trainer = Trainer(debug=True)

    # 1. Fit
    print("Starting training...")
    trainer.fit()

    # Check if checkpoint exists
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created!"
    print(f"Checkpoint created at {checkpoint_path}")

    # 2. Predict
    print("Starting prediction...")
    trainer.predict()

    # Check if submission exists
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created!"

    # Validate submission content roughly
    with open(Config.SUBMISSION_FILE, "r") as f:
        lines = f.readlines()
        print(f"Submission generated with {len(lines)} lines.")
        if len(lines) > 0:
            print(f"Sample line: {lines[0].strip()}")

    print("Trainer Demo: OK")


def main():
    # Set seeds for reproducibility
    set_seeds(42)

    # Clear cache to ensure data is re-processed with fixed logic
    if os.path.exists(Config.CACHE_DIR):
        print(f"Clearing cache at {Config.CACHE_DIR}...")
        shutil.rmtree(Config.CACHE_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Run tests
    try:
        test_utils()
        test_data_loader()
        test_model()
        run_trainer_demo()
        print("\nAll demonstrations completed successfully.")
    except AssertionError as e:
        print(f"\nFAILURE: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
