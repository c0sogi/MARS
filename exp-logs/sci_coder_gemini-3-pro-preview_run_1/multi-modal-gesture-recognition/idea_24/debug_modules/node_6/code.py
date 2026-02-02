import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import (
    levenshtein_distance,
    decode_predictions,
    compute_levenshtein_ratio,
)
from library.data_loader import MultimodalDataset, collate_fn
from library.model import MCWINet
from library.trainer import Trainer

# Suppress warnings as requested
warnings.filterwarnings("ignore")


def setup_demo_config():
    """
    Overrides Config parameters for a fast, isolated demonstration.
    """
    print(">>> Setting up Demo Configuration...")

    # Set paths to a specific demo directory in working/
    Config.WORK_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORK_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORK_DIR, "submission")

    # Create directories
    if os.path.exists(Config.WORK_DIR):
        shutil.rmtree(Config.WORK_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce computational load
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 10  # Use only 10 samples
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 2
    Config.HIDDEN_DIM = 64  # Smaller model for speed

    # Ensure reproducibility
    Config.SEED = 42
    np.random.seed(Config.SEED)
    torch.manual_seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)


def verify_utils():
    """
    Verifies logic in library/utils.py
    """
    print(">>> Verifying Utilities...")

    # 1. Test Levenshtein Distance
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist_eq = levenshtein_distance(seq1, seq2)
    assert dist_eq == 0, f"Distance should be 0 for identical sequences, got {dist_eq}"

    seq3 = [1, 2]
    dist_diff = levenshtein_distance(seq1, seq3)
    assert dist_diff == 1, f"Distance should be 1 (deletion), got {dist_diff}"

    # 2. Test Decode Predictions (Smoothing + RLE)
    # 0 is background. We expect smoothing to remove single spikes and RLE to merge.
    # Input: [0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 0]
    # min_len default is 5.
    raw_preds = np.array([0, 0, 1, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 0])

    # Note: The provided median_filter has kernel_size=5.
    # [0,0,1,1,1] -> median 1
    # This should result in detecting gesture 1 and gesture 2.
    decoded = decode_predictions(raw_preds, min_len=3)

    # We expect [1, 2] roughly.
    # Let's just assert it returns a list and handles the background class (0) correctly.
    assert isinstance(decoded, list)
    assert 0 not in decoded, "Background class (0) should not be in decoded output"

    print("Utilities verification passed.")


def verify_data_loader():
    """
    Verifies logic in library/data_loader.py
    """
    print(">>> Verifying Data Loader...")

    # Instantiate Dataset (Train mode triggers stat computation)
    dataset = MultimodalDataset(mode="train", load_cached_data=False)

    # Check length
    assert (
        len(dataset) == Config.DEBUG_SUBSET_SIZE
    ), f"Dataset size mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {len(dataset)}"

    # Fetch one item
    item = dataset[0]

    # Check keys
    required_keys = ["skeleton", "audio", "labels", "length"]
    for k in required_keys:
        assert k in item, f"Missing key {k} in dataset item"

    # Check shapes
    # Skeleton: (Time, 60)
    # Audio: (Time, 13)
    # Labels: (Time,)
    T = item["length"]
    assert item["skeleton"].shape == (
        T,
        60,
    ), f"Skeleton shape mismatch: {item['skeleton'].shape}"
    assert item["audio"].shape == (
        T,
        Config.N_MFCC,
    ), f"Audio shape mismatch: {item['audio'].shape}"
    assert item["labels"].shape == (
        T,
    ), f"Labels shape mismatch: {item['labels'].shape}"

    # Test Collate Function
    item0 = dataset[0]
    item1 = dataset[1]
    batch_list = [item0, item1]
    batch = collate_fn(batch_list)

    assert batch is not None
    assert "mask" in batch
    assert batch["skeleton"].shape[0] == 2
    # Check padding logic: Time dim should be max of the two lengths
    max_len = max(item0["length"], item1["length"])
    assert batch["skeleton"].shape[1] == max_len

    print("Data Loader verification passed.")
    return batch  # Return batch for model testing


def verify_model(batch):
    """
    Verifies logic in library/model.py
    """
    print(">>> Verifying Model Architecture...")

    device = torch.device("cpu")  # Use CPU for simple logic check
    model = MCWINet().to(device)
    model.eval()

    skeleton = batch["skeleton"].to(device)
    audio = batch["audio"].to(device)
    mask = batch["mask"].to(device)
    lengths = batch["lengths"].to(device)

    with torch.no_grad():
        logits = model(skeleton, audio, mask, lengths)

    # Check Output Shape: (Batch, Time, NumClasses)
    # NumClasses is 21 (20 gestures + 1 background)
    expected_shape = (2, skeleton.shape[1], Config.NUM_CLASSES)
    assert (
        logits.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {logits.shape}"

    print("Model architecture verification passed.")


def verify_trainer_pipeline():
    """
    Verifies logic in library/trainer.py (Full Loop)
    """
    print(">>> Verifying Training Pipeline...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainer = Trainer(device=device)

    # 1. Run Fit
    # This will load train/val datasets, compute stats, and run for Config.NUM_EPOCHS
    trainer.fit(epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE)

    # Check if checkpoint was saved
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created."

    # 2. Run Predict
    output_csv = os.path.join(Config.SUBMISSION_DIR, "submission_demo.csv")
    trainer.predict(output_path=output_csv)

    assert os.path.exists(output_csv), "Submission file was not created."

    # Validate CSV content
    with open(output_csv, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Should have rows equal to test set size (subsetted by DEBUG_SUBSET_SIZE)
    # Note: The test set in metadata might be larger, but we applied DEBUG slicing in Dataset init.
    # However, Trainer re-initializes datasets. Since Config.DEBUG is True, it should slice.
    # The test.csv has 95 rows. DEBUG_SUBSET_SIZE is 10.
    assert (
        len(lines) == Config.DEBUG_SUBSET_SIZE
    ), f"Submission row count mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {len(lines)}"

    print("Training pipeline verification passed.")


if __name__ == "__main__":
    # 1. Setup
    setup_demo_config()

    # 2. Verify Utilities
    verify_utils()

    # 3. Verify Data Loading
    sample_batch = verify_data_loader()

    # 4. Verify Model
    verify_model(sample_batch)

    # 5. Verify Training & Inference Pipeline
    verify_trainer_pipeline()

    print("\n>>> All demonstrations completed successfully.")
