import sys
import os
import torch
import numpy as np
import shutil

# Ensure the current directory is in the path to import the library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import (
    levenshtein_distance,
    compute_normalized_levenshtein,
    decode_sequence,
    apply_median_filter,
    post_process_and_decode,
)
from library.data_loader import GestureDataset, collate_fn
from library.model import FISGCN
from library.loss import FISGCNLoss
from library.train import train_model


def run_demo():
    print("=== Setting up Configuration ===")
    # Modify Config for a quick demo run
    # We use monkey-patching to adjust the static configuration for this run
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 10  # Process only 10 samples per split for speed
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Clean up demo directory if it exists to ensure a fresh run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    print("\n=== Testing Utils ===")
    # Test Levenshtein Distance
    seq1 = [1, 2, 3]
    seq2 = [1, 2]
    dist = levenshtein_distance(seq1, seq2)
    print(f"Levenshtein([1,2,3], [1,2]): {dist}")
    assert dist == 1, "Levenshtein distance calculation incorrect"

    # Test Normalized Levenshtein
    # Distance is 1, total GT gestures is 2. Result should be 0.5
    norm_dist = compute_normalized_levenshtein([seq1], [seq2])
    print(f"Normalized Levenshtein: {norm_dist}")
    assert abs(norm_dist - 0.5) < 1e-6, "Normalized Levenshtein incorrect"

    # Test Decode Sequence
    # Should collapse duplicates and remove background (0)
    raw_seq = [0, 0, 1, 1, 1, 0, 2, 2, 0, 0]
    decoded = decode_sequence(raw_seq, background_class_id=0)
    print(f"Decoded Sequence {raw_seq} -> {decoded}")
    assert decoded == [1, 2], "Sequence decoding incorrect"

    print("\n=== Testing Data Loader (Subset) ===")
    # Initialize dataset (this will trigger processing of 10 samples from raw files)
    # We use 'val' split as it is typically smaller, but debug limits it anyway.
    # load_cached_data=False ensures we test the raw processing logic.
    ds = GestureDataset(split="val", debug=True, load_cached_data=False)
    print(f"Dataset size: {len(ds)}")

    if len(ds) > 0:
        features, targets, sample_id = ds[0]
        print(f"Sample ID: {sample_id}")
        print(f"Features Shape: {features.shape}")
        print(f"Targets Shape: {targets.shape}")

        # Verify shapes
        assert features.ndim == 2, "Features should be (T, D)"
        assert (
            features.shape[1] == Config.INPUT_DIM
        ), f"Feature dim mismatch. Expected {Config.INPUT_DIM}, got {features.shape[1]}"
        assert targets.ndim == 1, "Targets should be (T,)"
        assert (
            features.shape[0] == targets.shape[0]
        ), "Temporal dimension mismatch between features and targets"
    else:
        print("Warning: Dataset is empty. Check input data availability.")

    print("\n=== Testing Model Forward Pass ===")
    model = FISGCN().to(Config.DEVICE)
    model.eval()

    # Create dummy input batch
    B, T = 2, 50
    dummy_features = torch.randn(B, T, Config.INPUT_DIM).to(Config.DEVICE)
    dummy_mask = torch.ones(B, T).to(Config.DEVICE)

    with torch.no_grad():
        outputs = model(dummy_features, dummy_mask)

    print(f"Model output stages: {len(outputs)}")
    # Output should correspond to Encoder + Refinement Stages
    assert (
        len(outputs) == 1 + Config.NUM_REFINEMENT_STAGES
    ), "Incorrect number of stages in output"

    last_stage = outputs[-1]
    cls_logits = last_stage["cls"]
    bnd_logits = last_stage["bnd"]

    print(f"Class Logits Shape: {cls_logits.shape}")
    assert cls_logits.shape == (B, T, Config.NUM_CLASSES), "Class logits shape mismatch"
    assert bnd_logits.shape == (B, T, 1), "Boundary logits shape mismatch"

    print("\n=== Testing Loss Function ===")
    criterion = FISGCNLoss().to(Config.DEVICE)
    dummy_targets = torch.randint(0, Config.NUM_CLASSES, (B, T)).to(Config.DEVICE)

    loss, metrics = criterion(outputs, dummy_targets, dummy_mask)
    print(f"Total Loss: {loss.item()}")
    print("Metrics:", metrics.keys())

    assert isinstance(loss, torch.Tensor), "Loss must be a tensor"
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    print("\n=== Running Full Training Loop (Demo) ===")
    # This runs the Trainer: fit (1 epoch) and predict on the test set
    # Using debug=True ensures it runs on a tiny subset
    train_model(debug=True)

    # Verify outputs
    submission_file = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    if os.path.exists(submission_file):
        print(f"Submission file generated at: {submission_file}")
        with open(submission_file, "r") as f:
            lines = f.readlines()
            print(f"Number of predictions: {len(lines)}")
            if len(lines) > 0:
                print(f"First line: {lines[0].strip()}")
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
