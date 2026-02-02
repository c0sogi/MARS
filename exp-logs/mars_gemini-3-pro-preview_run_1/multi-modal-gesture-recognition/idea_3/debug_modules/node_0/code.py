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
    levenshtein_distance,
    compute_normalized_levenshtein,
    decode_predictions,
)
from library.data_loader import GestureDataset, collate_fn
from library.model import MSDIGModel
from library.train import train_model, evaluate


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup Configuration for Demo
    print("\n[Step 1] Configuring Environment...")
    # Override Config for a fast demo run
    Config.WORKING_DIR = os.path.join(Config.ROOT_DIR, "working", "demo_run")
    Config.CACHE_TRAIN_DIR = os.path.join(Config.WORKING_DIR, "cache_train")
    Config.CACHE_VAL_DIR = os.path.join(Config.WORKING_DIR, "cache_val")
    Config.CACHE_TEST_DIR = os.path.join(Config.WORKING_DIR, "cache_test")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model_demo.pth")
    Config.STATS_PATH = os.path.join(Config.WORKING_DIR, "stats.npz")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    # Set hyperparameters for speed
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead/issues in demo

    # Initialize environment (creates directories)
    Config.setup_environment()
    set_seed(Config.SEED)
    print("Configuration updated and directories created.")

    # 2. Verify Utility Functions
    print("\n[Step 2] Verifying Utility Functions...")

    # Test Levenshtein
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist_eq = levenshtein_distance(seq1, seq2)
    assert (
        dist_eq == 0
    ), f"Levenshtein distance for identical sequences should be 0, got {dist_eq}"

    seq3 = [1, 2]
    dist_diff = levenshtein_distance(seq1, seq3)
    assert (
        dist_diff == 1
    ), f"Levenshtein distance for [1,2,3] vs [1,2] should be 1, got {dist_diff}"
    print("Levenshtein distance check passed.")

    # Test Metric
    preds = [[1, 2], [3, 4]]
    gts = [[1, 2], [3, 5]]  # 1 error in second sequence
    # Total dist = 0 + 1 = 1. Total len = 2 + 2 = 4. Score = 0.25
    score = compute_normalized_levenshtein(preds, gts)
    assert (
        abs(score - 0.25) < 1e-6
    ), f"Metric calculation failed. Expected 0.25, got {score}"
    print("Metric calculation check passed.")

    # Test Decoding
    # Create dummy logits: (Time=10, Classes=21).
    # Let's make class 5 have high probability
    dummy_logits = torch.zeros(10, Config.NUM_CLASSES)
    dummy_logits[:, 5] = 10.0
    decoded = decode_predictions(dummy_logits)
    # Config.MIN_GESTURE_LENGTH is 5. So 10 frames of class 5 should result in [5]
    print(f"Decoded sequence: {decoded}")
    assert isinstance(decoded, list), "decode_predictions should return a list"
    assert decoded == [5], f"Expected decoded sequence [5], got {decoded}"

    # 3. Verify Data Loading
    print("\n[Step 3] Verifying Data Loading...")

    # Use a small limit to speed up
    limit_samples = 8

    # Initialize Dataset (Train)
    # This will trigger stats computation on the first 8 samples
    train_ds = GestureDataset(
        split="train", limit=limit_samples, force_compute_stats=True
    )
    print(f"Train dataset size: {len(train_ds)}")
    assert (
        len(train_ds) == limit_samples
    ), f"Expected {limit_samples} samples, got {len(train_ds)}"

    # Fetch one sample
    sample = train_ds[0]
    # Check keys
    required_keys = {"skeleton", "audio", "labels", "sample_id"}
    assert required_keys.issubset(
        sample.keys()
    ), f"Sample missing keys. Got {sample.keys()}"

    # Check shapes
    skel = sample["skeleton"]
    audio = sample["audio"]
    lbl = sample["labels"]

    print(
        f"Sample 0 Shapes - Skeleton: {skel.shape}, Audio: {audio.shape}, Labels: {lbl.shape}"
    )
    assert skel.dim() == 2 and skel.shape[1] == Config.SKELETON_INPUT_SIZE
    assert audio.dim() == 2 and audio.shape[1] == Config.AUDIO_INPUT_SIZE
    assert lbl.dim() == 1
    assert skel.shape[0] == audio.shape[0] == lbl.shape[0]

    # Test Collate
    batch_list = [train_ds[i] for i in range(4)]
    batch = collate_fn(batch_list)
    assert batch is not None
    assert "mask" in batch
    assert "lengths" in batch
    print(
        f"Batch Shapes - Skeleton: {batch['skeleton'].shape}, Mask: {batch['mask'].shape}"
    )
    assert batch["skeleton"].shape[0] == 4

    # 4. Verify Model Architecture
    print("\n[Step 4] Verifying Model Architecture...")
    model = MSDIGModel()
    model.eval()

    # Create dummy batch inputs
    # Batch=2, Time=50
    dummy_skel = torch.randn(2, 50, Config.SKELETON_INPUT_SIZE)
    dummy_audio = torch.randn(2, 50, Config.AUDIO_INPUT_SIZE)
    dummy_lens = torch.tensor([50, 50])

    with torch.no_grad():
        logits = model(dummy_skel, dummy_audio, dummy_lens)

    print(f"Model Output Shape: {logits.shape}")
    assert logits.shape == (2, 50, Config.NUM_CLASSES), "Model output shape mismatch"
    print("Model forward pass successful.")

    # 5. Integration Test: Training Loop
    print("\n[Step 5] Running Training Integration Test...")
    # train_model uses the Config we modified (epochs=2, batch=4)
    # We pass limit to restrict dataset size
    best_error = train_model(limit=limit_samples, epochs=Config.NUM_EPOCHS)

    print(f"Training finished. Best Error Rate: {best_error}")
    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not saved."

    # 6. Inference Demonstration
    print("\n[Step 6] Running Inference on Test Subset...")

    # Load Test Data (Subset)
    test_ds = GestureDataset(split="test", limit=4)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # Load Model
    device = torch.device(Config.DEVICE)
    model = MSDIGModel().to(device)
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    results = []

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue

            skel = batch["skeleton"].to(device)
            audio = batch["audio"].to(device)
            lengths = batch["lengths"].to(device)
            ids = batch["sample_ids"]

            logits = model(skel, audio, lengths)

            # Decode
            batch_preds = decode_predictions(logits)

            for sample_id, pred_seq in zip(ids, batch_preds):
                # Convert list of ints to comma-separated string
                pred_str = ",".join(map(str, pred_seq))
                results.append(f"{sample_id},{pred_str}")

    # Save Submission
    with open(Config.SUBMISSION_PATH, "w") as f:
        for line in results:
            f.write(line + "\n")

    print(f"Inference complete. Submission generated at {Config.SUBMISSION_PATH}")
    print("First few lines of submission:")
    for line in results[:3]:
        print(line)

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
