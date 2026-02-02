import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import (
    WORKING_DIR,
    METADATA_DIR,
    TRAIN_METADATA_PATH,
    NUM_CLASSES,
    NUM_JOINTS,
    JOINT_DIM,
    NUM_MFCC,
    SEED,
    BATCH_SIZE,
)
from library.utils import set_seed, levenshtein_distance, compute_error_rate
from library.data_loader import prepare_dataset, GestureDataset, collate_fn
from library.model import SSG_CRCN
from library.loss import CombinedLoss
from library.predict import decode_sequence, post_process_sequence


def create_mini_metadata(source_path, dest_path, num_samples=10):
    """Creates a smaller metadata file for quick demonstration purposes."""
    print(f"Creating mini metadata from {source_path} to {dest_path}...")
    df = pd.read_csv(source_path)
    mini_df = df.head(num_samples)
    mini_df.to_csv(dest_path, index=False)
    print(f"Created mini metadata with {len(mini_df)} samples.")


def test_utils():
    print("\n=== Testing Utils ===")
    # Test Levenshtein distance
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist_eq = levenshtein_distance(seq1, seq2)
    assert dist_eq == 0, f"Distance should be 0 for identical sequences, got {dist_eq}"

    seq3 = [1, 2]
    dist_diff = levenshtein_distance(seq1, seq3)
    assert dist_diff == 1, f"Distance should be 1 for deletion, got {dist_diff}"

    # Test Error Rate
    preds = [[1, 2], [3, 4, 5]]
    truths = [[1, 2], [3, 5]]  # Distance 0 and 1
    # Total distance = 1, Total length = 2 + 2 = 4. Error rate = 0.25
    error_rate = compute_error_rate(preds, truths)
    assert error_rate == 0.25, f"Expected error rate 0.25, got {error_rate}"
    print("Utils tests passed.")


def test_data_pipeline():
    print("\n=== Testing Data Pipeline ===")

    # 1. Create a mini training metadata file to speed up processing
    mini_train_path = os.path.join(WORKING_DIR, "mini_train.csv")
    if not os.path.exists(TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Original metadata not found at {TRAIN_METADATA_PATH}")

    create_mini_metadata(TRAIN_METADATA_PATH, mini_train_path, num_samples=4)

    # 2. Test prepare_dataset
    # We use a unique cache name to avoid conflicts with existing caches
    cache_name = "mini_train_data_demo"
    print("Running prepare_dataset...")
    # Force load_cached_data=False to verify processing logic
    positions, audios, labels, boundaries, ids = prepare_dataset(
        mini_train_path, cache_name, load_cached_data=False
    )

    assert len(positions) == 4
    assert len(ids) == 4
    print(f"Loaded {len(positions)} samples successfully.")

    # 3. Test Dataset and DataLoader
    dataset = GestureDataset(positions, audios, labels, boundaries, augment=True)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    # Fetch one batch
    feats, lbls, bnds, mask = next(iter(loader))

    # Verify shapes
    # feats: (B, InputDim, T_max)
    # InputDim = (NUM_JOINTS * JOINT_DIM * 2) + NUM_MFCC
    expected_dim = (NUM_JOINTS * JOINT_DIM * 2) + NUM_MFCC

    print(
        f"Batch shapes -> Feats: {feats.shape}, Labels: {lbls.shape}, Mask: {mask.shape}"
    )

    assert feats.shape[0] == 2
    assert feats.shape[1] == expected_dim
    assert lbls.shape == mask.shape
    assert bnds.shape == mask.shape

    print("Data pipeline tests passed.")
    return loader, expected_dim


def test_model_and_loss(loader, input_dim):
    print("\n=== Testing Model and Loss ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Instantiate Model
    model = SSG_CRCN().to(device)
    print("Model instantiated.")

    # 2. Instantiate Loss
    criterion = CombinedLoss().to(device)

    # 3. Instantiate Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # 4. Run a single training step
    model.train()

    # Get a batch
    feats, lbls, bnds, mask = next(iter(loader))
    feats = feats.to(device)
    lbls = lbls.to(device)
    bnds = bnds.to(device)
    mask = mask.to(device)

    # Forward Pass
    outputs = model(feats, mask)

    # Verify output structure
    # Model returns a list of outputs (one per stage)
    assert isinstance(outputs, list)
    assert len(outputs) == 3  # NUM_STAGES is 3 in config

    # Check shape of the final stage output: (B, NUM_CLASSES + 1, T)
    final_out = outputs[-1]
    assert final_out.shape[0] == feats.shape[0]
    assert final_out.shape[1] == NUM_CLASSES + 1
    assert final_out.shape[2] == feats.shape[2]

    print("Forward pass successful.")

    # Loss Calculation
    loss, metrics = criterion(outputs, lbls, bnds, mask)

    print(f"Calculated Loss: {loss.item():.4f}")
    print(f"Metrics: {metrics}")

    assert not torch.isnan(loss)
    assert loss.item() > 0

    # Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Backward pass and optimization step successful.")


def test_inference_logic():
    print("\n=== Testing Inference Logic ===")

    # Mock predictions: (T,) array of class indices
    # Sequence: Background(0) -> Gesture(1) -> Gesture(1) -> Background(0) -> Gesture(2)
    raw_preds = np.array([0, 0, 1, 1, 1, 0, 0, 2, 2, 0])

    # Test decode_sequence
    decoded = decode_sequence(raw_preds)
    expected = [1, 2]
    assert decoded == expected, f"Expected {expected}, got {decoded}"

    # Test post_process_sequence (includes median filter)
    # With a small kernel, it might smooth out short glitches
    # Let's create a noisy sequence
    noisy_preds = np.array([1, 1, 1, 2, 1, 1, 1])  # The '2' is a glitch
    # Note: MEDIAN_FILTER_KERNEL is imported from config, usually 7.
    # If kernel is 7, the single '2' in the middle of '1's should be filtered out.

    # We need a sequence longer than kernel size for valid testing if kernel is large
    # Let's assume kernel size 7 for this test logic check
    long_noisy = np.array([1] * 10 + [2] + [1] * 10)
    processed = post_process_sequence(long_noisy)

    # The glitch '2' should be removed if median filter works
    assert 2 not in processed, "Median filter failed to remove single-frame glitch"
    assert processed == [1], f"Expected [1], got {processed}"

    print("Inference logic tests passed.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(SEED)

    try:
        # 1. Test Utility Functions
        test_utils()

        # 2. Test Data Loading (creates mini dataset)
        loader, input_dim = test_data_pipeline()

        # 3. Test Model, Forward, Backward, Loss
        test_model_and_loss(loader, input_dim)

        # 4. Test Inference/Post-processing
        test_inference_logic()

        print("\nAll demonstrations and verifications completed successfully.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
