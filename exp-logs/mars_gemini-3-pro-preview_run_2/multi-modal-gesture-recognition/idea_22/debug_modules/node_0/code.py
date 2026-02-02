import os
import torch
import pandas as pd
import numpy as np
import shutil
import time
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import set_seed, compute_levenshtein
from library.data_loader import load_data, GestureDataset, collate_fn
from library.model import DSG_CRCN
from library.loss import DeepSupervisionLoss
from library.train import decode_sequence


def run_demo():
    # 1. Setup and Configuration
    print("[*] Setting up demo environment...")
    set_seed(Config.SEED)

    # Define a temporary directory for this demo run
    demo_dir = os.path.join(Config.WORKING_DIR, "demo_run")
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config paths to use the demo directory
    # This prevents overwriting the main cache or checkpoints
    Config.CACHE_DIR = os.path.join(demo_dir, "cache")
    Config.CHECKPOINT_DIR = os.path.join(demo_dir, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 2. Prepare Mini-Dataset
    print("[*] Preparing mini-dataset for speed...")
    # Read the provided training metadata
    train_meta_path = Config.TRAIN_METADATA
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"Metadata not found at {train_meta_path}")

    df = pd.read_csv(train_meta_path)
    # Select top 4 samples for a quick batch test
    mini_df = df.head(4).copy()
    mini_meta_path = os.path.join(demo_dir, "mini_train.csv")
    mini_df.to_csv(mini_meta_path, index=False)

    # 3. Test Data Loading
    print("[*] Testing Data Loading and Processing...")
    # load_data handles parsing of .mat files (skeleton) and audio
    # We pass 'mini_train' as subset name so it creates 'mini_train_data.npz' in our demo cache
    data_list = load_data("mini_train", mini_meta_path, load_cached_data=False)

    # Verify data loading
    assert len(data_list) == 4, f"Expected 4 samples, got {len(data_list)}"
    sample = data_list[0]
    assert "skeleton" in sample
    assert "audio" in sample
    assert "labels" in sample
    assert sample["skeleton"].shape[1] == 20, "Skeleton should have 20 joints"
    assert (
        sample["skeleton"].shape[2] == 3
    ), "Skeleton should have 3 coordinates (x,y,z)"

    # Instantiate Dataset
    dataset = GestureDataset(data_list, is_train=True)

    # Verify Dataset __getitem__
    item = dataset[0]
    # Check feature dimension:
    # (12 joints * 3 pos) + (12 joints * 3 vel) + 13 MFCC = 36 + 36 + 13 = 85
    expected_dim = Config.INPUT_DIM
    assert (
        item["features"].shape[1] == expected_dim
    ), f"Expected feature dim {expected_dim}, got {item['features'].shape[1]}"
    assert item["targets"].dtype == torch.long
    assert item["boundaries"].dtype == torch.float

    # Instantiate DataLoader
    batch_size = 2
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    # Fetch one batch
    batch = next(iter(loader))
    features, targets, boundaries, mask, ids = batch

    print(f"    Batch shapes: Features={features.shape}, Targets={targets.shape}")
    assert features.shape[0] == batch_size
    assert features.shape[2] == expected_dim
    assert mask.shape == (batch_size, features.shape[1])

    # 4. Test Model Forward Pass
    print("[*] Testing Model Architecture...")
    device = Config.DEVICE
    model = DSG_CRCN().to(device)

    # Move batch to device
    features = features.to(device)
    targets = targets.to(device)
    boundaries = boundaries.to(device)
    mask = mask.to(device)

    # Forward pass
    outputs = model(features, mask)

    # Verify Outputs
    # Model returns a dictionary with outputs from 3 stages
    assert "final_cls" in outputs
    assert "stage1_cls" in outputs
    assert "stage3_bnd" in outputs

    # Check shape of final classification output: (B, T, NumClasses)
    final_cls = outputs["final_cls"]
    assert final_cls.shape == (batch_size, features.shape[1], Config.NUM_CLASSES)

    # 5. Test Loss Calculation
    print("[*] Testing Loss Function...")
    criterion = DeepSupervisionLoss().to(device)

    loss, metrics = criterion(outputs, targets, boundaries, mask)

    print(f"    Total Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.requires_grad, "Loss should require gradients for backprop"
    assert (
        "loss_s3_cls" in metrics
    ), "Metrics should contain stage 3 classification loss"

    # 6. Test Utilities (Decoding & Metrics)
    print("[*] Testing Utilities...")

    # Test decode_sequence
    # Scenario: 0=Background, 1=GestureA, 2=GestureB
    # Input indices with noise: 0 0 1 1 1 0 2 2 0
    # Expected output: [1, 2] (after removing background and collapsing repeats)
    mock_indices = np.array([0, 0, 1, 1, 1, 0, 2, 2, 0])
    decoded_seq = decode_sequence(mock_indices, kernel_size=3)
    print(f"    Decoded Sequence: {decoded_seq}")
    assert decoded_seq == [1, 2], f"Decoding failed. Expected [1, 2], got {decoded_seq}"

    # Test compute_levenshtein
    # Perfect match
    score_perfect = compute_levenshtein([[1, 2]], [[1, 2]])
    assert score_perfect == 0.0, "Levenshtein should be 0 for perfect match"

    # Mismatch
    # Pred: [1], Target: [1, 2] -> Distance is 1 (insertion/deletion), Length is 2 -> Score 0.5
    score_mismatch = compute_levenshtein([[1]], [[1, 2]])
    assert np.isclose(score_mismatch, 0.5), f"Expected 0.5, got {score_mismatch}"

    print("[+] All demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
