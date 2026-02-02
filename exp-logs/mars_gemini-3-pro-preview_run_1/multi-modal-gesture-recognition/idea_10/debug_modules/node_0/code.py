import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

# Import from the provided library
import library.config as config
import library.utils as utils
from library.data_loader import GestureDataset, collate_fn
from library.model import DGR_RN
from library.train import train_one_epoch, validate


def run_demo():
    # ==========================================
    # 1. Setup and Configuration Override
    # ==========================================
    print(">>> Setting up Demo Environment...")

    # Define demo directories
    DEMO_DIR = "./working/demo_run"
    DEMO_CACHE_DIR = os.path.join(DEMO_DIR, "cache_demo")
    DEMO_CHECKPOINT_DIR = os.path.join(DEMO_DIR, "checkpoints")

    os.makedirs(DEMO_DIR, exist_ok=True)
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)
    os.makedirs(DEMO_CHECKPOINT_DIR, exist_ok=True)

    # Override config paths to use demo directories
    # We monkey-patch the config module to ensure library functions use these paths where applicable
    config.CACHE_DIR = DEMO_CACHE_DIR
    config.CHECKPOINT_DIR = DEMO_CHECKPOINT_DIR
    config.BEST_MODEL_PATH = os.path.join(DEMO_CHECKPOINT_DIR, "best_model.pth")

    # Set seed
    utils.set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 2. Data Preparation (Subsets)
    # ==========================================
    print("\n>>> Preparing Data Subsets...")

    # Load original metadata
    train_df = pd.read_csv(config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(config.VAL_METADATA_PATH)
    test_df = pd.read_csv(config.TEST_METADATA_PATH)

    # Create subsets (e.g., 6 samples each for batch_size=2 to have 3 batches)
    subset_size = 6
    train_subset = train_df.head(subset_size).copy()
    val_subset = val_df.head(subset_size).copy()
    test_subset = test_df.head(subset_size).copy()

    # Save subsets
    train_subset_path = os.path.join(DEMO_DIR, "train_subset.csv")
    val_subset_path = os.path.join(DEMO_DIR, "val_subset.csv")
    test_subset_path = os.path.join(DEMO_DIR, "test_subset.csv")

    train_subset.to_csv(train_subset_path, index=False)
    val_subset.to_csv(val_subset_path, index=False)
    test_subset.to_csv(test_subset_path, index=False)

    print(f"Created subsets with {subset_size} samples each.")

    # ==========================================
    # 3. Verify Utility Functions
    # ==========================================
    print("\n>>> Verifying Utility Functions...")

    # Test Levenshtein Distance
    dist = utils.levenshtein_distance([1, 2, 3], [1, 2, 3])
    assert dist == 0, f"Expected distance 0, got {dist}"
    dist = utils.levenshtein_distance([1, 2], [1, 2, 3])
    assert dist == 1, f"Expected distance 1 (insertion), got {dist}"
    dist = utils.levenshtein_distance([1, 5, 3], [1, 2, 3])
    assert dist == 1, f"Expected distance 1 (substitution), got {dist}"
    print("Levenshtein Distance: OK")

    # Test RLE Decode
    # Sequence: 0, 0, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 0
    # Min duration 5.
    # 1s duration = 5 -> Keep. 2s duration = 5 -> Keep.
    raw_preds = [0, 0, 1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2, 0]
    decoded = utils.rle_decode(raw_preds, background_id=0, min_duration=5)
    assert decoded == [1, 2], f"Expected [1, 2], got {decoded}"

    # Test filtering short segments
    # 3s duration = 3 -> Drop
    raw_preds_short = [0, 0, 3, 3, 3, 0, 0]
    decoded_short = utils.rle_decode(raw_preds_short, background_id=0, min_duration=5)
    assert decoded_short == [], f"Expected [], got {decoded_short}"
    print("RLE Decode: OK")

    # Test Median Filter
    # Input: [0, 0, 10, 0, 0] -> Median window 3 -> [0, 0, 0, 0, 0] (roughly)
    data = np.array([0, 0, 10, 0, 0])
    filtered = utils.median_filter(data, window_size=3)
    # Median of [0,0,10] is 0.
    assert filtered[2] == 0, f"Expected median filtered value 0, got {filtered[2]}"
    print("Median Filter: OK")

    # ==========================================
    # 4. Data Loader Demonstration
    # ==========================================
    print("\n>>> Initializing Data Loaders...")

    stats_path = os.path.join(DEMO_DIR, "stats.npz")

    # Initialize Dataset (this will compute stats on the subset and cache data)
    train_dataset = GestureDataset(
        metadata_path=train_subset_path,
        split="train",
        transform=True,
        stats_path=stats_path,
    )

    # Check single item
    sample = train_dataset[0]
    assert "skeleton" in sample
    assert "audio" in sample
    assert "labels" in sample
    assert sample["skeleton"].shape[1] == config.SKELETON_INPUT_DIM
    assert sample["audio"].shape[1] == config.AUDIO_INPUT_DIM
    print(
        f"Single Sample Shapes - Skel: {sample['skeleton'].shape}, Audio: {sample['audio'].shape}"
    )

    # Initialize DataLoader
    batch_size = 2
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    val_dataset = GestureDataset(
        metadata_path=val_subset_path,
        split="val",
        transform=False,
        stats_path=stats_path,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Check Batch
    batch = next(iter(train_loader))
    assert batch["skeleton"].dim() == 3  # (B, T, C)
    assert batch["mask"].dim() == 2  # (B, T)
    assert batch["skeleton"].shape[0] == batch_size
    print(
        f"Batch Shapes - Skel: {batch['skeleton'].shape}, Mask: {batch['mask'].shape}"
    )

    # ==========================================
    # 5. Model Demonstration
    # ==========================================
    print("\n>>> Initializing Model...")

    model = DGR_RN().to(device)

    # Dummy Forward Pass
    dummy_skel = torch.randn(2, 50, config.SKELETON_INPUT_DIM).to(device)
    dummy_audio = torch.randn(2, 50, config.AUDIO_INPUT_DIM).to(device)

    with torch.no_grad():
        logits = model(dummy_skel, dummy_audio)

    assert logits.shape == (
        2,
        50,
        config.NUM_CLASSES,
    ), f"Expected output shape (2, 50, {config.NUM_CLASSES}), got {logits.shape}"
    print("Model Forward Pass: OK")

    # ==========================================
    # 6. Training Loop Demonstration
    # ==========================================
    print("\n>>> Running Training Loop (2 Epochs)...")

    # Setup Loss and Optimizer
    weights = torch.ones(config.NUM_CLASSES).to(device)
    weights[config.BACKGROUND_CLASS_ID] = config.BACKGROUND_WEIGHT
    criterion = nn.CrossEntropyLoss(
        weight=weights, label_smoothing=config.LABEL_SMOOTHING
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Train for 2 epochs
    for epoch in range(1, 3):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_ler = validate(model, val_loader, device)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val LER: {val_ler:.4f}")

        # Verify loss is not NaN
        assert not np.isnan(train_loss), "Training loss is NaN"

    # Save Model
    torch.save(model.state_dict(), config.BEST_MODEL_PATH)
    assert os.path.exists(config.BEST_MODEL_PATH), "Model checkpoint not found"
    print(f"Model saved to {config.BEST_MODEL_PATH}")

    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
