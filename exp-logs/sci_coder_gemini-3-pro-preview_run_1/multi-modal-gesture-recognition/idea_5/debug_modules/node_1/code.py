import os
import sys
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 1. Import Library Modules
import library.config
import library.utils
import library.graph_layers
import library.data_loader
import library.model
import library.train

# 2. Configuration Patching for Demo
# We patch the imported variables to ensure the demo runs quickly and uses a separate working directory.
DEMO_WORKING_DIR = "./working/demo_run"
DEMO_CACHE_DIR = os.path.join(DEMO_WORKING_DIR, "cache_demo")

# Patch library.config
library.config.DEBUG = True
library.config.DEBUG_SUBSET_SIZE = 5
library.config.BATCH_SIZE = 2
library.config.NUM_EPOCHS = 1
library.config.WORKING_DIR = DEMO_WORKING_DIR
library.config.CACHE_DIR = DEMO_CACHE_DIR

# Patch library.data_loader (since it uses 'from library.config import ...')
library.data_loader.DEBUG = True
library.data_loader.DEBUG_SUBSET_SIZE = 5
library.data_loader.CACHE_DIR = DEMO_CACHE_DIR
# Force stats path to be inside the demo working dir
library.data_loader.MultimodalDataset.stats_path = os.path.join(
    DEMO_WORKING_DIR, "stats.npz"
)

# Patch library.train
library.train.DEBUG = True
library.train.BATCH_SIZE = 2
library.train.NUM_EPOCHS = 1
library.train.WORKING_DIR = DEMO_WORKING_DIR

# Ensure demo directories exist
os.makedirs(DEMO_WORKING_DIR, exist_ok=True)
os.makedirs(DEMO_CACHE_DIR, exist_ok=True)


def set_seeds(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_utils():
    print("\n=== Testing Utils ===")

    # Test Levenshtein Distance
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 4]
    dist = library.utils.compute_levenshtein(seq1, seq2)
    # Distance should be 1 (substitution of 3 -> 4)
    assert dist == 1, f"Expected Levenshtein distance 1, got {dist}"
    print("Levenshtein check passed.")

    # Test RLE Decode
    # Sequence: 1, 1, 1, 1, 1 (valid), 0, 0 (bg), 2, 2, 2, 2, 2 (valid)
    # min_len=5, bg_class=0
    raw_preds = np.array([1, 1, 1, 1, 1, 0, 0, 2, 2, 2, 2, 2])
    decoded = library.utils.rle_decode(raw_preds, bg_class=0, min_len=5)
    assert decoded == [1, 2], f"Expected [1, 2], got {decoded}"
    print("RLE Decode check passed.")

    # Test Post Process
    # Create dummy probabilities favoring class 1 then class 2
    # Shape: (Time=10, Classes=3)
    probs = np.zeros((10, 3))
    probs[:5, 1] = 1.0  # First 5 frames class 1
    probs[5:, 2] = 1.0  # Last 5 frames class 2

    # Post process with window=1 (no smoothing) and min_len=5
    seq = library.utils.post_process_output(probs, window_size=1, min_len=5, bg_class=0)
    assert seq == [1, 2], f"Expected [1, 2] from post_process, got {seq}"
    print("Post-process check passed.")


def test_graph_layers():
    print("\n=== Testing Graph Layers ===")

    B, T, V, C = 2, 10, library.config.NUM_JOINTS, 3
    out_channels = 16

    # Instantiate AdaptiveGraphConv
    layer = library.graph_layers.AdaptiveGraphConv(
        in_channels=C, out_channels=out_channels
    )

    # Create dummy input
    x = torch.randn(B, T, V, C)

    # Forward pass
    y = layer(x)

    # Check output shape: (B, T, V, Out_Channels)
    expected_shape = (B, T, V, out_channels)
    assert y.shape == expected_shape, f"Expected shape {expected_shape}, got {y.shape}"
    print("AdaptiveGraphConv forward pass successful.")


def test_data_loader():
    print("\n=== Testing Data Loader ===")

    # Initialize Dataset (Train split)
    # This will trigger cache preparation for the subset defined by DEBUG_SUBSET_SIZE
    ds = library.data_loader.MultimodalDataset(split="train", load_cached_data=True)

    # Verify subset size
    print(f"Dataset size (Debug): {len(ds)}")
    assert (
        len(ds) <= library.config.DEBUG_SUBSET_SIZE
    ), "Dataset size exceeds debug limit."

    if len(ds) > 0:
        # Fetch one sample
        skel, audio, labels = ds[0]

        # Verify shapes
        # Skeleton: (T, 20, 3)
        assert (
            skel.dim() == 3
            and skel.shape[1] == library.config.NUM_JOINTS
            and skel.shape[2] == library.config.JOINT_CHANNELS
        )
        # Audio: (T, 13)
        assert audio.dim() == 2 and audio.shape[1] == library.config.MFCC_N_MFCC
        # Labels: (T,)
        assert labels.dim() == 1 and labels.shape[0] == skel.shape[0]

        print(
            f"Sample shapes verified: Skel {skel.shape}, Audio {audio.shape}, Labels {labels.shape}"
        )

    # Test Collate Function via DataLoader
    loader = DataLoader(ds, batch_size=2, collate_fn=library.data_loader.pad_collate)
    for batch in loader:
        padded_skel, padded_audio, padded_labels, mask, lengths = batch

        # Check batch dimension
        assert padded_skel.shape[0] == 2 or padded_skel.shape[0] == len(ds)
        # Check mask matches lengths
        assert mask.sum() == lengths.sum()

        print("DataLoader batch collation verified.")
        break


def test_model_and_training():
    print("\n=== Testing Model and Training Loop ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate Model
    model = library.model.AGGRN().to(device)

    # Create dummy batch for model verification
    B, T = 2, 50
    dummy_skel = torch.randn(
        B, T, library.config.NUM_JOINTS, library.config.JOINT_CHANNELS
    ).to(device)
    dummy_audio = torch.randn(B, T, library.config.MFCC_N_MFCC).to(device)
    dummy_lengths = torch.tensor([50, 40]).to(device)  # Second sample padded

    # Forward pass
    logits = model(dummy_skel, dummy_audio, dummy_lengths)

    # Check output: (B, T, NUM_CLASSES)
    assert logits.shape == (
        B,
        T,
        library.config.NUM_CLASSES,
    ), f"Expected logits shape {(B, T, library.config.NUM_CLASSES)}, got {logits.shape}"
    print("Model forward pass successful.")

    # Initialize Dataloaders (using patched config)
    # We manually create datasets to ensure they pick up the patched config values correctly
    train_ds = library.data_loader.MultimodalDataset(split="train")
    val_ds = library.data_loader.MultimodalDataset(split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=library.config.BATCH_SIZE,
        shuffle=True,
        collate_fn=library.data_loader.pad_collate,
        drop_last=False,  # Allow smaller last batch for demo
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=library.config.BATCH_SIZE,
        shuffle=False,
        collate_fn=library.data_loader.pad_collate,
    )

    # Initialize Trainer
    trainer = library.train.Trainer(model, device, train_loader, val_loader)

    # Run fit (1 epoch as patched)
    print("Starting training demo...")
    best_ler = trainer.fit(num_epochs=library.config.NUM_EPOCHS, patience=1)

    # Verify artifact creation
    best_model_path = os.path.join(DEMO_WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "best_model.pth was not created."
    print(f"Training demo finished. Best LER: {best_ler}")
    print(f"Model saved to: {best_model_path}")


if __name__ == "__main__":
    set_seeds(42)

    try:
        test_utils()
        test_graph_layers()
        test_data_loader()
        test_model_and_training()
        print("\nAll demonstrations completed successfully.")
    except AssertionError as e:
        print(f"\nValidation Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
