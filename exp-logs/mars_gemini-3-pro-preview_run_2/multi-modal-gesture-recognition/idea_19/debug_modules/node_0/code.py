import os
import shutil
import pandas as pd
import torch
import numpy as np
import warnings

# Import library components
from library.config import Config
from library.utils import set_seed, compute_levenshtein_distance, decode_predictions
from library.data_loader import get_data_loaders
from library.model import RSG_CRCN
from library.trainer import Trainer


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # -------------------------------------------------------------------------
    print("Setting up demo configuration...")

    # Set reproducible seed
    set_seed(42)

    # Define temporary directories for demo speed
    demo_work_dir = "./working/demo_run"
    demo_meta_dir = os.path.join(demo_work_dir, "metadata")
    demo_cache_dir = os.path.join(demo_work_dir, "cache")
    demo_submission_dir = os.path.join(demo_work_dir, "submission")

    # Clean up previous runs if any
    if os.path.exists(demo_work_dir):
        shutil.rmtree(demo_work_dir)

    os.makedirs(demo_meta_dir, exist_ok=True)
    os.makedirs(demo_cache_dir, exist_ok=True)
    os.makedirs(demo_submission_dir, exist_ok=True)

    # Override Config to point to demo directories and reduce workload
    Config.METADATA_DIR = demo_meta_dir
    Config.CACHE_DIR = demo_cache_dir
    Config.SUBMISSION_DIR = demo_submission_dir

    # Reduce model complexity for speed
    Config.TCN_NUM_LAYERS = 2
    Config.TCN_DILATIONS = [1, 2]
    Config.LSTM_HIDDEN_SIZE = 64
    Config.TCN_CHANNELS = 64

    # Reduce training duration
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 2
    Config.DEBUG_SUBSET_SIZE = None  # We handle subsetting via metadata files

    # -------------------------------------------------------------------------
    # 2. Create Subset Metadata (Optimization for Speed)
    # -------------------------------------------------------------------------
    print("Creating subset metadata for fast loading...")

    # Read original metadata and save top N rows to demo metadata folder
    # This prevents the data loader from processing thousands of videos
    subset_size = 10

    for split in ["train", "val", "test"]:
        original_csv = f"./metadata/{split}.csv"
        if os.path.exists(original_csv):
            df = pd.read_csv(original_csv)
            # Take a small subset
            df_subset = df.head(subset_size)
            df_subset.to_csv(os.path.join(demo_meta_dir, f"{split}.csv"), index=False)
            print(f"  Created {split}.csv with {len(df_subset)} samples.")
        else:
            print(f"  Warning: {original_csv} not found.")

    # -------------------------------------------------------------------------
    # 3. Data Loading
    # -------------------------------------------------------------------------
    print("\nLoading data...")
    # load_cached_data=False forces processing of our new subset
    train_loader, val_loader, test_loader, test_ids = get_data_loaders(
        load_cached_data=False
    )

    # Verify Data Loader
    print("Verifying data loader...")
    try:
        batch = next(iter(train_loader))
        features, labels, boundaries, mask = batch

        # Check shapes
        # Features: (B, T, F)
        # Labels: (B, T)
        # Boundaries: (B, T)
        # Mask: (B, T)
        B, T, F = features.shape
        print(
            f"  Batch shapes - Features: {features.shape}, Labels: {labels.shape}, Mask: {mask.shape}"
        )

        assert (
            B == Config.BATCH_SIZE or B == subset_size
        ), f"Batch size mismatch. Expected <= {Config.BATCH_SIZE}, got {B}"
        assert labels.shape == (B, T), "Labels shape mismatch"
        assert mask.shape == (B, T), "Mask shape mismatch"
        assert features.dtype == torch.float32, "Features should be float32"
        assert labels.dtype == torch.long, "Labels should be long"

        # Verify feature dimension
        # (12 joints * 3 coords * 2 [pos+vel]) + 13 MFCC = 72 + 13 = 85
        expected_dim = (
            Config.NUM_JOINTS * Config.CHANNELS_PER_JOINT * 2
        ) + Config.AUDIO_MFCC_N_MFCC
        assert (
            F == expected_dim
        ), f"Feature dimension mismatch. Expected {expected_dim}, got {F}"

        print("  Data loader verification passed.")

    except StopIteration:
        print("  Error: Data loader is empty.")
        return

    # -------------------------------------------------------------------------
    # 4. Model Initialization & Verification
    # -------------------------------------------------------------------------
    print("\nInitializing model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    model = RSG_CRCN().to(device)

    # Verify Forward Pass
    print("Verifying model forward pass...")
    features = features.to(device)
    mask = mask.to(device)

    with torch.no_grad():
        outputs = model(features, mask)

    # Check output structure
    assert "stage1" in outputs
    assert "stage2" in outputs
    assert "stage3" in outputs

    s3_cls, s3_bnd = outputs["stage3"]
    # s3_cls: (B, NumClasses, T)
    assert s3_cls.shape == (
        B,
        Config.NUM_CLASSES,
        T,
    ), f"Output shape mismatch. Got {s3_cls.shape}"
    assert s3_bnd.shape == (
        B,
        1,
        T,
    ), f"Boundary output shape mismatch. Got {s3_bnd.shape}"

    print("  Model verification passed.")

    # -------------------------------------------------------------------------
    # 5. Training Loop
    # -------------------------------------------------------------------------
    print("\nStarting training loop...")
    trainer = Trainer(model, train_loader, val_loader, device)

    # Run fit (demonstrates training and validation)
    trainer.fit()

    # Check if best model was saved
    best_model_path = os.path.join(Config.SUBMISSION_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print(f"  Best model saved at: {best_model_path}")
    else:
        print("  Warning: Best model not found (might not have improved in 2 epochs).")

    # -------------------------------------------------------------------------
    # 6. Utility Functions Verification
    # -------------------------------------------------------------------------
    print("\nVerifying utility functions...")

    # Test Levenshtein Distance
    seq1 = [1, 2, 3]
    seq2 = [1, 3]  # Deletion of '2' -> Distance 1
    dist = compute_levenshtein_distance(seq1, seq2)
    assert dist == 1, f"Levenshtein distance incorrect. Expected 1, got {dist}"

    # Test Decode Predictions
    # Frame predictions: [0, 0, 1, 1, 1, 0, 2, 2, 0] -> Should collapse to [1, 2] (0 is background)
    raw_preds = np.array([0, 0, 1, 1, 1, 0, 2, 2, 0])
    decoded = decode_predictions(raw_preds)
    assert decoded == [1, 2], f"Decoding incorrect. Expected [1, 2], got {decoded}"

    print("  Utility verification passed.")
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
