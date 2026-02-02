import os
import torch
import numpy as np
import pandas as pd
import shutil
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import (
    set_seed,
    decode_predictions_rle,
    compute_levenshtein_score,
    post_process_sequence,
    apply_median_filter,
)
from library.data_loader import get_data_loaders
from library.model import RCGRNet
from library.train import Trainer


def run_demo():
    # ==========================================
    # 1. Setup & Configuration Override
    # ==========================================
    print(">>> Step 1: Configuring environment for demo...")

    # Override Config for a fast, isolated run
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 10  # Use only 10 samples
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 2
    Config.WORK_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORK_DIR, "cache_demo")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORK_DIR, "checkpoints")
    Config.SUBMISSION_DIR = Config.WORK_DIR

    # Create directories
    if os.path.exists(Config.WORK_DIR):
        shutil.rmtree(Config.WORK_DIR)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Running on device: {device}")
    print("    Configuration updated successfully.")

    # ==========================================
    # 2. Verify Utilities
    # ==========================================
    print("\n>>> Step 2: Verifying utility functions...")

    # Test RLE Decoding
    # Sequence: 0(BG), 0, 1(Gesture A), 1, 1, 1, 1, 0, 2(Gesture B), 2
    # Min length 3: Gesture A (len 5) kept, Gesture B (len 2) dropped
    dummy_seq = np.array([0, 0, 1, 1, 1, 1, 1, 0, 2, 2])
    decoded = decode_predictions_rle(dummy_seq, min_length=3)
    assert decoded == [1], f"RLE Decoding failed. Expected [1], got {decoded}"

    # Test Levenshtein Score
    # Pred: [1, 2], Target: [1, 3] -> Distance 1 (Substitute 2->3)
    # Score = Dist / Target_Len = 1 / 2 = 0.5
    lev_score = compute_levenshtein_score([[1, 2]], [[1, 3]])
    assert (
        lev_score == 0.5
    ), f"Levenshtein calculation failed. Expected 0.5, got {lev_score}"

    print("    Utilities verified successfully.")

    # ==========================================
    # 3. Data Loading
    # ==========================================
    print("\n>>> Step 3: Initializing Data Loaders...")

    # This will compute stats on the fly and cache processed tensors
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    )

    # Fetch one batch to verify
    try:
        skels, audios, labels, lengths = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError(
            "DataLoader returned no data. Check input data availability."
        )

    # Verify shapes
    # Skel: (B, T, 60), Audio: (B, T, 13), Labels: (B, T)
    assert (
        skels.ndim == 3 and skels.shape[2] == 60
    ), f"Skeleton shape mismatch: {skels.shape}"
    assert (
        audios.ndim == 3 and audios.shape[2] == 13
    ), f"Audio shape mismatch: {audios.shape}"
    assert labels.ndim == 2, f"Labels shape mismatch: {labels.shape}"
    assert lengths.shape[0] == Config.BATCH_SIZE, "Batch size mismatch in lengths"

    print(f"    Batch loaded. Skeleton: {skels.shape}, Audio: {audios.shape}")

    # ==========================================
    # 4. Model Initialization & Forward Pass
    # ==========================================
    print("\n>>> Step 4: Instantiating RCGRNet Model...")

    model = RCGRNet().to(device)

    # Move batch to device
    skels = skels.to(device)
    audios = audios.to(device)

    # Forward pass
    with torch.no_grad():
        logits = model(skels, audios)

    # Verify output shape: (B, T, NumClasses)
    assert logits.shape == (
        Config.BATCH_SIZE,
        skels.shape[1],
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, skels.shape[1], Config.NUM_CLASSES)}, got {logits.shape}"

    print(f"    Model forward pass successful. Output shape: {logits.shape}")

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("\n>>> Step 5: Running Training Loop (Trainer)...")

    trainer = Trainer(device=device)

    # Run fit (uses Config.EPOCHS=2)
    trainer.fit(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG)

    expected_ckpt = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(expected_ckpt), "Best model checkpoint was not saved."

    print(f"    Training complete. Checkpoint saved at: {expected_ckpt}")

    # ==========================================
    # 6. Inference & Validation
    # ==========================================
    print("\n>>> Step 6: Performing Inference & Evaluation...")

    # Load best model
    model.load_state_dict(torch.load(expected_ckpt, map_location=device))
    model.eval()

    val_preds = []
    val_targets = []

    with torch.no_grad():
        # Process one batch from validation
        skels, audios, labels, lengths = next(iter(val_loader))
        skels, audios = skels.to(device), audios.to(device)

        logits = model(skels, audios)
        probs = torch.softmax(logits, dim=2)

        for i in range(len(lengths)):
            length = lengths[i].item()

            # Slice to valid length
            p = probs[i, :length]
            t = labels[i, :length]

            # Post-process
            pred_seq = post_process_sequence(p, kernel_size=5, min_length=3)
            target_seq = decode_predictions_rle(t.cpu().numpy(), min_length=3)

            val_preds.append(pred_seq)
            val_targets.append(target_seq)

            print(f"    Sample {i}: Pred={pred_seq}, Target={target_seq}")

    # Compute final metric on this batch
    final_score = compute_levenshtein_score(val_preds, val_targets)
    print(f"    Batch Levenshtein Score: {final_score:.4f}")

    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
