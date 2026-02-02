import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, decode_predictions
from library.data_loader import get_dataloaders
from library.model import MPCNet
from library.train import train_epoch, validate


def run_demo():
    # 1. Setup and Configuration
    print(">>> Setting up demonstration...")
    set_seed(42)

    # Override Config for speed
    Config.BATCH_SIZE = 2
    Config.NUM_EPOCHS = 1
    Config.DEBUG_SUBSET_SIZE = 4  # Only use 4 samples for train/val/test
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission_demo.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading Verification
    print("\n>>> Verifying Data Loading...")
    train_loader, val_loader, test_loader = get_dataloaders(
        debug_subset_size=Config.DEBUG_SUBSET_SIZE
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    skeleton = batch["skeleton"]
    audio = batch["audio"]
    labels = batch["labels"]
    lengths = batch["length"]

    print(
        f"Batch shapes - Skeleton: {skeleton.shape}, Audio: {audio.shape}, Labels: {labels.shape}"
    )

    # Assertions
    # Skeleton: (B, T, 60)
    assert skeleton.dim() == 3
    assert skeleton.shape[2] == Config.SKELETON_CHANNELS
    # Audio: (B, T, 20)
    assert audio.dim() == 3
    assert audio.shape[2] == Config.N_MFCC
    # Labels: (B, T)
    assert labels.dim() == 2
    # Lengths: (B)
    assert lengths.dim() == 1
    assert lengths.shape[0] == Config.BATCH_SIZE

    print("Data Loading verified successfully.")

    # 3. Model Initialization and Forward Pass
    print("\n>>> Verifying Model Architecture...")
    model = MPCNet().to(device)

    # Move batch to device
    skel_dev = skeleton.to(device)
    audio_dev = audio.to(device)
    len_dev = lengths.to(device)

    # Forward pass
    logits = model(skel_dev, audio_dev, len_dev)
    print(f"Logits shape: {logits.shape}")

    # Assertions
    # Output: (B, T, NumClasses)
    assert logits.dim() == 3
    assert logits.shape[0] == Config.BATCH_SIZE
    assert (
        logits.shape[1] == skeleton.shape[1]
    )  # Time dimension should match padded input
    assert logits.shape[2] == Config.NUM_CLASSES

    print("Model Forward Pass verified successfully.")

    # 4. Decoder Logic Verification
    print("\n>>> Verifying Decoding Logic...")
    # Create synthetic probabilities:
    # T=30.
    # 0-9: Class 1 (Vattene)
    # 10-19: Class 0 (Background)
    # 20-29: Class 2 (Vieniqui)
    # This should decode to [1, 2]

    T_synth = 30
    synth_logits = torch.zeros((1, T_synth, Config.NUM_CLASSES))
    # Set high logits for specific classes
    # Class 1
    synth_logits[0, 0:10, 1] = 10.0
    # Class 0 (Background) - implicit since others are 0? No, set explicitly high
    synth_logits[0, 10:20, 0] = 10.0
    # Class 2
    synth_logits[0, 20:30, 2] = 10.0

    synth_probs = torch.softmax(synth_logits, dim=2).numpy()

    decoded_seq = decode_predictions(synth_probs[0])
    print(f"Decoded Sequence: {decoded_seq}")

    assert decoded_seq == [1, 2], f"Expected [1, 2], got {decoded_seq}"
    print("Decoding Logic verified successfully.")

    # 5. Training Loop Demonstration
    print("\n>>> Verifying Training Loop...")

    # Setup optimizer and criterion
    weights = torch.ones(Config.NUM_CLASSES).to(device)
    weights[Config.BACKGROUND_CLASS_ID] = Config.BACKGROUND_WEIGHT_VALUE
    criterion = torch.nn.CrossEntropyLoss(
        weight=weights, label_smoothing=Config.LABEL_SMOOTHING, reduction="mean"
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Run 1 epoch
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # Run validation
    val_ler = validate(model, val_loader, device)
    print(f"Validation LER: {val_ler:.4f}")
    assert 0.0 <= val_ler, "LER should be non-negative"  # LER can be > 1.0

    print("Training loop verified successfully.")

    # 6. Submission Generation Simulation
    print("\n>>> Verifying Submission Generation...")

    # Simulate prediction on test set
    model.eval()
    results = []
    test_df = test_loader.dataset.df
    current_idx = 0

    with torch.no_grad():
        for batch in test_loader:
            skel_b = batch["skeleton"].to(device)
            audio_b = batch["audio"].to(device)
            len_b = batch["length"].to(device)

            logits_b = model(skel_b, audio_b, len_b)
            probs_b = torch.softmax(logits_b, dim=2).cpu().numpy()
            len_b_np = len_b.cpu().numpy()

            batch_size = len(probs_b)

            # Match IDs (simplified logic for demo, assuming order is maintained or handled)
            # In the real script, sorting is handled. Here we just take the slice.
            # Note: The real script sorts by length. We will mimic the real script logic strictly.
            df_slice = test_df.iloc[current_idx : current_idx + batch_size].copy()
            df_slice["sort_len"] = df_slice["num_frames"]
            df_slice_sorted = df_slice.sort_values(
                by="sort_len", ascending=False, kind="mergesort"
            )

            for i in range(batch_size):
                valid_l = len_b_np[i]
                sample_p = probs_b[i, :valid_l, :]
                pred_s = decode_predictions(sample_p)
                pred_str = ",".join(map(str, pred_s))

                sid = df_slice_sorted.iloc[i]["sample_id"]
                results.append((sid, pred_str))

            current_idx += batch_size

    # Write to file
    with open(Config.SUBMISSION_PATH, "w") as f:
        for sid, pred in results:
            f.write(f"{sid},{pred}\n")

    assert os.path.exists(Config.SUBMISSION_PATH)
    print(f"Submission file generated at {Config.SUBMISSION_PATH}")

    # Check content
    with open(Config.SUBMISSION_PATH, "r") as f:
        lines = f.readlines()
        print(f"First 2 lines of submission:\n{''.join(lines[:2])}")
        assert len(lines) > 0

    print("\n>>> Demonstration Completed Successfully!")


if __name__ == "__main__":
    run_demo()
