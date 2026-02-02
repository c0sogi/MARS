import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import (
    set_seed,
    compute_levenshtein_distance,
    calculate_levenshtein_accuracy,
    decode_predictions,
    median_filter_prediction,
    make_pad_mask,
)
from library.data_loader import SkeletonAudioDataset, collate_fn
from library.model import GMD_CRCN
from library.loss import CombinedLoss
from library.train import Trainer


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    print(">>> Setting up environment...")
    # Set seed for reproducibility
    set_seed(42)

    # Define paths for temporary mini-dataset
    mini_data_dir = os.path.join(Config.WORKING_DIR, "mini_demo")
    os.makedirs(mini_data_dir, exist_ok=True)

    mini_train_csv = os.path.join(mini_data_dir, "mini_train.csv")
    mini_val_csv = os.path.join(mini_data_dir, "mini_val.csv")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Utility Functions...")

    # Test Levenshtein Distance
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist_eq = compute_levenshtein_distance(seq1, seq2)
    assert dist_eq == 0, f"Distance should be 0, got {dist_eq}"

    seq3 = [1, 2]
    seq4 = [1, 3]
    dist_diff = compute_levenshtein_distance(seq3, seq4)
    assert dist_diff == 1, f"Distance should be 1 (substitution), got {dist_diff}"
    print(" - Levenshtein distance: OK")

    # Test Accuracy Calculation
    acc = calculate_levenshtein_accuracy([seq3], [seq4])
    # Distance is 1, Length of target (seq4) is 2. Error rate = 0.5
    assert acc == 0.5, f"Error rate should be 0.5, got {acc}"
    print(" - Levenshtein accuracy: OK")

    # Test Decoding
    # 0 is background. Repeats should be collapsed.
    raw_preds = [0, 0, 1, 1, 1, 0, 2, 2, 0, 3]
    decoded = decode_predictions(
        raw_preds, collapse_repeats=True, remove_background=True
    )
    expected = [1, 2, 3]
    assert decoded == expected, f"Decoded {decoded} != Expected {expected}"
    print(" - Prediction decoding: OK")

    # Test Median Filter
    noisy_seq = np.array([1, 1, 2, 1, 1])  # The '2' is noise in a window of 1s
    smoothed = median_filter_prediction(noisy_seq, kernel_size=3)
    # Median of [1,1,2] is 1. Median of [1,2,1] is 1. Median of [2,1,1] is 1.
    # Should smooth out the 2.
    assert smoothed[2] == 1, f"Median filter failed to smooth noise. Got {smoothed}"
    print(" - Median filter: OK")

    # -------------------------------------------------------------------------
    # 3. Create Mini Dataset for Speed
    # -------------------------------------------------------------------------
    print("\n>>> Creating Mini Dataset...")

    # Read original metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)

    # Take top 4 samples for speed
    df_mini_train = df_train.head(4).copy()
    df_mini_val = df_val.head(4).copy()

    df_mini_train.to_csv(mini_train_csv, index=False)
    df_mini_val.to_csv(mini_val_csv, index=False)
    print(
        f" - Created mini train ({len(df_mini_train)}) and val ({len(df_mini_val)}) CSVs."
    )

    # -------------------------------------------------------------------------
    # 4. Data Loader Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Testing Data Loader...")

    # Initialize Dataset with mini CSV
    # We use a temporary cache dir to avoid messing with real cache
    temp_cache_dir = os.path.join(mini_data_dir, "cache")
    os.makedirs(temp_cache_dir, exist_ok=True)

    train_ds = SkeletonAudioDataset(
        metadata_csv=mini_train_csv,
        input_dir=Config.INPUT_DIR,
        cache_dir=temp_cache_dir,
        split="train",
        load_cached_data=False,  # Force processing
    )

    assert len(train_ds) == 4, "Dataset length mismatch"

    # Check single item
    sample = train_ds[0]
    print(f" - Sample keys: {list(sample.keys())}")
    assert (
        "pos" in sample and "vel" in sample and "audio" in sample and "labels" in sample
    )
    assert sample["pos"].shape[1] == 36, "Incorrect position feature dimension"
    assert sample["audio"].shape[1] == 13, "Incorrect audio feature dimension"

    # Initialize DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=2, shuffle=False, collate_fn=collate_fn
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    print(f" - Batch keys: {list(batch.keys())}")
    print(
        f" - Batch shapes: Pos {batch['pos'].shape}, Audio {batch['audio'].shape}, Labels {batch['labels'].shape}"
    )

    assert batch["pos"].dim() == 3  # (B, T, D)
    assert batch["labels"].dim() == 2  # (B, T)
    print(" - Data Loading: OK")

    # -------------------------------------------------------------------------
    # 5. Model Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Testing Model Forward Pass...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" - Using device: {device}")

    model = GMD_CRCN().to(device)

    # Move batch to device
    pos = batch["pos"].to(device)
    vel = batch["vel"].to(device)
    audio = batch["audio"].to(device)
    lengths = batch["lengths"].to(device)
    labels = batch["labels"].to(device)

    # Forward pass
    outputs = model(pos, vel, audio, lengths)

    assert "stage1" in outputs
    assert "stage2" in outputs
    assert "stage3" in outputs

    # Check output shape: (Batch, NumClasses, Time)
    B, T = pos.shape[0], pos.shape[1]
    C = Config.NUM_CLASSES

    assert outputs["stage1"].shape == (
        B,
        C,
        T,
    ), f"Stage 1 shape mismatch: {outputs['stage1'].shape}"
    assert outputs["stage3"].shape == (
        B,
        C,
        T,
    ), f"Stage 3 shape mismatch: {outputs['stage3'].shape}"
    print(" - Model Forward Pass: OK")

    # -------------------------------------------------------------------------
    # 6. Loss Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Testing Loss Calculation...")

    criterion = CombinedLoss(device=device)
    loss, loss_dict = criterion(outputs, labels, lengths)

    print(f" - Total Loss: {loss.item():.4f}")
    print(f" - Loss Components: {list(loss_dict.keys())}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    assert loss.requires_grad, "Loss should require gradients"
    print(" - Loss Calculation: OK")

    # -------------------------------------------------------------------------
    # 7. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n>>> Testing Training Loop (1 Epoch, Mini Dataset)...")

    # Create Val Loader for Trainer
    val_ds = SkeletonAudioDataset(
        metadata_csv=mini_val_csv,
        input_dir=Config.INPUT_DIR,
        cache_dir=temp_cache_dir,
        split="val",
        load_cached_data=False,
    )
    val_loader = DataLoader(val_ds, batch_size=2, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    # Run 1 epoch of training
    train_loss = trainer.train_epoch(epoch=1)
    assert train_loss > 0, "Train loss should be positive"

    # Run validation
    val_loss, val_score = trainer.validate(epoch=1)
    print(f" - Validation Score (Levenshtein Error): {val_score:.4f}")

    print(" - Training Loop: OK")

    # Cleanup
    if os.path.exists(mini_data_dir):
        shutil.rmtree(mini_data_dir)
    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    run_demo()
