import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library components
from library.config import (
    TRAIN_CSV,
    SEED,
    DEVICE,
    NUM_CLASSES,
    BACKGROUND_LABEL,
    SKELETON_INPUT_SIZE,
    AUDIO_INPUT_SIZE,
    WORKING_DIR,
)
from library.data_utils import process_sample, compute_global_stats
from library.dataset import GestureDataset, collate_fn
from library.model import GCAResNet
from library.metrics import evaluate_batch, decode_predictions


def run_demo():
    print("=== Starting Gesture Recognition Pipeline Demo ===")

    # 1. Reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # 2. Load Metadata
    print("\n[Step 1] Loading Metadata...")
    if not os.path.exists(TRAIN_CSV):
        raise FileNotFoundError(f"Training metadata not found at {TRAIN_CSV}")

    full_df = pd.read_csv(TRAIN_CSV)
    # Use a tiny subset for speed (first 4 samples)
    subset_df = full_df.head(4).copy().reset_index(drop=True)
    print(f"Loaded {len(subset_df)} samples for demonstration.")

    # 3. Data Processing & Stats
    print("\n[Step 2] Processing Samples and Computing Stats...")
    # process_sample caches data to disk. We run it explicitly to verify it works.
    sample_res = process_sample(subset_df.iloc[0], load_cached_data=False)

    assert sample_res is not None, "process_sample returned None"
    assert "skeleton" in sample_res, "Missing skeleton data"
    assert "audio" in sample_res, "Missing audio data"
    assert (
        sample_res["skeleton"].shape[1] == SKELETON_INPUT_SIZE
    ), f"Incorrect skeleton feature dim: {sample_res['skeleton'].shape}"
    assert (
        sample_res["audio"].shape[1] == AUDIO_INPUT_SIZE
    ), f"Incorrect audio feature dim: {sample_res['audio'].shape}"

    # Compute stats on this subset
    stats = compute_global_stats(subset_df, load_cached_data=False)
    assert "skel_mean" in stats and "skel_std" in stats
    assert stats["skel_mean"].shape[0] == SKELETON_INPUT_SIZE
    print("Stats computed successfully.")

    # 4. Dataset & DataLoader
    print("\n[Step 3] Initializing Dataset and DataLoader...")
    # Initialize dataset with augmentation enabled
    dataset = GestureDataset(subset_df, stats=stats, is_train=True, augment=True)

    # Check __getitem__
    item = dataset[0]
    assert isinstance(item["skeleton"], torch.Tensor)
    assert isinstance(item["audio"], torch.Tensor)
    assert isinstance(item["labels"], torch.Tensor)

    # Initialize DataLoader with custom collate_fn
    batch_size = 2
    loader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
    )

    # Fetch one batch
    batch = next(iter(loader))
    assert batch is not None

    skeletons = batch["skeleton"]
    audios = batch["audio"]
    labels = batch["labels"]
    lengths = batch["lengths"]
    label_lengths = batch["label_lengths"]

    print(
        f"Batch shapes - Skeleton: {skeletons.shape}, Audio: {audios.shape}, Labels: {labels.shape}"
    )

    # Verify Padding
    # Max length in batch
    max_len = lengths.max().item()
    assert skeletons.shape[1] == max_len, "Skeleton temporal dimension mismatch"
    assert audios.shape[1] == max_len, "Audio temporal dimension mismatch"
    assert skeletons.shape[2] == SKELETON_INPUT_SIZE
    assert audios.shape[2] == AUDIO_INPUT_SIZE

    # 5. Model Initialization & Forward Pass
    print("\n[Step 4] Model Initialization and Inference...")
    model = GCAResNet().to(DEVICE)
    model.train()  # Set to train mode

    # Move batch to device
    skeletons = skeletons.to(DEVICE)
    audios = audios.to(DEVICE)
    lengths = lengths.to(DEVICE)
    labels = labels.to(DEVICE)
    label_lengths = label_lengths.to(DEVICE)

    # Forward pass
    logits = model(skeletons, audios, lengths)

    # Check output shape: (Batch, Time, NumClasses)
    assert logits.shape[0] == batch_size
    assert logits.shape[1] == max_len
    assert logits.shape[2] == NUM_CLASSES
    print(f"Logits shape verified: {logits.shape}")

    # 6. Loss Calculation
    print("\n[Step 5] Calculating CTC Loss...")
    criterion = nn.CTCLoss(blank=BACKGROUND_LABEL, reduction="mean", zero_infinity=True)

    # CTC expects (T, B, C) and LogSoftmax
    log_probs = nn.functional.log_softmax(logits, dim=2).permute(1, 0, 2)

    loss = criterion(log_probs, labels, lengths, label_lengths)
    print(f"Loss value: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"

    # 7. Metrics & Decoding
    print("\n[Step 6] Evaluating Metrics...")
    # Decode predictions
    preds = decode_predictions(logits, lengths)
    assert len(preds) == batch_size, "Number of predictions does not match batch size"
    assert isinstance(preds[0], list), "Predictions should be a list of lists"

    # Calculate Levenshtein distance
    total_dist, total_len = evaluate_batch(logits, lengths, labels)
    print(
        f"Batch Levenshtein Distance: {total_dist}, Total Ground Truth Length: {total_len}"
    )

    # 8. Optimization Step
    print("\n[Step 7] Running Optimization Step...")
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    optimizer.zero_grad()
    loss.backward()

    # Check gradients
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            break
    assert has_grad, "No gradients computed after backward pass"

    optimizer.step()
    print("Optimizer step completed.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
