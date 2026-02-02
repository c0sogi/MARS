import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import shutil

# Import from the provided library
from library.config import (
    WORKING_DIR,
    INPUT_DIM,
    NUM_CLASSES,
    TRAIN_METADATA_PATH,
    set_seed,
)
from library.data_loader import GestureDataset
from library.model import CascadedRefinementNet
from library.train import compute_loss
from library.utils import decode_predictions, compute_levenshtein_ratio, save_submission


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Reproducibility
    set_seed(42)
    temp_dir = os.path.join(WORKING_DIR, "demo_temp")
    os.makedirs(temp_dir, exist_ok=True)

    print("\n[1] Creating Mini Metadata for Fast Processing...")
    # Read the full training metadata
    if not os.path.exists(TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found at {TRAIN_METADATA_PATH}")

    df_full = pd.read_csv(TRAIN_METADATA_PATH)

    # Select a small subset (e.g., 4 samples) to speed up feature extraction
    # We choose samples that actually exist to ensure data loading works
    df_mini = df_full.head(4).copy()

    mini_metadata_path = os.path.join(temp_dir, "mini_train.csv")
    df_mini.to_csv(mini_metadata_path, index=False)
    print(
        f"    Saved mini metadata with {len(df_mini)} samples to {mini_metadata_path}"
    )

    print("\n[2] Demonstrating Data Loading...")
    # Instantiate Dataset with the mini metadata
    # This will trigger _process_and_cache for the 4 samples
    dataset = GestureDataset(
        metadata_path=mini_metadata_path,
        mode="train",
        load_cached_data=False,  # Force processing to verify logic
        cache_dir=temp_dir,
    )

    # Verify Dataset Length
    print(f"    Dataset size (windows): {len(dataset)}")
    assert len(dataset) > 0, "Dataset should not be empty."

    # Create DataLoader
    batch_size = 2
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Fetch one batch
    inputs, targets = next(iter(loader))
    print(f"    Batch Inputs Shape: {inputs.shape} (Expected: [B, Time, {INPUT_DIM}])")
    print(f"    Batch Targets Shape: {targets.shape} (Expected: [B, Time])")

    # Assertions
    assert inputs.dim() == 3, "Inputs should be 3-dimensional (Batch, Time, Feats)"
    assert (
        inputs.shape[2] == INPUT_DIM
    ), f"Feature dimension mismatch. Got {inputs.shape[2]}, expected {INPUT_DIM}"
    assert targets.dim() == 2, "Targets should be 2-dimensional (Batch, Time)"
    assert inputs.shape[0] == batch_size, "Batch size mismatch"

    print("\n[3] Demonstrating Model Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CascadedRefinementNet().to(device)

    inputs = inputs.to(device)
    targets = targets.to(device)

    # Forward Pass
    s1_logits, s2_logits = model(inputs)

    print(f"    Stage 1 Logits Shape: {s1_logits.shape}")
    print(f"    Stage 2 Logits Shape: {s2_logits.shape}")

    # Assertions
    assert s1_logits.shape == (
        batch_size,
        inputs.shape[1],
        NUM_CLASSES,
    ), "Stage 1 output shape mismatch"
    assert s2_logits.shape == (
        batch_size,
        inputs.shape[1],
        NUM_CLASSES,
    ), "Stage 2 output shape mismatch"

    print("\n[4] Demonstrating Training Step (Loss & Optimization)...")
    # Setup Criterion and Optimizer
    weights = torch.ones(NUM_CLASSES).to(device)
    weights[0] = 0.2  # Example BG weight
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Compute Loss
    total_loss, l1, l2, l_smooth = compute_loss(
        s1_logits, s2_logits, targets, criterion
    )

    print(f"    Total Loss: {total_loss.item():.4f}")
    print(
        f"    (Components -> S1: {l1.item():.4f}, S2: {l2.item():.4f}, Smooth: {l_smooth.item():.4f})"
    )

    # Backward Pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    print("    Backward pass and optimizer step successful.")

    print("\n[5] Verifying Utility Functions...")

    # A. Decode Predictions
    # Create a synthetic sequence: Background(0) -> Gesture(1) -> Gesture(1) -> Background(0) -> Gesture(2)
    # Expected output: [1, 2]
    dummy_indices = np.array([0, 0, 1, 1, 1, 0, 0, 2, 2, 0])
    decoded = decode_predictions(dummy_indices)
    print(f"    Input Indices: {dummy_indices}")
    print(f"    Decoded: {decoded}")

    assert decoded == [1, 2], f"Decoding logic failed. Expected [1, 2], got {decoded}"

    # Test with probability input (logits)
    # Create logits where class 1 is high, then class 2
    dummy_logits = np.zeros((5, NUM_CLASSES))
    dummy_logits[0:2, 1] = 10.0  # Class 1
    dummy_logits[2:5, 2] = 10.0  # Class 2
    decoded_logits = decode_predictions(dummy_logits)
    print(f"    Decoded from Logits: {decoded_logits}")
    assert decoded_logits == [1, 2], "Decoding from logits failed."

    # B. Levenshtein Ratio
    # Pred: [1, 2], Truth: [1, 3] -> Distance = 1 (substitute 2 with 3). Length = 2. Ratio = 0.5
    preds = [[1, 2]]
    truths = [[1, 3]]
    score = compute_levenshtein_ratio(preds, truths)
    print(f"    Levenshtein Ratio (Pred=[1,2], Truth=[1,3]): {score}")
    assert np.isclose(
        score, 0.5
    ), f"Levenshtein calculation failed. Expected 0.5, got {score}"

    # Test perfect match
    score_perfect = compute_levenshtein_ratio([[1, 2, 3]], [[1, 2, 3]])
    assert score_perfect == 0.0, "Perfect match should have 0.0 error."

    print("\n[6] Demonstrating Submission Generation...")
    sample_ids = ["Sample001", "Sample002"]
    predictions = [[1, 5, 10], [2]]
    output_csv = os.path.join(temp_dir, "submission_demo.csv")

    save_submission(predictions, sample_ids, output_csv)

    # Verify file content
    with open(output_csv, "r") as f:
        lines = f.read().strip().split("\n")
        print("    Submission File Content:")
        for line in lines:
            print(f"      {line}")

    assert len(lines) == 2
    assert lines[0] == "Sample001,1,5,10"
    assert lines[1] == "Sample002,2"

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
