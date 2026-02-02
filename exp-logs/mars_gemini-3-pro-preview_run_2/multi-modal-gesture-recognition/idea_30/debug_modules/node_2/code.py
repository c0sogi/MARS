import warnings
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import components from the provided library
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    INPUT_DIM,
    NUM_CLASSES,
    NUM_STAGES,
)
from library.utils import set_seed, levenshtein_score
from library.data_loader import GestureDataset, collate_fn
from library.model import GestureRecognitionModel
from library.losses import DeepSupervisionLoss


def demo():
    print("=== Starting Library Code Demonstration ===")

    # 1. Reproducibility
    print("\n[1] Setting Random Seeds")
    set_seed(42)
    print("Seeds set.")

    # 2. Data Loading
    print("\n[2] Demonstrating Data Loading (Subset)")
    # We use a small limit (4 samples) and force reprocessing (load_cached_data=False)
    # to ensure the demo runs quickly without loading large cache files.
    batch_size = 2

    print("Initializing Training Dataset (Limit: 4 samples)...")
    train_ds = GestureDataset(
        TRAIN_METADATA_PATH, is_train=True, load_cached_data=False, limit=4
    )

    print("Initializing Validation Dataset (Limit: 4 samples)...")
    val_ds = GestureDataset(
        VAL_METADATA_PATH, is_train=False, load_cached_data=False, limit=4
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Fetch a batch to verify structure
    batch = next(iter(train_loader))
    features = batch["features"]
    target_cls = batch["target_cls"]
    target_bnd = batch["target_bnd"]
    mask = batch["mask"]

    print(f"Batch Features Shape: {features.shape} (Expected: [B, T, {INPUT_DIM}])")
    print(f"Batch Mask Shape: {mask.shape} (Expected: [B, T])")

    # Assertions to verify data integrity
    assert features.dim() == 3
    assert features.shape[2] == INPUT_DIM
    assert mask.dim() == 2
    assert features.shape[0] <= batch_size

    # 3. Model Architecture
    print("\n[3] Demonstrating Model Architecture (MCAGCN)")
    # Initialize the high-level wrapper
    model_wrapper = GestureRecognitionModel()
    model = model_wrapper.model  # Access the underlying nn.Module

    # Move batch to device
    device = model_wrapper.device
    features = features.to(device)
    mask = mask.to(device)
    target_cls = target_cls.to(device)
    target_bnd = target_bnd.to(device)

    # Forward Pass
    print("Running Forward Pass...")
    outputs = model(features, mask)

    print(f"Number of Output Stages: {len(outputs)} (Expected: {NUM_STAGES})")
    assert len(outputs) == NUM_STAGES

    # Verify output of the final stage
    final_stage = outputs[-1]
    cls_probs = final_stage["cls_probs"]
    bnd_probs = final_stage["bnd_probs"]

    print(
        f"Final Stage Class Probs Shape: {cls_probs.shape} (Expected: [B, T, {NUM_CLASSES}])"
    )
    print(f"Final Stage Boundary Probs Shape: {bnd_probs.shape} (Expected: [B, T, 1])")

    assert cls_probs.shape[2] == NUM_CLASSES
    assert bnd_probs.shape[2] == 1

    # 4. Loss Calculation
    print("\n[4] Demonstrating Loss Calculation")
    criterion = DeepSupervisionLoss().to(device)

    loss, metrics = criterion(outputs, target_cls, target_bnd, mask)

    print(f"Total Loss: {loss.item():.4f}")
    print("Loss Metrics:", list(metrics.keys()))

    assert not torch.isnan(loss)
    assert loss.requires_grad

    # 5. Training Loop Integration
    print("\n[5] Demonstrating Training Loop (1 Epoch)")
    # We use the fit method of the wrapper with our small loaders
    # This verifies the optimizer step, gradient clipping, and validation logic
    model_wrapper.fit(train_loader, val_loader, epochs=1)
    print("Training epoch completed successfully.")

    # 6. Inference and Post-Processing
    print("\n[6] Demonstrating Inference and Post-Processing")

    # Create a dummy test loader
    print("Initializing Test Dataset (Limit: 2 samples)...")
    test_ds = GestureDataset(
        TEST_METADATA_PATH, is_train=False, load_cached_data=False, limit=2
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    predictions = model_wrapper.predict(test_loader)

    print(f"Generated {len(predictions)} predictions.")
    print(f"Sample Prediction Output: {predictions[0]}")

    assert len(predictions) == 2
    assert isinstance(predictions[0], str)
    # Check format: SessionID,label1,label2...
    assert "," in predictions[0]

    # 7. Metric Verification
    print("\n[7] Verifying Levenshtein Metric")
    # Case 1: Identical sequences
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    score_perfect = levenshtein_score([seq1], [seq2])
    print(f"Score (Identical): {score_perfect} (Expected: 0.0)")
    assert score_perfect == 0.0

    # Case 2: One deletion (distance=1, target_len=2) -> Score = 0.5
    seq3 = [1, 2]
    score_diff = levenshtein_score([seq1], [seq3])
    print(f"Score (Diff): {score_diff} (Expected: 0.5)")
    assert score_diff == 0.5

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    demo()
