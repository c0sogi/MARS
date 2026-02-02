import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Import from the provided library
from library.config import Config
from library.dataset import get_dataloaders
from library.model import HybridDualEncoder
from library.engine import train_fn, eval_fn
from library.utils import seed_everything, compute_spearmanr


def main():
    print("Starting demonstration of StackExchange QA Labeling pipeline...")

    # =========================================================================
    # 1. Setup and Configuration Override
    # =========================================================================
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Override Config for speed and demonstration purposes
    print("Configuring parameters for fast demonstration...")
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.ACCUMULATION_STEPS = 1  # No accumulation for demo
    Config.EPOCHS_ACTUAL = 1  # Run only 1 epoch
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple demo

    # Device setup
    device = Config.DEVICE
    print(f"Using device: {device}")

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("\nInitializing DataLoaders (Debug Mode)...")
    # debug=True loads a small subset (100 train, 50 val, 50 test)
    train_loader, val_loader, test_loader, target_cols = get_dataloaders(
        load_cached_data=True, debug=True
    )

    # Verify DataLoader output
    print("Verifying batch structure...")
    sample_batch = next(iter(train_loader))

    # Assertions to check batch integrity
    required_keys = [
        "qa_id",
        "input_ids_q",
        "attention_mask_q",
        "input_ids_a",
        "attention_mask_a",
        "labels",
    ]
    for key in required_keys:
        assert key in sample_batch, f"Missing key in batch: {key}"

    # Check shapes
    batch_size = sample_batch["input_ids_q"].shape[0]
    assert (
        batch_size == Config.TRAIN_BATCH_SIZE
    ), f"Batch size mismatch. Expected {Config.TRAIN_BATCH_SIZE}, got {batch_size}"
    assert sample_batch["labels"].shape == (
        batch_size,
        30,
    ), f"Labels shape mismatch. Expected ({batch_size}, 30), got {sample_batch['labels'].shape}"

    print("Batch structure verified successfully.")

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    print("\nInitializing HybridDualEncoder model...")
    model = HybridDualEncoder()
    model.to(device)

    # Verify model output shape with a dummy forward pass
    print("Verifying model forward pass...")
    with torch.no_grad():
        input_ids_q = sample_batch["input_ids_q"].to(device)
        mask_q = sample_batch["attention_mask_q"].to(device)
        input_ids_a = sample_batch["input_ids_a"].to(device)
        mask_a = sample_batch["attention_mask_a"].to(device)

        logits = model(input_ids_q, mask_q, input_ids_a, mask_a)

    assert logits.shape == (
        batch_size,
        30,
    ), f"Model output shape mismatch. Expected ({batch_size}, 30), got {logits.shape}"
    print("Model forward pass verified successfully.")

    # =========================================================================
    # 4. Training Loop Demonstration
    # =========================================================================
    print("\nRunning Training Loop (1 Epoch)...")

    # Setup Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=1e-4)
    num_train_steps = len(train_loader) * Config.EPOCHS_ACTUAL
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    # Run training function
    train_loss = train_fn(model, train_loader, optimizer, scheduler, device)

    print(f"Training completed. Loss: {train_loss:.4f}")
    assert isinstance(train_loss, float), "train_fn did not return a float loss."
    assert train_loss > 0, "Training loss should be positive."

    # =========================================================================
    # 5. Evaluation Loop Demonstration
    # =========================================================================
    print("\nRunning Evaluation Loop (Validation Set)...")

    val_score, val_preds = eval_fn(model, val_loader, device)

    print(f"Evaluation completed. Spearman Score: {val_score:.4f}")

    # Verify predictions shape
    # debug mode val set size is 50
    expected_val_size = 50
    # Adjust expectation if batch size doesn't divide perfectly (though drop_last is False for val)
    assert val_preds.shape == (
        expected_val_size,
        30,
    ), f"Prediction shape mismatch. Expected ({expected_val_size}, 30), got {val_preds.shape}"

    # Verify score range
    assert -1.0 <= val_score <= 1.0, f"Spearman score {val_score} out of range [-1, 1]"

    # =========================================================================
    # 6. Metric Verification
    # =========================================================================
    print("\nVerifying Metric Calculation (compute_spearmanr)...")

    # Case 1: Perfect correlation
    y_true = np.random.rand(100, 30)
    y_pred_perfect = y_true.copy()
    score_perfect = compute_spearmanr(y_pred_perfect, y_true)
    print(f"Perfect Correlation Score: {score_perfect:.4f}")
    assert np.isclose(score_perfect, 1.0), "Metric failed on perfect correlation."

    # Case 2: Inverse correlation
    y_pred_inverse = 1 - y_true
    score_inverse = compute_spearmanr(y_pred_inverse, y_true)
    print(f"Inverse Correlation Score: {score_inverse:.4f}")
    assert np.isclose(score_inverse, -1.0), "Metric failed on inverse correlation."

    # Case 3: Random noise (should be near 0)
    y_pred_random = np.random.rand(100, 30)
    score_random = compute_spearmanr(y_pred_random, y_true)
    print(f"Random Correlation Score: {score_random:.4f}")
    assert -0.2 < score_random < 0.2, "Metric failed on random data (should be near 0)."

    print("\nAll demonstrations and verifications passed successfully!")


if __name__ == "__main__":
    main()
