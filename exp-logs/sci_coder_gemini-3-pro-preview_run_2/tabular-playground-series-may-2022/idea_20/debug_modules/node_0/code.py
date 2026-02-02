import os
import sys
import torch
import pandas as pd
import numpy as np

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import AutoencodingHybridNet
from library.train import (
    train_one_epoch,
    validate,
    calculate_multitask_loss,
    predict_and_submit,
)


def main():
    print("=== Starting Demonstration of Manufacturing Control Task Solution ===\n")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("1. Configuring environment for rapid demonstration...")

    # Set deterministic seed
    seed_everything(42)

    # Override Config parameters for speed
    # We use DEBUG mode to limit the dataset size significantly (e.g., 2000 samples)
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 2000
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 64
    Config.NUM_WORKERS = 2  # Reduce worker overhead for small data

    # Ensure working directories exist (Config creates them on import, but we double-check)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    print(f"   Debug Mode: {Config.DEBUG}")
    print(f"   Batch Size: {Config.BATCH_SIZE}")

    # --------------------------------------------------------------------------
    # 2. Data Pipeline Verification
    # --------------------------------------------------------------------------
    print("\n2. Verifying Data Loading and Processing...")

    # We set load_cached_data=False to verify the raw data processing logic
    # This reads from ./input and ./metadata, processes features, and creates DataLoaders
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=False
    )

    # Fetch a single batch to verify structure and shapes
    batch = next(iter(train_loader))

    continuous = batch["continuous"]
    sequence = batch["sequence"]
    targets = batch["target"]
    recon_targets = batch["reconstruction_target"]

    print(f"   Batch keys found: {list(batch.keys())}")
    print(
        f"   Continuous shape: {continuous.shape} (Expected: {Config.BATCH_SIZE}, {Config.NUM_CONTINUOUS_FEATURES})"
    )
    print(
        f"   Sequence shape:   {sequence.shape}   (Expected: {Config.BATCH_SIZE}, {Config.SEQ_LEN})"
    )

    # Assertions
    assert continuous.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CONTINUOUS_FEATURES,
    ), "Continuous feature shape mismatch"
    assert sequence.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), "Sequence feature shape mismatch"
    assert targets.shape == (Config.BATCH_SIZE,), "Target shape mismatch"
    assert recon_targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), "Reconstruction target shape mismatch"

    print("   Data pipeline verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n3. Verifying Model Architecture and Forward Pass...")

    model = AutoencodingHybridNet().to(device)

    # Create dummy inputs on the device
    dummy_cont = torch.randn(Config.BATCH_SIZE, Config.NUM_CONTINUOUS_FEATURES).to(
        device
    )
    # Random integers for sequence (0 to VOCAB_SIZE-1)
    dummy_seq = torch.randint(
        0, Config.VOCAB_SIZE, (Config.BATCH_SIZE, Config.SEQ_LEN)
    ).to(device)

    # Perform forward pass
    cls_logits, recon_logits = model(dummy_cont, dummy_seq)

    print(
        f"   CLS Logits shape:   {cls_logits.shape} (Expected: {Config.BATCH_SIZE}, 1)"
    )
    print(
        f"   Recon Logits shape: {recon_logits.shape} (Expected: {Config.BATCH_SIZE}, {Config.SEQ_LEN}, {Config.VOCAB_SIZE})"
    )

    # Assertions
    assert cls_logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), "Classification logits shape mismatch"
    assert recon_logits.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.VOCAB_SIZE,
    ), "Reconstruction logits shape mismatch"

    print("   Model architecture verified successfully.")

    # --------------------------------------------------------------------------
    # 4. Loss Function Verification
    # --------------------------------------------------------------------------
    print("\n4. Verifying Multi-Task Loss Calculation...")

    dummy_targets = torch.randint(0, 2, (Config.BATCH_SIZE,)).float().to(device)
    dummy_recon_targets = (
        dummy_seq  # The target for reconstruction is the input sequence
    )

    total_loss, cls_loss, recon_loss = calculate_multitask_loss(
        cls_logits, recon_logits, dummy_targets, dummy_recon_targets
    )

    print(f"   Total Loss: {total_loss.item():.4f}")
    print(f"   CLS Loss:   {cls_loss.item():.4f}")
    print(f"   Recon Loss: {recon_loss.item():.4f}")

    # Assertions
    assert not torch.isnan(total_loss), "Loss is NaN"
    assert total_loss.item() > 0, "Loss should be positive"

    print("   Loss calculation verified successfully.")

    # --------------------------------------------------------------------------
    # 5. Training Loop Integration
    # --------------------------------------------------------------------------
    print("\n5. Running Training Loop Integration (1 Epoch)...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Run one epoch of training
    # This verifies the interaction between data, model, loss, and optimizer
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    print(f"   Epoch 1 Train Loss: {train_loss:.4f}")

    # Run validation
    val_loss, val_auc = validate(model, val_loader, device)
    print(f"   Epoch 1 Val Loss:   {val_loss:.4f}")
    print(f"   Epoch 1 Val AUC:    {val_auc:.4f}")

    # Assertions
    assert train_loss > 0, "Training loss invalid"
    assert 0 <= val_auc <= 1.0, "AUC score out of bounds"

    print("   Training loop execution verified successfully.")

    # --------------------------------------------------------------------------
    # 6. Inference and Submission
    # --------------------------------------------------------------------------
    print("\n6. Verifying Inference and Submission Generation...")

    # Generate predictions using the model (in its current state)
    predict_and_submit(model, test_loader, test_ids, device)

    # Verify the output file
    submission_path = Config.SUBMISSION_PATH
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"   Submission file loaded: {submission_path}")
    print(f"   Shape: {df_sub.shape}")
    print(f"   Columns: {df_sub.columns.tolist()}")

    # Assertions
    assert len(df_sub) == len(
        test_ids
    ), "Submission row count does not match test set size"
    assert (
        "id" in df_sub.columns and "target" in df_sub.columns
    ), "Submission columns missing"
    assert not df_sub.isnull().any().any(), "Submission contains null values"

    print("   Inference and submission verified successfully.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
