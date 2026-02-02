import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F

# Import from the provided library
from library.config import (
    DEVICE,
    SEQ_LEN,
    PRED_LEN,
    NUM_TARGETS,
    SUBMISSION_FILE,
    MODEL_SAVE_PATH,
)
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import RNAModel
from library.loss import MaskedMSELoss
from library.train import run_training


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup & Reproducibility
    print("\n[1] Setting random seeds...")
    seed_everything(42)

    # 2. Data Pipeline Verification
    print("\n[2] Verifying Data Pipeline...")
    debug_size = 32
    # Load a small subset of data for verification
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False,  # Force reprocessing to verify logic
        debug_sample_size=debug_size,
    )

    # Fetch one batch from training loader
    batch = next(iter(train_loader))

    # Extract components
    sequences = batch["sequence"]
    loop_types = batch["loop_type"]
    pair_dists = batch["pair_dist"]
    targets = batch["targets"]
    ids = batch["id"]

    # Verify Shapes
    print(f"    Batch size: {len(ids)}")
    print(f"    Sequence shape: {sequences.shape} (Expected: {len(ids)}, {SEQ_LEN})")
    print(
        f"    Targets shape: {targets.shape} (Expected: {len(ids)}, {SEQ_LEN}, {NUM_TARGETS})"
    )

    assert sequences.shape == (len(ids), SEQ_LEN), "Incorrect sequence shape"
    assert loop_types.shape == (len(ids), SEQ_LEN), "Incorrect loop_type shape"
    assert pair_dists.shape == (len(ids), SEQ_LEN), "Incorrect pair_dist shape"
    # Targets should be padded to SEQ_LEN in data.py, even though only PRED_LEN is valid
    assert targets.shape == (len(ids), SEQ_LEN, NUM_TARGETS), "Incorrect targets shape"

    print("    -> Data shapes verified successfully.")

    # 3. Model Architecture Verification
    print("\n[3] Verifying Model Architecture...")
    model = RNAModel().to(DEVICE)

    # Move batch to device
    sequences = sequences.to(DEVICE)
    loop_types = loop_types.to(DEVICE)
    pair_dists = pair_dists.to(DEVICE)

    # Forward pass
    preds = model(sequences, loop_types, pair_dists)

    print(
        f"    Predictions shape: {preds.shape} (Expected: {len(ids)}, {SEQ_LEN}, {NUM_TARGETS})"
    )
    assert preds.shape == (
        len(ids),
        SEQ_LEN,
        NUM_TARGETS,
    ), "Model output shape mismatch"
    print("    -> Forward pass successful.")

    # 4. Loss Function Verification
    print("\n[4] Verifying MaskedMSELoss...")
    criterion = MaskedMSELoss()
    targets = targets.to(DEVICE)

    # Calculate loss using the class
    loss = criterion(preds, targets)

    # Manual verification: Slice to PRED_LEN (68) and compute MSE
    preds_scored = preds[:, :PRED_LEN, :]
    targets_scored = targets[:, :PRED_LEN, :]
    manual_loss = F.mse_loss(preds_scored, targets_scored)

    print(f"    Computed Loss: {loss.item():.6f}")
    print(f"    Manual Loss:   {manual_loss.item():.6f}")

    assert torch.isclose(loss, manual_loss), "MaskedMSELoss calculation mismatch"
    print("    -> Loss function logic verified.")

    # 5. Training & Inference Integration
    print("\n[5] Running Training Integration Test...")
    # Run a very short training cycle (2 epochs, small dataset)
    # This uses the logic in library/train.py
    run_training(debug_sample_size=64, epochs=2)

    # Check if model file was created
    assert os.path.exists(MODEL_SAVE_PATH), f"Model file not found at {MODEL_SAVE_PATH}"
    print(f"    -> Model saved successfully at {MODEL_SAVE_PATH}")

    # 6. Output Validation
    print("\n[6] Validating Submission File...")
    assert os.path.exists(
        SUBMISSION_FILE
    ), f"Submission file not found at {SUBMISSION_FILE}"

    df_sub = pd.read_csv(SUBMISSION_FILE)
    print(f"    Submission shape: {df_sub.shape}")
    print(f"    Columns: {list(df_sub.columns)}")

    # Verify columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    # Verify ID format (id_XXXXX_0)
    sample_id = df_sub.iloc[0]["id_seqpos"]
    assert "_id_" in str(sample_id) or sample_id.startswith(
        "id_"
    ), f"Unexpected ID format: {sample_id}"

    # Verify non-predicted columns are zero (as per generate_submission logic)
    assert (df_sub["deg_pH10"] == 0).all(), "deg_pH10 should be 0.0"
    assert (df_sub["deg_50C"] == 0).all(), "deg_50C should be 0.0"

    print("    -> Submission file format verified.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
