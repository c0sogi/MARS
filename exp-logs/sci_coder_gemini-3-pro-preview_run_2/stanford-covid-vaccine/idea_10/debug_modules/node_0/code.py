import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import (
    DEVICE,
    WORKING_DIR,
    SEQ_LEN,
    PRED_LEN,
    ALL_TARGETS,
    SCORED_TARGETS,
    INPUT_CHANNELS,
    NUM_TARGETS,
    SEQ_LEN,
)
from library.utils import set_seed, GlobalMCRMSE
from library.data import get_loaders
from library.model import StagedInteractiveDenseNet
from library.engine import train_fn, eval_fn, predict_fn


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup
    set_seed(42)
    print(f"Device: {DEVICE}")
    print(f"Working Directory: {WORKING_DIR}")

    # 2. Data Loading
    print("\n--- 1. Data Loading Verification ---")
    # We load cached data if available, or process from metadata
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # Fetch one batch to verify shapes
    features, partners, targets = next(iter(train_loader))

    print(f"Batch Size: {features.shape[0]}")
    print(
        f"Features Shape: {features.shape} (Expected: [B, {INPUT_CHANNELS}, {SEQ_LEN}])"
    )
    print(f"Partners Shape: {partners.shape} (Expected: [B, {SEQ_LEN}])")
    print(f"Targets Shape: {targets.shape} (Expected: [B, {SEQ_LEN}, {NUM_TARGETS}])")

    # Assertions
    assert features.shape[1] == INPUT_CHANNELS, "Incorrect input channel dimension"
    assert features.shape[2] == SEQ_LEN, "Incorrect sequence length in features"
    assert partners.shape[1] == SEQ_LEN, "Incorrect sequence length in partners"
    assert targets.shape[2] == NUM_TARGETS, "Incorrect number of targets"
    print("Data shapes verified successfully.")

    # 3. Model Instantiation & Forward Pass
    print("\n--- 2. Model Architecture Verification ---")
    model = StagedInteractiveDenseNet().to(DEVICE)

    # Move batch to device
    features = features.to(DEVICE)
    partners = partners.to(DEVICE)

    # Forward pass
    outputs = model(features, partners)

    print(f"Output Shape: {outputs.shape} (Expected: [B, {SEQ_LEN}, {NUM_TARGETS}])")

    # Assertions
    assert outputs.shape == (
        features.shape[0],
        SEQ_LEN,
        NUM_TARGETS,
    ), "Model output shape mismatch"
    print("Model forward pass verified successfully.")

    # 4. Metric Logic Verification
    print("\n--- 3. Metric Logic Verification (GlobalMCRMSE) ---")
    metric = GlobalMCRMSE(device=DEVICE, seq_scored=68)

    # Create synthetic data
    # Targets: All zeros
    # Preds: All ones
    # Scored targets are indices [0, 1, 3] (reactivity, deg_Mg_pH10, deg_Mg_50C)
    # Error should be 1.0 for scored columns, 0.0 or ignored for others.
    # Since preds are 1 and targets 0, MSE is 1, RMSE is 1. Mean RMSE is 1.

    B_test = 5
    synth_targets = torch.zeros((B_test, SEQ_LEN, NUM_TARGETS), device=DEVICE)
    synth_preds = torch.ones((B_test, SEQ_LEN, NUM_TARGETS), device=DEVICE)

    metric.reset()
    metric.update(synth_preds, synth_targets)
    score = metric.compute()

    print(f"Computed MCRMSE on synthetic data (Preds=1, Targets=0): {score:.4f}")

    # Assert close to 1.0
    assert (
        abs(score - 1.0) < 1e-5
    ), f"Metric calculation incorrect. Expected 1.0, got {score}"
    print("Metric logic verified successfully.")

    # 5. Training Loop Demonstration
    print("\n--- 4. Training Loop Demonstration (2 Epochs) ---")
    # We use a reduced number of epochs for demonstration speed
    DEMO_EPOCHS = 2
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(DEMO_EPOCHS):
        # Train
        train_loss = train_fn(model, train_loader, optimizer, DEVICE)

        # Validation
        val_score = eval_fn(model, val_loader, DEVICE)

        print(
            f"Epoch {epoch+1}/{DEMO_EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        # Basic assertion to ensure loss is not NaN
        assert not np.isnan(train_loss), "Training loss is NaN"
        assert not np.isnan(val_score), "Validation score is NaN"

    print("Training loop completed successfully.")

    # 6. Inference & Submission Generation
    print("\n--- 5. Inference and Submission Formatting ---")

    # Predict on test set
    preds, ids = predict_fn(model, test_loader, DEVICE)

    print(
        f"Prediction Array Shape: {preds.shape} (Expected: [{len(test_loader.dataset)}, {SEQ_LEN}, {NUM_TARGETS}])"
    )
    assert preds.shape[0] == len(test_loader.dataset), "Incorrect number of predictions"

    # Format a small sample of the submission to verify logic
    print("Formatting sample submission rows...")
    submission_data = []

    # Process just the first sample for demonstration
    sample_idx = 0
    sample_id = ids[sample_idx]
    sample_preds = preds[sample_idx]

    # We iterate up to PRED_LEN (107)
    for seqpos in range(PRED_LEN):
        row_id = f"{sample_id}_{seqpos}"
        vals = sample_preds[seqpos]

        row_data = {
            "id_seqpos": row_id,
            "reactivity": vals[0],
            "deg_Mg_pH10": vals[1],
            "deg_pH10": vals[2],
            "deg_Mg_50C": vals[3],
            "deg_50C": vals[4],
        }
        submission_data.append(row_data)

    submission_df = pd.DataFrame(submission_data)

    # Verify columns
    expected_cols = ["id_seqpos"] + ALL_TARGETS
    submission_df = submission_df[expected_cols]

    print("Sample Submission DataFrame:")
    print(submission_df.head(3))

    assert len(submission_df) == PRED_LEN, "Submission rows for single sample mismatch"
    assert list(submission_df.columns) == expected_cols, "Submission columns mismatch"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
