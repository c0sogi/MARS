import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library components
# Note: We do NOT import library.train to avoid triggering the automatic run_training() call.
from library.config import Config
from library.utils import seed_everything, compute_mae
from library.data import load_and_preprocess
from library.model import RDHNet


def main():
    print("=== Ventilator Pressure Prediction Demo ===")

    # ------------------------------------------------------------------------
    # 1. Configuration Setup
    # ------------------------------------------------------------------------
    print("\n[1] Configuring Environment...")

    # Enable Debug mode to use a data subset (1000 breaths) for speed
    Config.DEBUG = True

    # Set a specific working directory for this demo to isolate outputs
    Config.WORKING_DIR = "./working/demo_execution"

    # Manually update dependent paths since they were assigned at import time
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SCALER_PATH = os.path.join(Config.WORKING_DIR, "scaler.joblib")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Initialize directories
    Config.setup()

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # ------------------------------------------------------------------------
    # 2. Data Loading & Preprocessing
    # ------------------------------------------------------------------------
    print("\n[2] Loading and Preprocessing Data...")

    # load_and_preprocess handles feature engineering, scaling, and caching.
    # We force load_cached_data=False to demonstrate the processing logic.
    train_ds, val_ds, test_ds = load_and_preprocess(load_cached_data=False)

    # Verify dataset integrity
    print(f"    Train samples: {len(train_ds)}")
    print(f"    Val samples:   {len(val_ds)}")
    print(f"    Test samples:  {len(test_ds)}")

    assert len(train_ds) > 0, "Training dataset is empty."
    assert len(val_ds) > 0, "Validation dataset is empty."

    # Verify sample structure
    sample = train_ds[0]
    input_dim = Config.get_input_dim()

    assert "input" in sample, "Dataset sample missing 'input' key."
    assert "target" in sample, "Dataset sample missing 'target' key."
    assert "u_out" in sample, "Dataset sample missing 'u_out' key."
    assert sample["input"].shape == (
        80,
        input_dim,
    ), f"Incorrect input shape. Expected (80, {input_dim}), got {sample['input'].shape}"

    # ------------------------------------------------------------------------
    # 3. Model Instantiation
    # ------------------------------------------------------------------------
    print("\n[3] Instantiating RDHNet Model...")

    model = RDHNet().to(device)

    # Verify forward pass with a dummy batch
    # Unsqueeze to add batch dimension: (80, F) -> (1, 80, F)
    dummy_input = sample["input"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    # Expected output shape: (Batch_Size, Sequence_Length) -> (1, 80)
    assert output.shape == (
        1,
        80,
    ), f"Model output shape mismatch. Expected (1, 80), got {output.shape}"
    print("    Model forward pass successful.")

    # ------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # ------------------------------------------------------------------------
    print("\n[4] Demonstrating Training Loop (5 Steps)...")

    # Create DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, drop_last=True
    )

    # Setup Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    model.train()
    steps_to_run = 5

    for i, batch in enumerate(train_loader):
        if i >= steps_to_run:
            break

        inputs = batch["input"].to(device)
        targets = batch["target"].to(device)
        u_out = batch["u_out"].to(device)

        optimizer.zero_grad()

        # Forward
        preds = model(inputs)

        # Masked L1 Loss Calculation (Logic from library.train)
        # Only penalize predictions during the inspiratory phase (u_out == 0)
        mask = (u_out == 0).float()
        loss = (torch.abs(preds - targets) * mask).sum() / (mask.sum() + 1e-8)

        # Backward
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        print(f"    Step {i+1}/{steps_to_run} | Loss: {loss.item():.4f}")

    # ------------------------------------------------------------------------
    # 5. Validation & Metrics
    # ------------------------------------------------------------------------
    print("\n[5] Demonstrating Validation & Metric Calculation...")

    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    batch = next(iter(val_loader))

    inputs = batch["input"].to(device)
    targets = batch["target"].to(device)
    u_out = batch["u_out"].to(device)

    model.eval()
    with torch.no_grad():
        preds = model(inputs)
        # Use provided utility to compute MAE
        mae = compute_mae(preds, targets, u_out)

    print(f"    Validation Batch MAE: {mae:.4f}")
    assert mae >= 0, "MAE must be non-negative."

    # ------------------------------------------------------------------------
    # 6. Submission Generation
    # ------------------------------------------------------------------------
    print("\n[6] Generating Sample Submission...")

    # Predict on a small subset of test data
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False)
    preds_list = []

    # Run inference on first 2 batches
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 2:
                break
            inputs = batch["input"].to(device)
            preds = model(inputs)
            preds_list.append(preds.cpu().numpy())

    # Flatten predictions
    flat_preds = np.concatenate(preds_list, axis=0).flatten()

    # Load test IDs from cache (generated during preprocessing)
    test_ids_path = os.path.join(Config.CACHE_DIR, "test_ids.npy")
    test_ids = np.load(test_ids_path)

    # Match IDs to predictions (subset)
    subset_ids = test_ids[: len(flat_preds)]

    # Create DataFrame
    submission = pd.DataFrame({"id": subset_ids, "pressure": flat_preds})

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"    Submission saved to: {Config.SUBMISSION_PATH}")

    # Verify file exists and format
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."
    df_check = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(df_check.columns) == ["id", "pressure"], "Submission columns mismatch."
    assert len(df_check) == len(flat_preds), "Submission length mismatch."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
