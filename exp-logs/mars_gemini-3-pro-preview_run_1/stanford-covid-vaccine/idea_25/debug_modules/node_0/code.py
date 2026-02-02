import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.dataset import get_dataloaders, RNADataset
from library.model import DynamicDepthWideStreamBiGRU
from library.loss import MaskedMSELoss
from library.metrics import compute_mcrmse
from library.engine import Engine


def main():
    print("=== Starting Demonstration of RNA Degradation Library ===")

    # --------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Set DEBUG to True to load only a subset of data (100 samples)
    Config.DEBUG = True

    # Reduce model complexity for quick CPU/GPU execution
    Config.HIDDEN_DIM = 64
    Config.EMBED_DIM = 32
    Config.NUM_LAYERS = 2

    # Reduce training parameters
    Config.BATCH_SIZE = 8
    Config.EPOCHS = 2
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Setup directories and seeds
    Config.setup()

    # Clean up previous cache to ensure we process the debug subset
    for mode in ["train", "val", "test"]:
        cache_path = os.path.join(Config.WORKING_DIR, f"{mode}_data.npz")
        if os.path.exists(cache_path):
            os.remove(cache_path)

    print("Configuration updated: DEBUG=True, Small Model, Short Training.")

    # --------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # --------------------------------------------------------------------------
    print("\n[2] Testing Data Loading...")

    # Load dataloaders (this will trigger preprocessing of the subset)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch one batch to verify structure
    seq, loop, dist, targets, mask = next(iter(train_loader))

    print(
        f"Batch shapes -> Seq: {seq.shape}, Targets: {targets.shape}, Mask: {mask.shape}"
    )

    # Assertions
    assert seq.shape == (Config.BATCH_SIZE, Config.SEQ_LEN), "Incorrect sequence shape"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), "Incorrect target shape"
    assert mask.shape == (Config.BATCH_SIZE, Config.SEQ_LEN), "Incorrect mask shape"
    assert mask.dtype == torch.bool, "Mask should be boolean"

    # Verify mask logic: First 68 should be True, rest False (based on Config.PRED_LEN)
    assert torch.all(
        mask[:, : Config.PRED_LEN]
    ), "First 68 positions should be masked True"
    assert not torch.any(
        mask[:, Config.PRED_LEN :]
    ), "Positions > 68 should be masked False"

    print("Data Loading verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Model Architecture Demonstration
    # --------------------------------------------------------------------------
    print("\n[3] Testing Model Forward Pass...")

    device = torch.device(Config.DEVICE)
    model = DynamicDepthWideStreamBiGRU().to(device)

    # Move batch to device
    seq = seq.to(device)
    loop = loop.to(device)
    dist = dist.to(device)
    targets = targets.to(device)
    mask = mask.to(device)

    # Forward pass
    preds = model(seq, loop, dist, mask)

    print(f"Output shape: {preds.shape}")

    # Assertions
    assert preds.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), "Model output shape mismatch"
    assert not torch.isnan(preds).any(), "Model produced NaNs"

    print("Model forward pass verified successfully.")

    # --------------------------------------------------------------------------
    # 4. Loss Function Demonstration
    # --------------------------------------------------------------------------
    print("\n[4] Testing Loss Calculation...")

    criterion = MaskedMSELoss()
    loss = criterion(preds, targets, mask)

    print(f"Calculated Loss: {loss.item():.6f}")

    # Assertions
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() >= 0, "MSE Loss cannot be negative"
    assert loss.requires_grad, "Loss must require gradients for training"

    print("Loss function verified successfully.")

    # --------------------------------------------------------------------------
    # 5. Metrics Demonstration
    # --------------------------------------------------------------------------
    print("\n[5] Testing MCRMSE Metric...")

    # Create dummy predictions and targets to test metric logic manually
    # Case 1: Perfect prediction
    dummy_preds = torch.ones((2, 107, 3)) * 0.5
    dummy_targets = torch.ones((2, 107, 3)) * 0.5
    dummy_mask = torch.zeros((2, 107), dtype=torch.bool)
    dummy_mask[:, :68] = True

    score_perfect = compute_mcrmse(dummy_preds, dummy_targets, dummy_mask)
    assert (
        score_perfect == 0.0
    ), f"Perfect prediction should have 0 error, got {score_perfect}"

    # Case 2: Known error
    # Pred = 0.5, Target = 1.5 -> Diff = 1.0 -> SqErr = 1.0 -> RMSE = 1.0
    dummy_targets_off = torch.ones((2, 107, 3)) * 1.5
    score_off = compute_mcrmse(dummy_preds, dummy_targets_off, dummy_mask)
    assert abs(score_off - 1.0) < 1e-6, f"Expected error 1.0, got {score_off}"

    # Real batch metric
    batch_score = compute_mcrmse(preds, targets, mask)
    print(f"Batch MCRMSE: {batch_score:.6f}")

    print("Metric calculation verified successfully.")

    # --------------------------------------------------------------------------
    # 6. Full Engine Pipeline (Train & Predict)
    # --------------------------------------------------------------------------
    print("\n[6] Running Full Engine Pipeline...")

    # Re-initialize Engine to use the configured model and settings
    engine = Engine()

    # Run Training Loop
    print("Starting fit()...")
    engine.fit(
        train_loader, val_loader, epochs=Config.EPOCHS, early_stopping_patience=2
    )

    # Check if checkpoint was created
    assert os.path.exists(Config.CHECKPOINT_PATH), "Checkpoint file was not created."
    print("Training complete. Checkpoint verified.")

    # Run Inference
    print("Starting predict()...")
    engine.predict(test_loader)

    # Verify Submission
    assert os.path.exists(
        Config.FINAL_SUBMISSION_PATH
    ), "Submission file was not created."

    df_sub = pd.read_csv(Config.FINAL_SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    print(f"Submission columns: {list(df_sub.columns)}")

    # Verify submission content
    required_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(df_sub.columns) == required_cols, "Submission columns mismatch"

    # Verify row count
    # In Debug mode, test_loader loads 100 samples (or less if file is small).
    # The test.json has 240 samples. head(100) -> 100 samples.
    # Each sample has 107 positions.
    # Expected rows = 100 * 107 = 10700
    expected_rows = 100 * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df_sub)}"

    # Verify unscored columns are 0.0
    assert (df_sub["deg_pH10"] == 0.0).all(), "deg_pH10 should be 0.0"
    assert (df_sub["deg_50C"] == 0.0).all(), "deg_50C should be 0.0"

    print("Pipeline executed and submission verified successfully.")
    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
