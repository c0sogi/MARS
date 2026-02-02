import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, MCRMSELoss, metric_calculator
from library.data import get_train_val_datasets, get_test_dataset
from library.model import RNANet
from library.engine import train_model, generate_submission


def run_demo():
    # 1. Setup and Configuration Overrides for Speed
    print(">>> 1. Setting up environment and configuration...")
    seed_everything(42)

    # Override Config for a fast demo run
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.DEBUG_SUBSET_SIZE = 20  # Use only 20 samples for demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Data Loading Demonstration
    print("\n>>> 2. Loading Datasets (Debug Mode)...")
    # Load train/val datasets with debug=True to use the small subset
    train_dataset, val_dataset = get_train_val_datasets(
        load_cached_data=False, debug=True
    )

    # Assertions to verify data loading
    assert (
        len(train_dataset) == Config.DEBUG_SUBSET_SIZE
    ), f"Expected {Config.DEBUG_SUBSET_SIZE} training samples, got {len(train_dataset)}"
    assert (
        len(val_dataset) == Config.DEBUG_SUBSET_SIZE
    ), f"Expected {Config.DEBUG_SUBSET_SIZE} validation samples, got {len(val_dataset)}"

    # Check a single sample structure
    sample = train_dataset[0]
    assert "inputs" in sample and "adj_map" in sample and "targets" in sample
    assert sample["inputs"].shape == (
        107,
        14,
    ), f"Input shape mismatch: {sample['inputs'].shape}"
    assert sample["targets"].shape == (
        68,
        5,
    ), f"Target shape mismatch: {sample['targets'].shape}"

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )
    print("DataLoaders created successfully.")

    # 3. Model Initialization and Verification
    print("\n>>> 3. Initializing Model...")
    model = RNANet().to(device)

    # Test Forward Pass
    dummy_batch = next(iter(train_loader))
    inputs = dummy_batch["inputs"].to(device)
    adj_map = dummy_batch["adj_map"].to(device)

    with torch.no_grad():
        outputs = model(inputs, adj_map)

    # Output shape should be (Batch, Seq_Len=107, Targets=5)
    expected_shape = (Config.BATCH_SIZE, Config.SEQ_LEN, 5)
    assert (
        outputs.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {outputs.shape}"
    print("Model forward pass successful. Output shape verified.")

    # 4. Metric and Loss Verification
    print("\n>>> 4. Verifying Metrics and Loss...")
    criterion = MCRMSELoss()

    # Create synthetic predictions and targets
    # Preds: (B, 107, 5), Targets: (B, 68, 5)
    # Case 1: Perfect prediction (first 68 positions match)
    syn_targets = torch.rand(Config.BATCH_SIZE, Config.SEQ_SCORED, 5)
    syn_preds = torch.zeros(Config.BATCH_SIZE, Config.SEQ_LEN, 5)
    syn_preds[:, : Config.SEQ_SCORED, :] = syn_targets

    loss = criterion(syn_preds, syn_targets)
    # Loss should be 0.0 (or extremely close due to float precision)
    assert (
        loss.item() < 1e-6
    ), f"Loss should be near 0 for perfect predictions, got {loss.item()}"

    # Case 2: Known error
    # Add 1.0 to all predictions. RMSE should be 1.0.
    syn_preds_err = syn_preds.clone()
    syn_preds_err[:, : Config.SEQ_SCORED, :] += 1.0
    loss_err = criterion(syn_preds_err, syn_targets)
    assert (
        abs(loss_err.item() - 1.0) < 1e-5
    ), f"Loss should be 1.0, got {loss_err.item()}"

    # Verify Metric Calculator (Scored columns only)
    # metric_calculator expects numpy arrays
    score = metric_calculator(syn_preds.numpy(), syn_targets.numpy())
    assert score < 1e-6, f"Metric should be near 0 for perfect predictions, got {score}"
    print("Metric and Loss functions verified.")

    # 5. Training Loop Demonstration
    print("\n>>> 5. Running Training Loop...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    best_score = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device,
        num_epochs=Config.EPOCHS,
        patience=Config.PATIENCE,
    )

    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not saved."
    print(f"Training demo complete. Best Score: {best_score}")

    # 6. Inference and Submission
    print("\n>>> 6. Generating Submission...")
    # Load test data (debug mode)
    test_dataset = get_test_dataset(load_cached_data=False, debug=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load best model weights
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    # Generate submission
    generate_submission(model, test_loader, device)

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {df_sub.shape}")

    # Expected rows: Num_Test_Samples (20) * Seq_Len (107) = 2140
    expected_rows = Config.DEBUG_SUBSET_SIZE * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Check columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch."

    print("Submission verification passed.")
    print("\n>>> Demo completed successfully!")


if __name__ == "__main__":
    run_demo()
