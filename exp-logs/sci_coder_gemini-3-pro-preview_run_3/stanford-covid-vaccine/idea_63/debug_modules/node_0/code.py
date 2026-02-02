import os
import torch
import pandas as pd
import numpy as np
import sys

# Import library modules
from library.config import Config
from library import utils, data, model, train


def run_demo():
    print("=== Starting RNA Degradation Prediction Demo ===")

    # 1. Setup and Configuration Override for Speed
    # We override the Config class attributes to run a fast debug session
    print("\n[1] Configuring environment for fast demonstration...")
    utils.seed_everything(Config.SEED)

    # Patch Config for speed
    Config.EPOCHS = 2
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 100  # Small subset for quick processing
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure clean state for debug run by forcing reprocessing
    load_cached = False

    print(f"    Device: {Config.DEVICE}")
    print(f"    Epochs: {Config.EPOCHS}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # 2. Data Loading and Verification
    print("\n[2] Loading Data...")
    train_loader, val_loader, test_loader = data.get_dataloaders(
        load_cached_data=load_cached, debug=Config.DEBUG
    )

    # Verify Train Loader
    print("    Verifying Train Loader batch structure...")
    batch = next(iter(train_loader))

    inputs = batch["inputs"]
    pair_indices = batch["pair_indices"]
    pair_mask = batch["pair_mask"]
    targets = batch["targets"]

    # Assertions for shapes
    # Inputs: (Batch, Seq_Len=107, Channels=14)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_CHANNELS,
    ), f"Input shape mismatch: {inputs.shape}"
    # Pair Indices: (Batch, Seq_Len=107)
    assert pair_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Pair indices shape mismatch: {pair_indices.shape}"
    # Targets: (Batch, Seq_Len=107, Targets=5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Targets shape mismatch: {targets.shape}"

    print("    Data shapes verified successfully.")

    # 3. Model Initialization and Forward Pass Verification
    print("\n[3] Initializing Model and Verifying Forward Pass...")
    net = model.HC_SDBR_BiGRU().to(Config.DEVICE)

    # Move batch to device
    inputs_dev = inputs.to(Config.DEVICE)
    pidx_dev = pair_indices.to(Config.DEVICE)
    pmask_dev = pair_mask.to(Config.DEVICE)
    targets_dev = targets.to(Config.DEVICE)

    # Forward pass
    outputs = net(inputs_dev, pidx_dev, pmask_dev)

    # Verify output shape: (Batch, Seq_Len, Num_Targets)
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Model output shape mismatch: {outputs.shape}"

    print("    Forward pass successful. Output shape verified.")

    # 4. Loss Calculation Verification
    print("\n[4] Verifying Loss Calculation...")
    criterion = utils.MCRMSELoss()
    loss = criterion(outputs, targets_dev)

    assert loss.dim() == 0, "Loss should be a scalar."
    assert loss.item() >= 0, "Loss should be non-negative."

    print(f"    Initial Loss: {loss.item():.6f}")

    # 5. Training Loop Execution
    print("\n[5] Executing Training Loop (Trainer Class)...")

    optimizer = torch.optim.AdamW(
        net.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    trainer = train.Trainer(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
    )

    # Run training
    trainer.fit(epochs=Config.EPOCHS)

    # Verify best model was saved
    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not created."
    print("    Training complete. Best model saved.")

    # 6. Submission Generation
    print("\n[6] Generating Submission...")
    train.generate_submission(net, test_loader, Config.DEVICE)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission loaded. Shape: {sub_df.shape}")

    # Expected rows: Number of test samples * Seq Length
    # Test set in metadata has 240 samples.
    expected_rows = 240 * Config.SEQ_LEN
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Expected columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    print("    Submission format verified successfully.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
