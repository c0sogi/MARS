import os
import torch
import numpy as np
import pandas as pd
import time
from library import config, data, model, utils, train


def main():
    print("=== Starting Demonstration of RNA Degradation Prediction Library ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("[Step 1] Configuring environment for rapid demonstration...")

    # Enable debug mode to use a tiny subset of data
    config.DEBUG = True
    config.DEBUG_SUBSET_SIZE = 40  # Enough for >2 batches given BATCH_SIZE=16

    # Reduce training duration
    config.NUM_EPOCHS = 1
    config.PATIENCE = 1

    # Ensure we force reprocessing of data to apply the DEBUG subsetting
    # We define new cache paths to avoid conflicts with existing full caches
    config.CACHE_TRAIN = os.path.join(config.WORKING_DIR, "demo_train.npz")
    config.CACHE_VAL = os.path.join(config.WORKING_DIR, "demo_val.npz")
    config.CACHE_TEST = os.path.join(config.WORKING_DIR, "demo_test.npz")
    config.MODEL_PATH = os.path.join(config.WORKING_DIR, "demo_model.pth")
    config.SUBMISSION_PATH = os.path.join(config.WORKING_DIR, "demo_submission.csv")

    # Set seed
    utils.seed_everything(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Configuration updated. Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # -------------------------------------------------------------------------
    print("\n[Step 2] Loading DataLoaders...")

    # Load data (load_cached_data=False forces reprocessing the debug subset)
    train_loader, val_loader, test_loader = data.get_dataloaders(load_cached_data=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Fetch one batch to verify structure
    inputs, partner_indices, targets, ids = next(iter(train_loader))

    print("\n[Step 3] Verifying Data Shapes...")
    print(f"Inputs shape: {inputs.shape}")  # Expected: (16, 107, 18)
    print(f"Partner Indices shape: {partner_indices.shape}")  # Expected: (16, 107)
    print(f"Targets shape: {targets.shape}")  # Expected: (16, 107, 5)

    # Assertions
    assert inputs.shape == (
        config.BATCH_SIZE,
        config.SEQ_LENGTH,
        18,
    ), "Incorrect input shape"
    assert partner_indices.shape == (
        config.BATCH_SIZE,
        config.SEQ_LENGTH,
    ), "Incorrect partner indices shape"
    assert targets.shape == (
        config.BATCH_SIZE,
        config.SEQ_LENGTH,
        config.NUM_TARGETS,
    ), "Incorrect target shape"
    assert len(ids) == config.BATCH_SIZE, "Incorrect number of IDs"
    print("Data validation passed.")

    # -------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # -------------------------------------------------------------------------
    print("\n[Step 4] Initializing HC_HIDN Model...")
    net = model.HC_HIDN().to(device)

    print("Running forward pass on a single batch...")
    inputs = inputs.to(device)
    partner_indices = partner_indices.to(device)
    targets = targets.to(device)

    # The model returns two outputs: y1 (Pass 1) and y2 (Pass 2 - Refined)
    y1, y2 = net(inputs, partner_indices)

    print(f"Output y1 shape: {y1.shape}")
    print(f"Output y2 shape: {y2.shape}")

    # Assertions
    expected_out_shape = (config.BATCH_SIZE, config.SEQ_LENGTH, config.NUM_TARGETS)
    assert y1.shape == expected_out_shape, "Model output y1 shape mismatch"
    assert y2.shape == expected_out_shape, "Model output y2 shape mismatch"

    # -------------------------------------------------------------------------
    # 4. Loss Calculation
    # -------------------------------------------------------------------------
    print("\n[Step 5] Calculating Loss...")
    loss_val = utils.mcrmse_loss(y2, targets)
    print(f"Calculated MCRMSE Loss: {loss_val.item():.6f}")

    assert torch.is_tensor(loss_val), "Loss should be a tensor"
    assert loss_val.ndim == 0, "Loss should be a scalar"
    assert not torch.isnan(loss_val), "Loss is NaN"

    # -------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 6] Running Training Epoch...")
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.LEARNING_RATE)

    # Run one epoch
    start_time = time.time()
    avg_train_loss = train.train_epoch(net, train_loader, optimizer, device)
    duration = time.time() - start_time

    print(f"Epoch finished in {duration:.2f}s. Avg Train Loss: {avg_train_loss:.6f}")

    # Run validation
    print("Running Validation...")
    val_metric = train.validate(net, val_loader, device)
    print(f"Validation Global MCRMSE: {val_metric:.6f}")

    # Save this 'trained' model to demonstrate checkpointing
    torch.save(net.state_dict(), config.MODEL_PATH)
    assert os.path.exists(config.MODEL_PATH), "Model file was not saved."

    # -------------------------------------------------------------------------
    # 6. Inference & Submission
    # -------------------------------------------------------------------------
    print("\n[Step 7] Generating Submission...")

    # Reload model to ensure state dict loading works
    net.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))

    # Generate submission
    train.generate_submission(net, test_loader, device)

    # Verify submission file
    if os.path.exists(config.SUBMISSION_PATH):
        df_sub = pd.read_csv(config.SUBMISSION_PATH)
        print(f"Submission file loaded. Shape: {df_sub.shape}")
        print("First 3 rows:")
        print(df_sub.head(3))

        # Validation of submission format
        # Expected rows: Number of test samples * Seq Length
        # Note: test_loader uses the DEBUG subset of test.json
        expected_rows = len(test_loader.dataset) * config.SEQ_LENGTH
        assert (
            len(df_sub) == expected_rows
        ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

        # Check columns
        expected_cols = ["id_seqpos"] + config.TARGET_COLS
        assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

        # Check value types (should be float)
        assert pd.api.types.is_float_dtype(
            df_sub["reactivity"]
        ), "Reactivity column should be float"

        print("Submission format verification passed.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
