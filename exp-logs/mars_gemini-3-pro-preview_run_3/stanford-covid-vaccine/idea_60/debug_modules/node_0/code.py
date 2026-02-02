import os
import torch
import numpy as np
import pandas as pd
import sys

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, scored_mcrmse
from library.data import get_dataloaders
from library.model import HCSDBR_BiGRU
from library.loss import MCRMSELoss
from library.train import run_training, Trainer


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration Overrides for Speed
    # We override Config attributes to run a fast demo (debug mode)
    print("\n[1] Configuring environment for fast demonstration...")
    set_seed(42)

    # Modify Config for this run
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Only use 50 samples
    Config.BATCH_SIZE = 8
    Config.NUM_EPOCHS = 1  # Train for only 1 epoch
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    print(f"Working directory: {Config.WORKING_DIR}")
    print(f"Debug mode: {Config.DEBUG}")
    print(f"Subset size: {Config.DEBUG_SUBSET_SIZE}")

    # 2. Data Loading Verification
    print("\n[2] Verifying Data Loading...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,  # Force processing to demonstrate pipeline
        debug=Config.DEBUG,
    )

    # Fetch one batch to verify shapes
    batch = next(iter(train_loader))
    inputs = batch["inputs"]
    targets = batch["targets"]
    adj = batch["adjacency"]
    mask = batch["mask"]
    ids = batch["ids"]

    print(f"Batch keys: {list(batch.keys())}")
    print(f"Inputs shape: {inputs.shape} (Expected: B, 107, 14)")
    print(f"Targets shape: {targets.shape} (Expected: B, 107, 5)")
    print(f"Adjacency shape: {adj.shape} (Expected: B, 107)")

    # Assertions
    assert inputs.shape[1] == 107, "Sequence length mismatch in inputs"
    assert inputs.shape[2] == 14, "Feature dimension mismatch in inputs"
    assert targets.shape[2] == 5, "Target dimension mismatch"
    assert adj.shape == (inputs.shape[0], 107), "Adjacency matrix shape mismatch"
    assert mask.shape == (inputs.shape[0], 107), "Mask shape mismatch"
    print("Data loading verification passed.")

    # 3. Model Architecture Verification
    print("\n[3] Verifying Model Architecture...")
    device = Config.DEVICE
    model = HCSDBR_BiGRU().to(device)

    # Move batch to device
    inputs_dev = inputs.to(device)
    adj_dev = adj.to(device)
    mask_dev = mask.to(device)

    # Forward pass
    outputs = model(inputs_dev, adj_dev, mask_dev)

    print(f"Output shape: {outputs.shape} (Expected: B, 107, 5)")

    # Assertions
    assert outputs.shape == (inputs.shape[0], 107, 5), "Model output shape is incorrect"
    assert not torch.isnan(outputs).any(), "Model produced NaN values"
    print("Model forward pass verification passed.")

    # 4. Loss and Metric Verification
    print("\n[4] Verifying Loss and Metric functions...")
    criterion = MCRMSELoss()
    targets_dev = targets.to(device)

    # Calculate Loss
    loss = criterion(outputs, targets_dev)
    print(f"Calculated Loss: {loss.item():.6f}")

    # Assertions
    assert loss.item() >= 0, "Loss should be non-negative"
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"

    # Calculate Metric (Scored MCRMSE)
    # Note: scored_mcrmse expects inputs on CPU usually, but handles it internally
    metric = scored_mcrmse(targets_dev, outputs)
    print(f"Calculated Metric: {metric:.6f}")

    assert isinstance(metric, float), "Metric should return a float"
    assert metric >= 0, "Metric should be non-negative"
    print("Loss and Metric verification passed.")

    # 5. Training Loop Execution
    print("\n[5] Executing Training Loop (1 Epoch)...")
    # We use the provided run_training function
    # Note: We set load_cached_data=True now because we just processed it in step 2
    run_training(load_cached_data=True, debug=Config.DEBUG, epochs=Config.NUM_EPOCHS)

    # Verify model checkpoint exists
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Model checkpoint found at {Config.MODEL_SAVE_PATH}")
    else:
        # It's possible validation metric didn't improve if initialized randomly poorly,
        # but in a demo script usually at least one save happens or we check logic.
        # However, Trainer saves only if val_metric < best_score (inf). So it should save after epoch 0.
        print(
            "Warning: Model checkpoint not found (Validation might not have run or improved)."
        )

    # 6. Inference and Submission Generation
    print("\n[6] Demonstrating Inference and Submission Generation...")

    # Load best model
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print("Loaded best model state.")
    else:
        print("Using current model state (checkpoint missing).")

    model.eval()

    preds_list = []
    ids_list = []

    print("Running inference on Test set...")
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            adj = batch["adjacency"].to(device)
            mask = batch["mask"].to(device)
            ids = batch["ids"]

            # Forward
            outputs = model(inputs, adj, mask)  # (B, 107, 5)

            # Move to CPU
            preds_np = outputs.cpu().numpy()

            preds_list.append(preds_np)
            ids_list.extend(ids)

    # Concatenate predictions
    preds_all = np.concatenate(preds_list, axis=0)  # (N_test, 107, 5)

    print(f"Total Test Predictions shape: {preds_all.shape}")

    # Format for submission
    # We need to flatten: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # For each sample, we have 107 positions.
    # Note: The competition submission requires prediction for all positions,
    # but only seq_scored are scored. The sample submission format implies 1 row per pos.

    submission_data = []
    target_cols = (
        Config.TARGET_COLS
    )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids_list):
        sample_preds = preds_all[i]  # (107, 5)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_preds = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for t_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_preds[t_idx]

            submission_data.append(row_dict)

    submission_df = pd.DataFrame(submission_data)

    # Verify Submission
    print(f"Submission DataFrame shape: {submission_df.shape}")
    expected_rows = len(ids_list) * Config.SEQ_LEN
    assert (
        len(submission_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(submission_df)}"

    # Save submission
    sub_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    submission_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
