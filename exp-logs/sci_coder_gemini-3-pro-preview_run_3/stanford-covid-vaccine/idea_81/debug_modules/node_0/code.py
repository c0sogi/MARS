import os
import sys
import torch
import numpy as np
import pandas as pd

# Ensure the current directory is in the path to import library modules
sys.path.append(".")

from library.config import Config
from library.utils import set_seed, mcrmse
from library.data import get_dataloaders
from library.model import DeepResBiGRU
from library.engine import Trainer


def main():
    print("==== RNA Degradation Prediction Demo ====")

    # ----------------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # ----------------------------------------------------------------
    print("\n[1/6] Setting up configuration...")

    # Define a specific directory for this demo execution
    demo_dir = "./working/demo_execution"
    cache_dir = os.path.join(demo_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Override Config parameters for speed and isolation
    Config.WORKING_DIR = cache_dir
    Config.MODEL_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size for debug
    Config.DEBUG_SUBSET_SIZE = 50  # Only use 50 samples
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(f"      Working directory set to: {Config.WORKING_DIR}")
    print(f"      Epochs: {Config.EPOCHS}, Batch Size: {Config.BATCH_SIZE}")

    # ----------------------------------------------------------------
    # 2. Data Loading
    # ----------------------------------------------------------------
    print("\n[2/6] Loading data (Debug Mode)...")

    # Load dataloaders with debug=True to use a small subset
    # load_cached_data=False forces processing to demonstrate the pipeline
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    print(f"      Train batches: {len(train_loader)}")
    print(f"      Val batches:   {len(val_loader)}")
    print(f"      Test batches:  {len(test_loader)}")

    # Verify Data Shapes
    batch = next(iter(train_loader))
    inputs = batch["input"]
    adj = batch["adjacency"]
    mask = batch["pair_mask"]
    targets = batch["target"]
    ids = batch["id"]

    print("      Verifying tensor shapes...")
    # Inputs: (Batch, Seq_Len=107, Channels=14)
    assert (
        inputs.ndim == 3 and inputs.shape[1] == 107 and inputs.shape[2] == 14
    ), f"Incorrect input shape: {inputs.shape}"

    # Adjacency: (Batch, Seq_Len=107)
    assert (
        adj.ndim == 2 and adj.shape[1] == 107
    ), f"Incorrect adjacency shape: {adj.shape}"

    # Targets: (Batch, Seq_Scored=68, Num_Targets=5)
    assert (
        targets.ndim == 3 and targets.shape[1] == 68 and targets.shape[2] == 5
    ), f"Incorrect target shape: {targets.shape}"

    print("      Data shapes verified successfully.")

    # ----------------------------------------------------------------
    # 3. Model Initialization & Forward Pass Verification
    # ----------------------------------------------------------------
    print("\n[3/6] Initializing Model and Verifying Forward Pass...")

    model = DeepResBiGRU()
    device = Config.DEVICE
    model.to(device)

    # Move batch to device
    inputs = inputs.to(device)
    adj = adj.to(device)
    mask = mask.to(device)

    # Run dummy forward pass
    with torch.no_grad():
        outputs = model(inputs, adj, mask)

    # Expected Output: (Batch, Seq_Len=107, Num_Targets=5)
    # Note: The model outputs predictions for the full sequence length (107),
    # even though ground truth is only available for the first 68.
    assert outputs.shape == (
        inputs.shape[0],
        107,
        5,
    ), f"Output shape mismatch. Expected {(inputs.shape[0], 107, 5)}, got {outputs.shape}"

    print(f"      Forward pass successful. Output shape: {outputs.shape}")

    # ----------------------------------------------------------------
    # 4. Metric Logic Verification
    # ----------------------------------------------------------------
    print("\n[4/6] Verifying Metric Calculation (MCRMSE)...")

    # Create synthetic ground truth (B, 68, 5) and predictions (B, 107, 5)
    # The metric function should handle the length mismatch by slicing
    y_true_dummy = torch.rand(4, 68, 5)
    y_pred_dummy = torch.rand(4, 107, 5)

    score = mcrmse(y_true_dummy, y_pred_dummy, only_scored=True)

    assert isinstance(score.item(), float), "Metric did not return a scalar float."
    assert score.item() >= 0, "Metric returned negative value."
    print(f"      Metric calculation check passed. Score: {score.item():.4f}")

    # ----------------------------------------------------------------
    # 5. Training Loop
    # ----------------------------------------------------------------
    print("\n[5/6] Starting Training Loop...")

    trainer = Trainer(model, train_loader, val_loader)
    trainer.fit()

    # Verify model was saved
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")
    print("      Training complete. Best model saved.")

    # ----------------------------------------------------------------
    # 6. Inference and Submission Generation
    # ----------------------------------------------------------------
    print("\n[6/6] Generating Submission from Test Set...")

    # Load the best model state
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    preds_container = []
    ids_container = []

    print("      Running inference on test set...")
    with torch.no_grad():
        for batch in test_loader:
            b_inputs = batch["input"].to(device)
            b_adj = batch["adjacency"].to(device)
            b_mask = batch["pair_mask"].to(device)
            b_ids = batch["id"]

            # Forward pass
            b_outputs = model(b_inputs, b_adj, b_mask)

            # Store results (move to CPU)
            preds_container.append(b_outputs.cpu().numpy())
            ids_container.extend(b_ids)

    # Concatenate all predictions: (N_Test, 107, 5)
    all_preds = np.concatenate(preds_container, axis=0)

    # Format for submission
    # We need to flatten the predictions: one row per sequence position
    submission_rows = []
    target_cols = (
        Config.TARGET_COLS
    )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids_container):
        sample_pred = all_preds[i]  # Shape (107, 5)

        for seq_pos in range(sample_pred.shape[0]):
            row_id = f"{sample_id}_{seq_pos}"
            row_values = sample_pred[seq_pos]

            # Create dictionary for this row
            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_values[col_idx]

            submission_rows.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_rows)

    # Save to CSV
    sub_path = os.path.join(demo_dir, "submission_demo.csv")
    submission_df.to_csv(sub_path, index=False)

    print(f"      Submission saved to: {sub_path}")
    print(f"      Submission dimensions: {submission_df.shape}")

    # Final Validation
    assert (
        submission_df.shape[1] == 6
    ), "Submission should have 6 columns (id_seqpos + 5 targets)"
    assert (
        submission_df.shape[0] == len(ids_container) * 107
    ), "Incorrect number of rows in submission"

    print("\n==== Demo Execution Completed Successfully ====")


if __name__ == "__main__":
    main()
