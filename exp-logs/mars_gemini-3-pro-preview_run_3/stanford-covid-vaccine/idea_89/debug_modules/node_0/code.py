import sys
import os
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, mcrmse_metric, get_scored_indices
from library.data import get_dataloaders
from library.model import RNARegressor
from library.train import train_one_epoch, validate

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("==== RNA Degradation Prediction Pipeline Demo ====")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Purposes
    # ---------------------------------------------------------
    print("\n[1] Configuring environment...")

    # Enable debug mode to use a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Use only 20 samples for speed

    # Training hyperparameters for quick execution
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Disable multiprocessing to avoid overhead in demo

    # Set paths to a specific demo directory in working
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    # Cache files specific to this demo run
    Config.TRAIN_CACHE_PATH = os.path.join(Config.WORKING_DIR, "train_cache.npy")
    Config.VAL_CACHE_PATH = os.path.join(Config.WORKING_DIR, "val_cache.npy")
    Config.TEST_CACHE_PATH = os.path.join(Config.WORKING_DIR, "test_cache.npy")

    # Force reprocessing to demonstrate data pipeline logic (optional, but good for demo)
    Config.LOAD_CACHED_DATA = False

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Data Loading and Verification
    # ---------------------------------------------------------
    print("\n[2] Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders()

    print("    Verifying Train Loader Batch...")
    batch = next(iter(train_loader))

    # Extract components
    inputs = batch["inputs"]
    adj_indices = batch["adjacency_indices"]
    adj_mask = batch["adjacency_mask"]
    targets = batch["targets"]

    # Assertions for shapes
    # Inputs: (Batch, SeqLen, 14)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        107,
        14,
    ), f"Input shape mismatch: {inputs.shape}"
    # Targets: (Batch, SeqLen, 5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), f"Target shape mismatch: {targets.shape}"
    # Adjacency: (Batch, SeqLen)
    assert adj_indices.shape == (
        Config.BATCH_SIZE,
        107,
    ), f"Adj indices shape mismatch: {adj_indices.shape}"

    print("    Data shapes verified successfully.")

    # ---------------------------------------------------------
    # 3. Model Initialization and Forward Pass
    # ---------------------------------------------------------
    print("\n[3] Initializing Model...")
    model = RNARegressor().to(device)

    print("    Running Forward Pass check...")
    with torch.no_grad():
        # Move batch to device
        b_in = inputs.to(device)
        b_idx = adj_indices.to(device)
        b_msk = adj_mask.to(device)

        outputs = model(b_in, b_idx, b_msk)

    assert outputs.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), f"Output shape mismatch: {outputs.shape}"

    print("    Forward pass successful. Output shape: (Batch, 107, 5)")

    # ---------------------------------------------------------
    # 4. Training Loop Demonstration
    # ---------------------------------------------------------
    print("\n[4] Running Training Loop (2 Epochs)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        print(
            f"    Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val MCRMSE = {val_score:.4f}"
        )

    # Save the model
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print(f"    Model saved to {Config.BEST_MODEL_PATH}")

    # ---------------------------------------------------------
    # 5. Metric Logic Verification
    # ---------------------------------------------------------
    print("\n[5] Verifying Metric Logic...")
    # Create synthetic data
    # Case: Prediction is exactly 0.5 away from Target for all scored positions
    # Scored length is 68.
    y_true_syn = torch.zeros(2, 107, 5)
    y_pred_syn = torch.ones(2, 107, 5) * 0.5

    # We only score specific columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = get_scored_indices()  # [0, 1, 3]

    # Expected calculation:
    # Error = 0.5 -> Squared Error = 0.25 -> MSE = 0.25 -> RMSE = 0.5
    # Average across columns = 0.5

    metric_val = mcrmse_metric(
        y_true_syn, y_pred_syn, seq_scored=68, target_indices=scored_indices
    )

    print(f"    Calculated Metric: {metric_val:.4f}")
    assert (
        abs(metric_val - 0.5) < 1e-4
    ), f"Metric verification failed. Expected 0.5, got {metric_val}"
    print("    Metric logic verified.")

    # ---------------------------------------------------------
    # 6. Inference and Submission Generation
    # ---------------------------------------------------------
    print("\n[6] Generating Submission...")
    model.eval()

    all_ids = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["inputs"].to(device)
            adj_indices = batch["adjacency_indices"].to(device)
            adj_mask = batch["adjacency_mask"].to(device)
            ids = batch["id"]

            outputs = model(inputs, adj_indices, adj_mask)

            all_preds.append(outputs.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate predictions: (N_test, 107, 5)
    preds_arr = np.concatenate(all_preds, axis=0)

    # Flatten to submission format
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    print("    Formatting submission file...")

    submission_rows = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(all_ids):
        sample_pred = preds_arr[i]  # Shape (107, 5)

        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_pred[seqpos].tolist()

            row_data = [row_id] + row_values
            submission_rows.append(row_data)

    # Create DataFrame
    sub_df = pd.DataFrame(submission_rows, columns=["id_seqpos"] + target_cols)

    # Save
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"    Submission saved to {Config.SUBMISSION_PATH}")
    print(f"    Submission shape: {sub_df.shape}")

    # Verify submission integrity
    assert sub_df.shape[1] == 6, "Submission must have 6 columns"
    assert not sub_df.isnull().values.any(), "Submission contains NaN values"

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    run_demo()
