import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import library components
from library.config import Config
from library.utils import seed_everything, MCRMSE, get_device
from library.dataset import get_dataloader, load_and_process_data
from library.model import RNAModel
from library.engine import train_and_evaluate, get_predictions


def run_demo():
    print("=== RNA Degradation Prediction Pipeline Demo ===")

    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # -------------------------------------------------------------------------
    print("\n[1] Setting up configuration for demo run...")

    # Define a temporary working directory for this demo
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths and parameters for speed
    Config.WORKING_DIR = DEMO_DIR
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Reduce compute load for demonstration
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead/issues in demo
    Config.HIDDEN_DIM = 64  # Smaller model for speed
    Config.EMBED_DIM = 32
    Config.NUM_LAYERS = 2

    # Set global seed
    seed_everything(Config.SEED)
    device = get_device()
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Data Preparation (Mini-Subsets)
    # -------------------------------------------------------------------------
    print("\n[2] Creating mini-datasets from metadata...")

    # Load a small slice of the provided metadata to create a fast demo dataset
    # We assume ./metadata/train.parquet exists as per task description
    full_train = pd.read_parquet("./metadata/train.parquet")
    full_val = pd.read_parquet("./metadata/val.parquet")
    full_test = pd.read_parquet("./metadata/test.parquet")

    # Take top 20 samples for train, 10 for val, 10 for test
    mini_train = full_train.head(20).copy()
    mini_val = full_val.head(10).copy()
    mini_test = full_test.head(10).copy()

    # Save these mini files to the demo directory
    mini_train_path = os.path.join(DEMO_DIR, "mini_train.parquet")
    mini_val_path = os.path.join(DEMO_DIR, "mini_val.parquet")
    mini_test_path = os.path.join(DEMO_DIR, "mini_test.parquet")

    mini_train.to_parquet(mini_train_path, index=False)
    mini_val.to_parquet(mini_val_path, index=False)
    mini_test.to_parquet(mini_test_path, index=False)

    # Point Config to these new mini files
    Config.TRAIN_PATH = mini_train_path
    Config.VAL_PATH = mini_val_path
    Config.TEST_PATH = mini_test_path

    print(
        f"    Created mini_train ({len(mini_train)}), mini_val ({len(mini_val)}), mini_test ({len(mini_test)})"
    )

    # -------------------------------------------------------------------------
    # 3. Component Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Library Components...")

    # --- A. DataLoader Verification ---
    print("    A. Verifying DataLoader...")
    # Force reload from scratch by setting load_cached_data=False initially or ensuring cache doesn't exist
    train_loader = get_dataloader("train", shuffle=True, load_cached_data=False)

    # Fetch one batch
    seq, loop, dist, targets = next(iter(train_loader))

    # Assertions
    assert seq.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Seq shape mismatch: {seq.shape}"
    assert loop.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Loop shape mismatch: {loop.shape}"
    assert dist.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Dist shape mismatch: {dist.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Targets shape mismatch: {targets.shape}"

    # Verify that targets beyond SEQ_SCORED (68) are 0 (masked/padded in dataset.py)
    # The dataset class fills 0s for positions >= seq_scored
    assert torch.all(
        targets[:, Config.SEQ_SCORED :, :] == 0
    ), "Targets should be 0 for unscored positions."
    print("       DataLoader shapes correct.")

    # --- B. Model Verification ---
    print("    B. Verifying RNAModel...")
    model = RNAModel(Config)
    model.to(device)

    # Move batch to device
    seq = seq.to(device)
    loop = loop.to(device)
    dist = dist.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(seq, loop, dist)

    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Model output shape mismatch: {outputs.shape}"
    print("       Model forward pass successful.")

    # --- C. Metric Verification ---
    print("    C. Verifying MCRMSE Metric...")
    metric_fn = MCRMSE()

    # Create dummy data:
    # Batch=2, Seq=1 (simplified), Targets=3
    # Case 1: Perfect prediction -> Score 0
    y_true = torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    y_pred = torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])
    score = metric_fn(y_true, y_pred)
    assert torch.isclose(score, torch.tensor(0.0)), f"Expected 0.0, got {score}"

    # Case 2: Known error
    # Col 1 error: 1.0 (MSE=1, RMSE=1)
    # Col 2 error: 0.0
    # Col 3 error: 0.0
    # MCRMSE = (1 + 0 + 0) / 3 = 0.3333...
    y_pred_err = torch.tensor(
        [[[2.0, 2.0, 3.0]], [[5.0, 5.0, 6.0]]]
    )  # +1 error on first col
    score_err = metric_fn(y_true, y_pred_err)
    expected_score = 1.0 / 3.0
    assert torch.isclose(
        score_err, torch.tensor(expected_score), atol=1e-5
    ), f"Expected {expected_score}, got {score_err}"
    print("       Metric calculation logic verified.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[4] Running Training Loop...")

    # Get Val Loader
    val_loader = get_dataloader("val", shuffle=False, load_cached_data=False)

    # Run Engine
    train_and_evaluate(model, train_loader, val_loader, patience=2)

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Best model file was not saved."
    print("    Training completed and model saved.")

    # -------------------------------------------------------------------------
    # 5. Inference and Submission Generation
    # -------------------------------------------------------------------------
    print("\n[5] Generating Predictions and Submission...")

    # Load Test Data
    test_loader = get_dataloader("test", shuffle=False, load_cached_data=False)

    # Load Best Model
    best_model = RNAModel(Config)
    best_model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    best_model.to(device)

    # Get Predictions
    preds_tensor, sample_ids = get_predictions(best_model, test_loader)

    # preds_tensor shape: (N_Samples, 107, 3)
    # We need to format this to the submission format:
    # id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # Note: The model predicts 3 targets: reactivity, deg_Mg_pH10, deg_Mg_50C.
    # The submission requires 5 columns. We will fill the missing ones with 0 or copy as needed.
    # The task description says: "While the submission format requires all 5 to be predicted,
    # only the following are scored: reactivity, deg_Mg_pH10, and deg_Mg_50C."

    print(f"    Raw predictions shape: {preds_tensor.shape}")

    # Prepare data for DataFrame
    submission_data = []

    # The predicted columns correspond to Config.TARGET_COLS: ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    # We need to map these to the 5 required columns.
    # Required: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    preds_np = preds_tensor.numpy()

    for i, sample_id in enumerate(sample_ids):
        sample_preds = preds_np[i]  # Shape (107, 3)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"

            # Extract predictions
            reactivity = sample_preds[seqpos, 0]
            deg_Mg_pH10 = sample_preds[seqpos, 1]
            deg_Mg_50C = sample_preds[seqpos, 2]

            # Fill unscored/unpredicted columns with 0.0 (or dummy values)
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_data.append(
                [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            )

    # Create DataFrame
    submission_df = pd.DataFrame(
        submission_data,
        columns=[
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ],
    )

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)

    # Verification
    assert (
        len(submission_df) == len(mini_test) * Config.SEQ_LEN
    ), f"Submission rows mismatch. Expected {len(mini_test) * Config.SEQ_LEN}, got {len(submission_df)}"

    print(f"    Submission saved to {Config.SUBMISSION_PATH}")
    print(f"    First 3 rows:\n{submission_df.head(3)}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
