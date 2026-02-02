import sys
import os
import torch
import numpy as np
import pandas as pd
import shutil

# Ensure the library modules can be imported
sys.path.append(".")

from library.config import Config
from library.utils import set_seed, mcrmse_loss
from library.data import get_dataloaders
from library.model import MultiTaskRNANet
from library.engine import Engine


def run_demo():
    # --------------------------------------------------------------------------
    # 1. Setup and Configuration Override
    # --------------------------------------------------------------------------
    print(">>> Setting up demonstration configuration...")

    # Set a fixed seed for reproducibility
    set_seed(42)

    # Modify Config for a fast demonstration run
    Config.PROJECT_NAME = "RNA_Demo_Run"
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples for speed
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Redirect outputs to a demo directory to avoid overwriting main work
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.create_dirs()

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # --------------------------------------------------------------------------
    # 2. Verify Metric Logic (MCRMSE)
    # --------------------------------------------------------------------------
    print("\n>>> Verifying MCRMSE Metric...")

    # Create dummy data: 2 samples, scored length 3, 2 targets
    # Shape: (2, 3, 2)
    y_true = np.array(
        [[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], [[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]]]
    )
    # Predict slightly off
    y_pred = np.array(
        [
            [[1.1, 1.9], [1.1, 1.9], [1.1, 1.9]],  # Diff: +0.1, -0.1
            [[3.1, 3.9], [3.1, 3.9], [3.1, 3.9]],  # Diff: +0.1, -0.1
        ]
    )

    # Manual Calculation
    # Flattened diffs per col:
    # Col 0 diffs: 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 -> MSE = 0.01 -> RMSE = 0.1
    # Col 1 diffs: -0.1, ... -> MSE = 0.01 -> RMSE = 0.1
    # Mean RMSE = 0.1

    score = mcrmse_loss(y_true, y_pred, num_scored=3)
    print(f"    Calculated Score: {score}")

    assert np.isclose(
        score, 0.1
    ), f"MCRMSE calculation incorrect. Expected 0.1, got {score}"
    print("    Metric verification passed.")

    # --------------------------------------------------------------------------
    # 3. Data Loading and Verification
    # --------------------------------------------------------------------------
    print("\n>>> Loading Data...")

    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False,  # Force processing from parquet
        batch_size=Config.BATCH_SIZE,
        debug=Config.DEBUG,
    )

    print("    DataLoaders created.")

    # Fetch a batch to verify structure
    batch = next(iter(train_loader))

    required_keys = [
        "id",
        "seq_input",
        "loop_input",
        "dist_input",
        "reconstruction_labels",
        "targets",
    ]
    for key in required_keys:
        assert key in batch, f"Batch missing key: {key}"

    # Verify Shapes
    # seq_input: (Batch, Seq_Len)
    assert batch["seq_input"].shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"seq_input shape mismatch: {batch['seq_input'].shape}"

    # targets: (Batch, Scored_Len, Num_Targets) -> (4, 68, 5)
    # Note: process_data stacks targets to (N, 68, 5)
    assert batch["targets"].shape == (
        Config.BATCH_SIZE,
        68,
        5,
    ), f"targets shape mismatch: {batch['targets'].shape}"

    print("    Batch structure and shapes verified.")

    # --------------------------------------------------------------------------
    # 4. Model Instantiation and Forward Pass
    # --------------------------------------------------------------------------
    print("\n>>> Verifying Model...")

    model = MultiTaskRNANet().to(Config.DEVICE)

    # Move batch to device
    seq_in = batch["seq_input"].to(Config.DEVICE)
    loop_in = batch["loop_input"].to(Config.DEVICE)
    dist_in = batch["dist_input"].to(Config.DEVICE)

    # Forward pass
    outputs = model(seq_in, loop_in, dist_in)

    # Verify Outputs
    assert "pred_degradation" in outputs
    assert "pred_reconstruction" in outputs

    # pred_degradation: (Batch, Seq_Len, 5)
    assert outputs["pred_degradation"].shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), f"Degradation output shape incorrect: {outputs['pred_degradation'].shape}"

    # pred_reconstruction: (Batch, Seq_Len, 4)
    assert outputs["pred_reconstruction"].shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        4,
    ), f"Reconstruction output shape incorrect: {outputs['pred_reconstruction'].shape}"

    print("    Model forward pass successful.")

    # --------------------------------------------------------------------------
    # 5. Engine Execution (Training & Inference)
    # --------------------------------------------------------------------------
    print("\n>>> Running Engine (Train & Predict)...")

    engine = Engine()

    # Run Training (1 Epoch as configured)
    engine.run_training()

    # Check if best model was saved
    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model file was not created."
    print(f"    Model saved at {Config.BEST_MODEL_PATH}")

    # Run Submission Generation
    engine.generate_submission()

    # Check if submission file was created
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify Submission Format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"    Submission loaded. Shape: {df_sub.shape}")

    # Expected rows: Num_Test_Samples * Seq_Len
    # In Debug mode, we use 50 samples. 50 * 107 = 5350 rows.
    expected_rows = Config.DEBUG_SUBSET_SIZE * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Expected columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(df_sub.columns)}"

    # Check content of id_seqpos
    first_id = df_sub.iloc[0]["id_seqpos"]
    assert (
        "_0" in first_id
    ), "id_seqpos format seems incorrect (expected suffix _0, _1, etc.)"

    print("    Submission format verified.")
    print("\n>>> Demo completed successfully.")


if __name__ == "__main__":
    run_demo()
