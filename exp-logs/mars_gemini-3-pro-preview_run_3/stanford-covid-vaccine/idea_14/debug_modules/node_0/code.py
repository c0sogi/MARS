import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import provided library modules
from library.config import Config
from library.utils import set_seed, mcrmse_loss
from library.data import get_loaders
from library.model import RNAModel
from library.engine import train_engine, predict_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("==== RNA Degradation Prediction Demo ====")

    # 1. Configuration Setup
    # Override Config parameters for a fast demonstration run
    print("\n[1] Configuring environment...")
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_FILE = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # optimize for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = (
        0  # Use main process for data loading to ensure stability in demo
    )

    set_seed(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # 2. Data Loading Verification
    print("\n[2] Initializing Data Loaders...")
    # load_cached_data=False forces processing from parquet files to demonstrate the full pipeline
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    print("Verifying Train Batch structure...")
    batch = next(iter(train_loader))
    inputs = batch["sequence"]
    pair_indices = batch["pair_index"]
    targets = batch["targets"]
    ids = batch["id"]

    print(
        f"  Input Shape: {inputs.shape} (Expected: {Config.BATCH_SIZE}, {Config.SEQ_LENGTH}, {Config.INPUT_DIM})"
    )
    print(
        f"  Target Shape: {targets.shape} (Expected: {Config.BATCH_SIZE}, {Config.SEQ_SCORED}, {Config.OUTPUT_DIM})"
    )

    # Assertions to ensure data integrity
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.INPUT_DIM,
    ), "Input shape mismatch!"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_SCORED,
        Config.OUTPUT_DIM,
    ), "Target shape mismatch!"
    assert pair_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
    ), "Pair index shape mismatch!"
    print("  Data Loading verification successful.")

    # 3. Model Architecture Verification
    print("\n[3] Verifying Model Architecture...")
    device = torch.device(Config.DEVICE)
    model = RNAModel().to(device)

    # Move batch to device
    inputs_dev = inputs.to(device)
    pair_indices_dev = pair_indices.to(device)

    # Perform dummy forward pass
    with torch.no_grad():
        outputs = model(inputs_dev, pair_indices_dev)

    print(f"  Model Output Shape: {outputs.shape}")

    # Verify output shape: (Batch, Seq_Length, Output_Dim)
    # Note: Model outputs predictions for the full sequence length (107), not just scored length (68)
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.OUTPUT_DIM,
    ), "Model output shape mismatch!"
    print("  Model forward pass successful.")

    # 4. Metric Logic Verification
    print("\n[4] Verifying Metric Logic (MCRMSE)...")
    # Create dummy predictions (all 1.0) and targets (all 0.0)
    # Since RMSE(1, 0) = 1, the MCRMSE should be exactly 1.0
    dummy_preds = torch.ones((Config.BATCH_SIZE, Config.SEQ_LENGTH, Config.OUTPUT_DIM))
    dummy_targets = torch.zeros(
        (Config.BATCH_SIZE, Config.SEQ_SCORED, Config.OUTPUT_DIM)
    )

    loss_val = mcrmse_loss(dummy_preds, dummy_targets)
    print(f"  Calculated Loss: {loss_val.item():.4f} (Expected: 1.0000)")

    assert (
        abs(loss_val.item() - 1.0) < 1e-5
    ), f"Metric calculation failed! Expected 1.0, got {loss_val.item()}"
    print("  Metric verification successful.")

    # 5. Training Loop Execution
    print("\n[5] Executing Training Loop...")
    # Train for limited epochs (defined in Config override)
    best_val_score = train_engine(train_loader, val_loader, epochs=Config.EPOCHS)

    print(f"  Training finished. Best Val MCRMSE: {best_val_score:.4f}")

    # Verify model checkpoint exists
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model checkpoint not found at {Config.MODEL_PATH}")
    print(f"  Model saved to {Config.MODEL_PATH}")

    # 6. Inference and Submission Generation
    print("\n[6] Generating Submission...")
    predict_submission(test_loader, output_file=Config.SUBMISSION_FILE)

    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_FILE}"
        )

    # Verify Submission Format
    sub_df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"  Submission Shape: {sub_df.shape}")

    # Expected Rows: 240 samples * 107 positions = 25680
    expected_rows = 240 * 107
    if len(sub_df) != expected_rows:
        raise AssertionError(
            f"Submission row count mismatch! Expected {expected_rows}, got {len(sub_df)}"
        )

    # Expected Columns: id_seqpos + 5 targets
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    if list(sub_df.columns) != expected_cols:
        raise AssertionError(
            f"Submission columns mismatch! Expected {expected_cols}, got {list(sub_df.columns)}"
        )

    print("  Submission format verification successful.")

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    main()
