import os
import sys
import torch
import pandas as pd
import numpy as np

# Import library components
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import HybridNet
from library.loss import MaskedMCRMSELoss
from library.train import run_training
from library.inference import run_inference


def main():
    print("=== Starting Demonstration Script ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Demo Isolation
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment...")

    # Enable Debug mode to use a subset of data
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples for speed

    # Reduce training parameters
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8

    # Redirect outputs to a demo directory
    Config.WORKING_DIR = "./working/demo_execution"
    Config.MODEL_CHECKPOINT = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Update cache paths to lie within the demo directory
    cache_dir = os.path.join(Config.WORKING_DIR, "data_cache")
    Config.TRAIN_CACHE = os.path.join(cache_dir, "train_cache.npz")
    Config.VAL_CACHE = os.path.join(cache_dir, "val_cache.npz")
    Config.TEST_CACHE = os.path.join(cache_dir, "test_cache.npz")

    # Create necessary directories
    os.makedirs(cache_dir, exist_ok=True)

    # Set random seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration updated for rapid execution.")

    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Data Pipeline...")

    # Load data (force reprocessing to ensure debug subsetting applies)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch a single batch from training loader
    try:
        batch = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    inputs = batch["inputs"]
    partner_indices = batch["partner_indices"]
    targets = batch["targets"]
    ids = batch["id"]

    print(f"  Batch loaded. Keys: {list(batch.keys())}")
    print(f"  Inputs shape: {inputs.shape}")
    print(f"  Partner Indices shape: {partner_indices.shape}")
    print(f"  Targets shape: {targets.shape}")

    # Validate shapes
    expected_input_shape = (inputs.size(0), Config.INPUT_CHANNELS, Config.SEQ_LENGTH)
    assert (
        inputs.shape == expected_input_shape
    ), f"Input shape mismatch. Expected {expected_input_shape}, got {inputs.shape}"

    assert partner_indices.shape == (
        inputs.size(0),
        Config.SEQ_LENGTH,
    ), "Partner indices shape mismatch."

    assert targets.shape == (
        inputs.size(0),
        Config.SEQ_SCORED,
        Config.NUM_TARGETS,
    ), "Targets shape mismatch."

    print("Data pipeline verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model and Loss Verification
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Model and Loss...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")

    # Instantiate Model
    model = HybridNet().to(device)

    # Move data to device
    inputs_dev = inputs.to(device)
    partner_indices_dev = partner_indices.to(device)
    targets_dev = targets.to(device)

    # Forward Pass
    preds = model(inputs_dev, partner_indices_dev)
    print(f"  Predictions shape: {preds.shape}")

    # Validate Prediction Shape (Batch, SeqLen, NumTargets)
    # Note: Model predicts for full sequence length (107), targets are only for first 68.
    assert preds.shape == (
        inputs.size(0),
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), "Prediction shape mismatch."

    # Instantiate Loss
    criterion = MaskedMCRMSELoss().to(device)

    # Calculate Loss
    loss = criterion(preds, targets_dev)
    print(f"  Calculated Loss: {loss.item():.6f}")

    # Validate Loss
    assert not torch.isnan(loss), "Loss is NaN."
    assert loss.item() >= 0, "Loss is negative."

    print("Model and Loss verification passed.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[Step 4] Executing Training Loop...")

    # Run training (uses the cache generated in Step 2)
    trained_model = run_training(epochs=Config.EPOCHS, load_cached_data=True)

    # Verify checkpoint creation
    assert os.path.exists(
        Config.MODEL_CHECKPOINT
    ), f"Model checkpoint not found at {Config.MODEL_CHECKPOINT}"

    print("Training execution completed.")

    # -------------------------------------------------------------------------
    # 5. Inference Execution
    # -------------------------------------------------------------------------
    print("\n[Step 5] Executing Inference...")

    run_inference(
        model_path=Config.MODEL_CHECKPOINT,
        output_path=Config.SUBMISSION_PATH,
        load_cached_data=True,
    )

    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    print("Inference execution completed.")

    # -------------------------------------------------------------------------
    # 6. Submission Validation
    # -------------------------------------------------------------------------
    print("\n[Step 6] Validating Submission File...")

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission shape: {sub_df.shape}")

    # Check Columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    # Check Row Count
    # We used Config.DEBUG_SUBSET_SIZE (50) samples for the test set.
    # Each sample requires predictions for the full sequence length (107).
    expected_rows = Config.DEBUG_SUBSET_SIZE * Config.SEQ_LENGTH
    assert (
        len(sub_df) == expected_rows
    ), f"Row count mismatch. Expected {expected_rows} (50 samples * 107 pos), got {len(sub_df)}"

    # Check for NaNs
    assert not sub_df.isnull().values.any(), "Submission contains NaNs."

    print("Submission file is valid.")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
