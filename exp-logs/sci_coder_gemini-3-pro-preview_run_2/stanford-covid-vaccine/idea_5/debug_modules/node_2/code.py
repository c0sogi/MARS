import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.data import get_loaders, process_inputs, get_couples
from library.model import HybridNet
from library.loss import MCRMSELoss
from library.train import run_training
from library.inference import run_inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def main():
    print("==== STARTING DEMO SCRIPT ====")

    # 1. SETUP & CONFIGURATION OVERRIDE
    # We modify the Config class directly to make the demo run fast.
    print("\n[Step 1] Configuring environment for rapid demonstration...")

    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 16  # Small batch size
    Config.HIDDEN_DIM = 32  # Reduced model size
    Config.NUM_LAYERS = 2  # Fewer layers in TCN
    Config.NUM_WORKERS = 0  # Main process only to avoid overhead
    Config.WORKING_DIR = "./working/demo_execution"  # Separate dir for demo

    # Update derived paths based on new WORKING_DIR
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Clean up demo directory if it exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. DATA PIPELINE VERIFICATION
    print("\n[Step 2] Verifying Data Pipeline...")

    # Load data (this will trigger processing and caching)
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    # Fetch one batch
    batch = next(iter(train_loader))
    inputs = batch["inputs"].to(device)
    targets = batch["targets"].to(device)
    ids = batch["ids"]

    print(f"Batch loaded. Batch size: {inputs.size(0)}")
    print(f"Inputs shape: {inputs.shape} (Expected: B, Channels, Seq_Len)")
    print(f"Targets shape: {targets.shape} (Expected: B, Seq_Len, 5)")

    # Assertions
    assert inputs.dim() == 3, "Inputs should be 3-dimensional"
    assert targets.dim() == 3, "Targets should be 3-dimensional"
    assert (
        inputs.size(2) == Config.SEQ_LEN
    ), f"Sequence length should be {Config.SEQ_LEN}"
    assert (
        targets.size(2) == 5
    ), "Targets should have 5 channels (reactivity + 4 deg conditions)"

    # Verify utility functions
    sample_struct = "((...))"
    pairs = get_couples(sample_struct)
    assert pairs[0] == 6 and pairs[1] == 5, "Structure parsing logic failed"
    print("Data loading and utility functions verified.")

    # 3. MODEL ARCHITECTURE & FORWARD PASS
    print("\n[Step 3] Verifying Model Architecture...")

    model = HybridNet().to(device)

    # Perform forward pass
    outputs = model(inputs)

    print(f"Model output shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        inputs.size(0),
        Config.SEQ_LEN,
        5,
    ), f"Output shape mismatch. Expected {(inputs.size(0), Config.SEQ_LEN, 5)}, got {outputs.shape}"

    print("Model forward pass verified.")

    # 4. LOSS FUNCTION VERIFICATION
    print("\n[Step 4] Verifying Loss Function (MCRMSE)...")

    criterion = MCRMSELoss().to(device)
    loss = criterion(outputs, targets)

    print(f"Calculated Loss: {loss.item()}")

    # Assertions
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    # Manual check: MCRMSE only scores specific columns and first 68 positions
    # Create dummy data to verify logic
    # Config.SCORED_TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"] -> indices 0, 1, 3 in ALL_TARGET_COLS
    dummy_pred = torch.zeros((1, 107, 5)).to(device)
    dummy_target = torch.zeros((1, 107, 5)).to(device)

    # Set a known error in a scored position (index 0, col 0)
    dummy_pred[0, 0, 0] = 1.0  # Error = 1.0, Squared = 1.0
    # Set a known error in an unscored position (index 70, col 0) -> Should be ignored
    dummy_pred[0, 70, 0] = 100.0
    # Set a known error in an unscored column (index 0, col 2 'deg_pH10') -> Should be ignored
    dummy_pred[0, 0, 2] = 100.0

    # Calculation:
    # Scored columns: 0, 1, 3.
    # Col 0 MSE: (1.0^2 + 0...)/68 = 1/68
    # Col 1 MSE: 0
    # Col 3 MSE: 0
    # RMSEs: sqrt(1/68), 0, 0
    # MCRMSE: (sqrt(1/68) + 0 + 0) / 3

    expected_val = (np.sqrt(1 / 68)) / 3
    calc_val = criterion(dummy_pred, dummy_target).item()

    assert np.isclose(
        calc_val, expected_val, atol=1e-5
    ), f"MCRMSE logic check failed. Expected {expected_val}, got {calc_val}"

    print("Loss function logic verified.")

    # 5. TRAINING LOOP DEMO
    print("\n[Step 5] Running Training Loop (1 Epoch)...")

    # This function handles the loop, checkpointing, etc.
    # We use load_cached_data=True because we generated the cache in Step 2.
    run_training(load_cached_data=True)

    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not created."
    print("Training loop completed successfully.")

    # 6. INFERENCE & SUBMISSION
    print("\n[Step 6] Running Inference and Generating Submission...")

    run_inference(load_cached_data=True)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify submission content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission shape: {sub_df.shape}")

    # Expected rows: 240 test samples * 107 positions = 25680
    # Expected cols: id_seqpos + 5 targets = 6
    assert sub_df.shape[1] == 6, "Submission should have 6 columns"
    # Note: Depending on test.json size provided in environment, row count might vary.
    # The prompt metadata says test.json has 240 lines.
    expected_rows = 240 * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}"

    print("Inference and submission generation verified.")

    print("\n==== DEMO SCRIPT COMPLETE ====")


if __name__ == "__main__":
    main()
