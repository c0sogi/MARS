import os
import shutil
import torch
import pandas as pd
import numpy as np

# Import components from the provided library files
from library.config import Config
from library.data import load_data, get_loaders
from library.model import SDBR_BiGRU
from library.utils import seed_everything, MCRMSELoss, calculate_metric
from library.train import run_training, predict_and_submit


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print(">>> Initializing Demo Configuration...")
    seed_everything(42)

    # Override Config parameters for a fast, lightweight demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_PATH = "./working/demo_execution/submission_demo.csv"
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.HIDDEN_DIM = 64  # Reduced from 384 for speed
    Config.NUM_LAYERS = 1  # Reduced from 3 for speed
    Config.STEM_FILTERS = 32  # Reduced from 256 for speed

    # Clean up any existing demo directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # ==========================================
    # 2. Data Loading & Verification
    # ==========================================
    print("\n>>> Testing Data Loading (Debug Mode)...")

    # Load subsampled dataframes (debug=True loads Batch_Size * 2 rows)
    train_df, val_df, test_df = load_data(debug=True, load_cached_data=False)

    print(f"Train DataFrame Shape: {train_df.shape}")
    print(f"Val DataFrame Shape:   {val_df.shape}")
    print(f"Test DataFrame Shape:  {test_df.shape}")

    # Assertions to verify data loading
    assert len(train_df) == Config.BATCH_SIZE * 2, "Train subset size mismatch"
    assert len(val_df) == Config.BATCH_SIZE * 2, "Val subset size mismatch"
    assert len(test_df) == Config.BATCH_SIZE * 2, "Test subset size mismatch"

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_loaders(
        debug=True, load_cached_data=False
    )

    # Fetch one batch to verify tensor shapes
    batch = next(iter(train_loader))
    features = batch["features"]
    pair_index = batch["pair_index"]
    targets = batch["targets"]

    print(f"Batch Features:   {features.shape}")  # Expected: (B, 107, 14)
    print(f"Batch Pair Index: {pair_index.shape}")  # Expected: (B, 107)
    print(f"Batch Targets:    {targets.shape}")  # Expected: (B, 68, 5)

    assert features.shape == (Config.BATCH_SIZE, Config.SEQ_LEN, Config.INPUT_CHANNELS)
    assert pair_index.shape == (Config.BATCH_SIZE, Config.SEQ_LEN)
    assert targets.shape == (Config.BATCH_SIZE, Config.SEQ_SCORED, Config.OUTPUT_DIM)

    # ==========================================
    # 3. Model Instantiation & Forward Pass
    # ==========================================
    print("\n>>> Testing Model Architecture...")

    device = torch.device("cpu")  # Use CPU for assertion checks
    model = SDBR_BiGRU().to(device)

    # Move batch to device
    features = features.to(device)
    pair_index = pair_index.to(device)

    # Forward pass
    output = model(features, pair_index)
    print(f"Model Output:     {output.shape}")  # Expected: (B, 107, 5)

    # Verify output shape and integrity
    assert output.shape == (Config.BATCH_SIZE, Config.SEQ_LEN, Config.OUTPUT_DIM)
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    # ==========================================
    # 4. Loss & Metric Calculation
    # ==========================================
    print("\n>>> Testing Loss and Metrics...")

    criterion = MCRMSELoss()

    # Slice output to match target length (68) for loss calculation
    output_scored = output[:, : Config.SEQ_SCORED, :]
    targets = targets.to(device)

    # Calculate Loss
    loss = criterion(output_scored, targets)
    print(f"Calculated Loss:  {loss.item():.6f}")
    assert loss.item() >= 0, "Loss should be non-negative"

    # Calculate Metric (handles slicing internally)
    metric = calculate_metric(output, targets)
    print(f"Calculated Metric: {metric:.6f}")
    assert isinstance(metric, float)

    # ==========================================
    # 5. Training Loop Execution
    # ==========================================
    print("\n>>> Executing Training Pipeline...")

    # Run training for 2 epochs
    best_model_path = run_training(
        epochs=Config.EPOCHS, debug=True, load_cached_data=False, patience=2
    )

    assert os.path.exists(best_model_path), "Best model file was not saved."
    print(f"Training complete. Model saved to: {best_model_path}")

    # ==========================================
    # 6. Inference & Submission Generation
    # ==========================================
    print("\n>>> Generating Submission...")

    predict_and_submit(model_path=best_model_path, debug=True, load_cached_data=False)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Verify Submission Content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission DataFrame Shape: {sub_df.shape}")

    # Expected rows: N_test_samples * Seq_Len
    expected_rows = len(test_df) * Config.SEQ_LEN
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}"

    # Verify Columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert (
        list(sub_df.columns) == expected_cols
    ), "Submission columns do not match requirements"

    print("\n>>> Demo Execution Completed Successfully.")


if __name__ == "__main__":
    main()
