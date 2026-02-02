import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, get_device, calculate_mcrmse
from library.data import get_loaders, RNADataset
from library.model import RNAModel
from library.loss import MCRMSELoss
from library.train import train_one_epoch, evaluate, generate_submission


def main():
    # =========================================================================
    # 1. Setup and Configuration Override for Demo
    # =========================================================================
    print("1. Setting up configuration for rapid demonstration...")

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Override Config for speed and isolation
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples
    Config.BATCH_SIZE = 8
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Use a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = Config.WORKING_DIR
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission_demo.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set reproducibility
    set_seed(Config.SEED)
    device = get_device()
    print(f"   Device: {device}")

    # =========================================================================
    # 2. Data Loading and Verification
    # =========================================================================
    print("\n2. Loading data and verifying DataLoaders...")

    # Force reload to ensure we use the debug subset
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    # Fetch one batch to verify structure
    batch = next(iter(train_loader))
    inputs = batch["inputs"].to(device)
    pair_indices = batch["pair_indices"].to(device)
    pair_masks = batch["pair_masks"].to(device)
    targets = batch["targets"].to(device)
    ids = batch["ids"]

    # Assertions for Data Shapes
    # Inputs: (Batch, Seq_Len=107, Input_Dim=14)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_DIM,
    ), f"Input shape mismatch: {inputs.shape}"

    # Targets: (Batch, Pred_Len=68, Num_Targets=5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.PRED_LEN,
        Config.NUM_TARGETS,
    ), f"Target shape mismatch: {targets.shape}"

    # Structural features
    assert pair_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), "Pair indices shape mismatch"
    assert pair_masks.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), "Pair masks shape mismatch"

    print("   Data shapes verified successfully.")

    # =========================================================================
    # 3. Model Initialization and Forward Pass
    # =========================================================================
    print("\n3. Initializing Model and running forward pass...")

    model = RNAModel(config=Config).to(device)

    # Run forward pass
    preds = model(inputs, pair_indices, pair_masks)

    # Verify Output Shape
    # Model outputs predictions for the full sequence length (107)
    expected_out_shape = (Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_TARGETS)
    assert (
        preds.shape == expected_out_shape
    ), f"Model output shape mismatch. Expected {expected_out_shape}, got {preds.shape}"

    print("   Model forward pass successful. Output shape verified.")

    # =========================================================================
    # 4. Loss and Metric Verification
    # =========================================================================
    print("\n4. Verifying Loss and Metric calculations...")

    criterion = MCRMSELoss()

    # Calculate Loss
    loss = criterion(preds, targets)

    # Check loss properties
    assert loss.dim() == 0, "Loss should be a scalar"
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss is NaN"

    print(f"   MCRMSE Loss calculated: {loss.item():.4f}")

    # Calculate Metric (using utility function)
    # This function handles moving to CPU and specific column selection
    metric_score = calculate_mcrmse(preds, targets)
    assert isinstance(metric_score, float), "Metric should return a float"
    assert metric_score >= 0, "Metric should be non-negative"

    print(f"   Metric (MCRMSE) calculated: {metric_score:.4f}")

    # =========================================================================
    # 5. Training Loop Demonstration
    # =========================================================================
    print("\n5. Running simplified training loop...")

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Train one epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"   Train Loss after 1 epoch: {train_loss:.4f}")

    # Evaluate
    val_score = evaluate(model, val_loader, device)
    print(f"   Validation Score: {val_score:.4f}")

    # Save model (simulating checkpointing)
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved."
    print("   Model checkpoint saved.")

    # =========================================================================
    # 6. Submission Generation
    # =========================================================================
    print("\n6. Generating Submission...")

    # Load the model back to ensure save/load works
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Generate submission
    generate_submission(model, test_loader, device)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(df_sub.columns)}"

    # Check row count
    # In DEBUG mode, we used 50 samples.
    # Total rows = Num_Samples * Seq_Len (107)
    # Note: The test loader might have fewer than 50 samples if the source test.json is small or filtered.
    # The provided test.json has 240 lines. DEBUG takes head(50).
    expected_rows = Config.DEBUG_SUBSET_SIZE * Config.SEQ_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Check content validity (no NaNs)
    assert not df_sub.isnull().values.any(), "Submission contains NaNs"

    print(f"   Submission generated successfully with {len(df_sub)} rows.")
    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
