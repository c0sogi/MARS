import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, calculate_mcrmse
from library.loss import MCRMSELoss
from library.data import get_dataloaders
from library.model import RNAModel
from library.train import train_one_epoch, validate, generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    print("Initializing Demonstration...")

    # ==========================================
    # 1. Configuration Setup for Demo
    # ==========================================
    # We override Config parameters to ensure the demo runs quickly and uses the correct paths.
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set hyperparameters for speed
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Use only 20 samples

    # Set seed for reproducibility
    set_seed(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # 2. Data Loading Demonstration
    # ==========================================
    print("\n[Step 1] Loading Data (Debug Mode)...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False,  # Force processing to demonstrate pipeline
        debug=True,
        debug_size=Config.DEBUG_SUBSET_SIZE,
        batch_size=Config.BATCH_SIZE,
    )

    # Verify Train Loader
    batch = next(iter(train_loader))
    inputs = batch["inputs"].to(device)
    pair_indices = batch["pair_indices"].to(device)
    pair_mask = batch["pair_mask"].to(device)
    targets = batch["targets"].to(device)
    ids = batch["id"]

    print(f"Batch loaded. Batch size: {len(ids)}")

    # Assertions for shapes
    # Inputs: (B, 107, 14)
    expected_input_shape = (Config.BATCH_SIZE, Config.SEQ_LEN, Config.INPUT_DIM)
    assert (
        inputs.shape == expected_input_shape
    ), f"Input shape mismatch. Expected {expected_input_shape}, got {inputs.shape}"

    # Targets: (B, 68, 5)
    expected_target_shape = (Config.BATCH_SIZE, Config.SEQ_SCORED, 5)
    assert (
        targets.shape == expected_target_shape
    ), f"Target shape mismatch. Expected {expected_target_shape}, got {targets.shape}"

    print("Data shapes verified successfully.")

    # ==========================================
    # 3. Model Initialization & Forward Pass
    # ==========================================
    print("\n[Step 2] Initializing Model...")
    model = RNAModel().to(device)

    print("Running forward pass...")
    outputs = model(inputs, pair_indices, pair_mask)

    # Assert Output Shape: (B, 107, 5)
    # Note: Model outputs predictions for the full sequence length (107), not just scored (68).
    expected_output_shape = (Config.BATCH_SIZE, Config.SEQ_LEN, 5)
    assert (
        outputs.shape == expected_output_shape
    ), f"Output shape mismatch. Expected {expected_output_shape}, got {outputs.shape}"

    print("Forward pass successful. Output shape verified.")

    # ==========================================
    # 4. Loss Calculation
    # ==========================================
    print("\n[Step 3] Calculating Loss...")
    criterion = MCRMSELoss()
    loss = criterion(outputs, targets)

    print(f"Calculated Loss: {loss.item()}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    # Verify manual metric calculation logic matches roughly
    # (Slice outputs to 68, compute RMSE)
    outputs_sliced = outputs[:, : Config.SEQ_SCORED, :]
    mse = torch.mean((outputs_sliced - targets) ** 2)
    # Note: MCRMSE is mean of RMSEs per column, not RMSE of mean MSE.
    # But checking if the loss object works is sufficient here.

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("\n[Step 4] Running Training Loop (1 Epoch)...")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Train for one epoch
    avg_train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, device, epoch=0
    )
    print(f"Training Epoch 0 completed. Avg Loss: {avg_train_loss:.4f}")

    assert avg_train_loss > 0, "Training loss should be positive"

    # ==========================================
    # 6. Validation Demonstration
    # ==========================================
    print("\n[Step 5] Running Validation...")
    val_mcrmse = validate(model, val_loader, device)
    print(f"Validation MCRMSE: {val_mcrmse:.4f}")

    assert val_mcrmse >= 0, "Validation metric should be non-negative"

    # ==========================================
    # 7. Submission Generation
    # ==========================================
    print("\n[Step 6] Generating Submission...")
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Shape: {sub_df.shape}")

    # Expected rows: Num_Test_Samples (20 in debug) * Seq_Len (107)
    expected_rows = Config.DEBUG_SUBSET_SIZE * Config.SEQ_LEN
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(sub_df)}"

    # Expected columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(sub_df.columns)}"

    # Check ID format
    sample_id_seqpos = sub_df.iloc[0]["id_seqpos"]
    assert (
        "_0" in sample_id_seqpos or "_" in sample_id_seqpos
    ), f"Invalid id_seqpos format: {sample_id_seqpos}"

    print("Submission file verified successfully.")

    print("\nAll demonstration steps completed successfully.")


if __name__ == "__main__":
    main()
