import os
import shutil
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import AS_DRN
from library.train import train_epoch, validate, generate_submission, MCRMSELoss


def run_demo():
    print("Initializing Demo Script...")

    # --------------------------------------------------------------------------
    # 1. Configuration Override for Speed and Isolation
    # --------------------------------------------------------------------------
    print("Step 1: Overriding Config for fast execution...")

    # Use a separate cache directory for the demo to avoid conflicts
    Config.CACHE_DIR = "./working/demo_cache"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "demo_submission.csv")

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Reduce computational load
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.DEBUG_SAMPLES = 20  # Use a tiny subset
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # --------------------------------------------------------------------------
    # 2. Data Loading & Verification
    # --------------------------------------------------------------------------
    print("Step 2: Loading Data and Verifying Shapes...")

    # Get dataloaders in debug mode
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Fetch one batch from training loader
    batch = next(iter(train_loader))
    inputs = batch["inputs"]
    partner_indices = batch["partner_indices"]
    targets = batch["targets"]

    # Verify Input Shapes
    # Expected: (Batch, Seq_Len=107, Channels=18)
    assert inputs.dim() == 3, f"Expected 3D inputs, got {inputs.shape}"
    assert (
        inputs.shape[1] == Config.SEQ_LENGTH
    ), f"Expected seq len {Config.SEQ_LENGTH}, got {inputs.shape[1]}"
    assert inputs.shape[2] == 18, f"Expected 18 input channels, got {inputs.shape[2]}"

    # Verify Partner Indices
    # Expected: (Batch, Seq_Len=107)
    assert (
        partner_indices.dim() == 2
    ), f"Expected 2D partner indices, got {partner_indices.shape}"
    assert partner_indices.shape[1] == Config.SEQ_LENGTH

    # Verify Targets
    # Expected: (Batch, Seq_Len=107, Targets=5)
    assert targets.dim() == 3, f"Expected 3D targets, got {targets.shape}"
    assert targets.shape[2] == Config.NUM_TARGETS

    print("  Data shapes verified successfully.")

    # --------------------------------------------------------------------------
    # 3. Model Initialization & Logic Verification
    # --------------------------------------------------------------------------
    print("Step 3: Initializing Model and Verifying Forward Pass...")

    device = Config.DEVICE
    model = AS_DRN().to(device)

    # Move batch to device
    inputs = inputs.to(device)
    partner_indices = partner_indices.to(device)

    # Forward pass
    # Model returns (y_2, y_1)
    y_2, y_1 = model(inputs, partner_indices)

    # Verify Output Shapes
    # Expected: (Batch, Seq_Len=107, Targets=5)
    assert y_2.shape == (inputs.shape[0], Config.SEQ_LENGTH, Config.NUM_TARGETS)
    assert y_1.shape == (inputs.shape[0], Config.SEQ_LENGTH, Config.NUM_TARGETS)

    # Verify Outputs are not NaN (basic stability check)
    assert not torch.isnan(y_2).any(), "Model output y_2 contains NaNs"
    assert not torch.isnan(y_1).any(), "Model output y_1 contains NaNs"

    print("  Model forward pass verified successfully.")

    # --------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print(f"Step 4: Running Training Loop for {Config.EPOCHS} epochs...")

    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
    criterion = MCRMSELoss()

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate(model, val_loader, device)

        print(
            f"  Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val MCRMSE={val_score:.4f}"
        )

        # Assertions
        assert not np.isnan(train_loss), "Training loss is NaN"
        assert not np.isnan(val_score), "Validation score is NaN"
        assert train_loss > 0, "Training loss should be positive"

    print("  Training loop completed successfully.")

    # --------------------------------------------------------------------------
    # 5. Submission Generation & Verification
    # --------------------------------------------------------------------------
    print("Step 5: Generating Submission...")

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

    # Verify File Exists
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    # Verify Content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check Row Count: num_test_samples * seq_len
    # Note: In debug mode, test set is also limited to DEBUG_SAMPLES
    expected_rows = Config.DEBUG_SAMPLES * Config.SEQ_LENGTH
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(sub_df)}"

    # Check Columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns do not match. Expected {expected_cols}, got {list(sub_df.columns)}"

    # Check Value Integrity (No NaNs)
    assert not sub_df.isnull().values.any(), "Submission contains null values"

    print(f"  Submission verified: {len(sub_df)} rows.")
    print("Demo completed successfully.")


if __name__ == "__main__":
    run_demo()
