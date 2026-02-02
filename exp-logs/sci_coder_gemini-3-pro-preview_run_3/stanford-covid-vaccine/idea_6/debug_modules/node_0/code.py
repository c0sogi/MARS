import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import warnings

# Import library components
from library.config import Config
from library.utils import set_seed, MCRMSELoss
from library.data import get_dataloaders
from library.model import AttentionAugmentedResBiGRU
from library.train import train_one_epoch, validate, generate_submission

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration of RNA Degradation Library ===\n")

    # --------------------------------------------------------------------------
    # 1. Configuration Override for Demo
    # --------------------------------------------------------------------------
    print("1. Configuring environment for fast demonstration...")

    # Override Config parameters to run a small, fast experiment
    Config.DEBUG = True
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set up demo-specific working directories
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)

    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(DEMO_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Create the new directories
    Config.create_directories()

    # Set reproducible seed
    set_seed(Config.SEED)
    print("   Configuration updated. Debug mode: ON.")

    # --------------------------------------------------------------------------
    # 2. Data Loading Verification
    # --------------------------------------------------------------------------
    print("\n2. Verifying Data Loading...")

    # Force reload to ensure debug slicing applies
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    # Fetch one batch from training loader
    train_inputs, train_targets = next(iter(train_loader))

    print(
        f"   Train Batch - Inputs: {train_inputs.shape}, Targets: {train_targets.shape}"
    )

    # Assertions for shapes
    # Input: (Batch, Seq_Len=107, Input_Dim=14)
    assert train_inputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_DIM,
    ), f"Expected input shape {(Config.BATCH_SIZE, Config.SEQ_LEN, Config.INPUT_DIM)}, got {train_inputs.shape}"

    # Target: (Batch, Pred_Len=68, Output_Dim=5)
    assert train_targets.shape == (
        Config.BATCH_SIZE,
        Config.PRED_LEN,
        Config.OUTPUT_DIM,
    ), f"Expected target shape {(Config.BATCH_SIZE, Config.PRED_LEN, Config.OUTPUT_DIM)}, got {train_targets.shape}"

    # Fetch one batch from test loader (returns inputs and ids)
    test_inputs, test_ids = next(iter(test_loader))
    print(f"   Test Batch - Inputs: {test_inputs.shape}, IDs: {len(test_ids)}")
    assert test_inputs.shape[1] == Config.SEQ_LEN

    print("   Data Loading verification passed.")

    # --------------------------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # --------------------------------------------------------------------------
    print("\n3. Verifying Model Architecture...")

    device = Config.DEVICE
    model = AttentionAugmentedResBiGRU().to(device)

    # Move inputs to device
    dummy_input = train_inputs.to(device)

    # Forward pass
    output = model(dummy_input)
    print(f"   Model Output Shape: {output.shape}")

    # Assert output shape: (Batch, Seq_Len=107, Output_Dim=5)
    # Note: Model outputs predictions for the full sequence length (107),
    # even though targets are only provided for the first 68.
    assert output.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.OUTPUT_DIM,
    ), f"Expected output shape {(Config.BATCH_SIZE, Config.SEQ_LEN, Config.OUTPUT_DIM)}, got {output.shape}"

    print("   Model architecture verification passed.")

    # --------------------------------------------------------------------------
    # 4. Loss Function Verification
    # --------------------------------------------------------------------------
    print("\n4. Verifying MCRMSE Loss...")

    # Move targets to device
    dummy_targets = train_targets.to(device)

    # Initialize Loss
    criterion = MCRMSELoss()

    # Calculate Loss
    loss = criterion(output, dummy_targets)
    print(f"   Calculated Loss: {loss.item():.6f}")

    # Assertions
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    # Verify Backward Pass (Gradient Check)
    loss.backward()
    # Check if gradients exist for a sample parameter
    param = next(model.parameters())
    assert param.grad is not None, "Gradients not computed during backward pass"

    # Verify Scored Columns Logic (Validation Metric)
    # Scored indices: [0, 1, 3]
    criterion_val = MCRMSELoss(select_columns=Config.SCORED_TARGET_INDICES)
    loss_val = criterion_val(output, dummy_targets)
    print(f"   Calculated Validation Loss (Scored Cols Only): {loss_val.item():.6f}")

    print("   Loss function verification passed.")

    # --------------------------------------------------------------------------
    # 5. Training Loop Simulation
    # --------------------------------------------------------------------------
    print("\n5. Simulating Training Loop...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Capture weights before update to verify learning
    initial_weights = model.head.weight.data.clone()

    print("   Running Epoch 1...")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

    print("   Running Validation...")
    val_loss = validate(model, val_loader, criterion_val, device)

    print(f"   Epoch 1 Result - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    # Verify weights updated
    final_weights = model.head.weight.data
    assert not torch.equal(
        initial_weights, final_weights
    ), "Model weights did not update after training step"

    # Save the model (required for submission generation step)
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model file was not saved"

    print("   Training simulation passed.")

    # --------------------------------------------------------------------------
    # 6. Submission Generation Verification
    # --------------------------------------------------------------------------
    print("\n6. Verifying Submission Generation...")

    # Generate submission using the saved model
    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

    # Verify output file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"   Submission File Shape: {df_sub.shape}")
    print(f"   Submission Columns: {list(df_sub.columns)}")

    # Logic Check:
    # We used debug mode with 100 samples.
    # Prediction length per sample is 68.
    # Total rows should be 100 * 68 = 6800.
    # However, the debug slicer in `data.py` slices inputs to 100.
    expected_rows = 100 * Config.PRED_LEN
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows in submission, found {len(df_sub)}"

    # Check column names
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert (
        list(df_sub.columns) == expected_cols
    ), "Submission columns do not match requirements"

    # Check content format of id_seqpos
    sample_id_seqpos = df_sub.iloc[0]["id_seqpos"]
    assert "_" in sample_id_seqpos, f"Invalid id_seqpos format: {sample_id_seqpos}"

    print("   Submission generation verification passed.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
