import os
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, format_submission
from library.data import get_loaders, process_structure_info
from library.model import RHS_GFN
from library.loss import MCRMSELoss
from library.engine import train_fn, eval_fn, predict_test

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_pipeline_demonstration():
    """
    Demonstrates the end-to-end pipeline using the provided library.
    """
    print("=== RNA Degradation Prediction Pipeline Demonstration ===\n")

    # 1. Setup & Reproducibility
    # ----------------------------------------------------------------
    print("[1] Setting up environment and seeds...")
    seed_everything(Config.SEED)

    # Ensure the working directory exists (as defined in Config)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    print(f"    Working directory: {Config.WORKING_DIR}")
    print(f"    Device: {Config.DEVICE}")

    # 2. Logic Verification: Feature Engineering
    # ----------------------------------------------------------------
    print("\n[2] Verifying Feature Engineering Logic (Structure Processing)...")

    # Test case: Nested parentheses
    # Sequence indices: 0123456789
    # Structure:        ..((..))..
    # Expected pairs:   (2, 7) and (3, 6)
    test_seq = "AAAAAAAAAA"
    test_struct = "..((..)).."

    p_indices, p_identity = process_structure_info(test_seq, test_struct)

    # Assertions to verify correctness
    assert len(p_indices) == 10, "Partner indices length mismatch"
    assert p_indices[0] == -1, "Unpaired base should be -1"
    assert p_indices[2] == 7, "Index 2 should pair with 7"
    assert p_indices[7] == 2, "Index 7 should pair with 2"
    assert p_indices[3] == 6, "Index 3 should pair with 6"
    assert p_indices[6] == 3, "Index 6 should pair with 3"

    # Verify p_identity shape (Length, 4 one-hot bases)
    assert p_identity.shape == (10, 4), "Partner identity shape mismatch"

    print("    -> Structure processing logic verified successfully.")

    # 3. Data Loading
    # ----------------------------------------------------------------
    print("\n[3] Loading Data (Debug Mode)...")
    # Using debug=True to load a tiny subset (32 samples) for speed
    train_loader, val_loader, test_loader = get_loaders(debug=True)

    # Inspect one batch
    inputs, partner_indices, targets = next(iter(train_loader))

    print(f"    Batch Inputs Shape: {inputs.shape} (Expected: [B, 107, 18])")
    print(
        f"    Batch Partner Indices Shape: {partner_indices.shape} (Expected: [B, 107])"
    )
    print(f"    Batch Targets Shape: {targets.shape} (Expected: [B, 107, 5])")

    # Verify dimensions
    assert inputs.shape[1] == Config.SEQ_LEN
    assert inputs.shape[2] == 18  # 4(seq)+3(struct)+7(loop)+4(partner)
    assert targets.shape[2] == Config.NUM_TARGETS
    print("    -> Data shapes verified.")

    # 4. Model Initialization & Forward Pass
    # ----------------------------------------------------------------
    print("\n[4] Initializing RHS_GFN Model...")
    model = RHS_GFN().to(Config.DEVICE)

    # Move batch to device
    inputs = inputs.to(Config.DEVICE)
    partner_indices = partner_indices.to(Config.DEVICE)
    targets = targets.to(Config.DEVICE)

    print("    Running forward pass...")
    # Model returns tuple (y_2, y_1) due to iterative refinement
    y_2, y_1 = model(inputs, partner_indices)

    print(f"    Output y_2 shape: {y_2.shape}")

    assert y_2.shape == targets.shape, "Output shape mismatch"
    assert not torch.isnan(y_2).any(), "Model produced NaN values"
    print("    -> Forward pass successful.")

    # 5. Loss Calculation
    # ----------------------------------------------------------------
    print("\n[5] Verifying Loss Function (MCRMSE)...")
    criterion = MCRMSELoss()

    # Calculate loss
    loss = criterion(y_2, targets)
    print(f"    Calculated Loss: {loss.item():.6f}")

    assert loss.item() >= 0, "Loss cannot be negative"
    print("    -> Loss function verified.")

    # 6. Training Loop (Mini-Run)
    # ----------------------------------------------------------------
    print("\n[6] Executing Training Loop (1 Epoch)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Train for 1 epoch
    train_loss = train_fn(model, train_loader, optimizer, criterion, Config.DEVICE)
    print(f"    Epoch 1 Train Loss: {train_loss:.6f}")

    # Evaluate
    val_score = eval_fn(model, val_loader, Config.DEVICE)
    print(f"    Validation MCRMSE: {val_score:.6f}")

    # Save model (simulating checkpointing)
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    print(f"    Model saved to {Config.BEST_MODEL_PATH}")

    # 7. Inference & Submission
    # ----------------------------------------------------------------
    print("\n[7] Generating Submission...")

    # Reload best model to ensure saving/loading works
    model.load_state_dict(
        torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
    )

    # Predict on test set
    test_ids, test_preds = predict_test(model, test_loader, Config.DEVICE)

    print(f"    Test Predictions Shape: {test_preds.shape}")

    # Format and save submission
    format_submission(test_ids, test_preds, Config.SUBMISSION_PATH)

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"    Submission file loaded. Shape: {df_sub.shape}")

        # Check integrity
        expected_rows = len(test_ids) * Config.SEQ_LEN
        assert (
            len(df_sub) == expected_rows
        ), f"Expected {expected_rows} rows, got {len(df_sub)}"
        assert df_sub.columns[0] == "id_seqpos", "First column should be id_seqpos"
        print("    -> Submission file is valid.")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_pipeline_demonstration()
