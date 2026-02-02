import os
import shutil
import torch
import numpy as np
import pandas as pd
import sys

# Import library modules
import library.config
import library.train
import library.model
import library.loss_metric
import library.data


def run_demo():
    print("==== STARTING DEMO SCRIPT ====")

    # ------------------------------------------------------------------------
    # 1. SETUP & PATCHING
    # ------------------------------------------------------------------------
    # We patch the configuration to run a quick demo in a separate directory
    # with fewer epochs and a debug subset of data.

    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    print(f"Setting up demo environment in: {DEMO_DIR}")

    # Patch library.config
    library.config.WORKING_DIR = DEMO_DIR
    library.config.EPOCHS = 2  # Run only 2 epochs for speed

    # Patch library.train (since it imports variables directly)
    library.train.WORKING_DIR = DEMO_DIR
    library.train.EPOCHS = 2
    library.train.SUBMISSION_PATH = os.path.join(DEMO_DIR, "demo_submission.csv")

    # Ensure reproducibility
    library.config.set_seed(42)

    # ------------------------------------------------------------------------
    # 2. UNIT TESTING: MODEL & LOSS
    # ------------------------------------------------------------------------
    print("\n[Test] Verifying Model and Loss components...")

    device = torch.device("cpu")  # Use CPU for simple unit tests

    # Define shapes
    B, L = 4, 107
    in_channels = 18
    num_targets = 5

    # Instantiate Model
    model = library.model.DSRDN(in_channels=in_channels).to(device)
    model.eval()

    # Create dummy inputs
    dummy_input = torch.randn(B, in_channels, L).to(device)
    dummy_pairs = torch.full((B, L), -1, dtype=torch.long).to(device)  # No pairs

    # Forward Pass Check
    with torch.no_grad():
        preds, z = model(dummy_input, dummy_pairs)

    # Assertions
    assert preds.shape == (
        B,
        num_targets,
        L,
    ), f"Expected output shape {(B, num_targets, L)}, got {preds.shape}"
    assert z.shape == (
        B,
        library.config.HIDDEN_DIM,
        L,
    ), f"Expected latent shape {(B, library.config.HIDDEN_DIM, L)}, got {z.shape}"
    print("  -> Model forward pass successful.")

    # Loss Function Check
    criterion = library.loss_metric.MCRMSELoss()

    # Case 1: Perfect prediction (Loss should be 0)
    loss_zero = criterion(preds, preds)
    assert torch.isclose(
        loss_zero, torch.tensor(0.0)
    ), f"Loss should be 0 for identical inputs, got {loss_zero}"

    # Case 2: Known offset
    # Scored columns are indices 0, 1, 3.
    # If we add 1.0 to these columns, MSE is 1.0, RMSE is 1.0, Mean RMSE is 1.0
    target_offset = preds.clone()
    scored_indices = [0, 1, 3]
    target_offset[:, scored_indices, :] += 1.0

    # We need to consider the mask. The loss function applies a mask for the first 68 positions.
    # Since we added error everywhere, the RMSE on the first 68 positions will still be 1.0.
    loss_one = criterion(preds, target_offset)
    assert torch.isclose(
        loss_one, torch.tensor(1.0), atol=1e-5
    ), f"Expected loss 1.0 for offset 1.0 on scored cols, got {loss_one}"

    print("  -> Loss function verification successful.")

    # ------------------------------------------------------------------------
    # 3. INTEGRATION: TRAINING PIPELINE
    # ------------------------------------------------------------------------
    print("\n[Pipeline] Starting Training (Debug Mode)...")

    # train_model(debug=True) limits the dataset to 100 samples
    best_score = library.train.train_model(debug=True)

    # Verify model file creation
    model_path = os.path.join(DEMO_DIR, "best_model.pth")
    assert os.path.exists(model_path), "best_model.pth was not created!"
    print(f"  -> Training complete. Best validation score: {best_score:.4f}")

    # ------------------------------------------------------------------------
    # 4. INTEGRATION: INFERENCE PIPELINE
    # ------------------------------------------------------------------------
    print("\n[Pipeline] Generating Submission (Debug Mode)...")

    library.train.generate_submission(debug=True)

    submission_path = library.train.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file was not created!"
    print(f"  -> Submission generated at {submission_path}")

    # ------------------------------------------------------------------------
    # 5. VALIDATION OF OUTPUT
    # ------------------------------------------------------------------------
    print("\n[Validation] Checking Submission File...")

    df_sub = pd.read_csv(submission_path)

    # Expected columns
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
    ), f"Columns mismatch. Got {df_sub.columns}"

    # Expected rows: debug=True uses 100 test samples. Each sample has 107 positions.
    # Total rows = 100 * 107 = 10700
    expected_rows = 100 * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Value checks (ensure no NaNs and reasonable range)
    assert not df_sub.isnull().values.any(), "Submission contains NaNs"

    # Check if id_seqpos format is correct (e.g., "id_xxxxx_0")
    sample_id_seqpos = df_sub.iloc[0]["id_seqpos"]
    assert "_" in sample_id_seqpos and any(
        char.isdigit() for char in sample_id_seqpos
    ), f"Invalid id_seqpos format: {sample_id_seqpos}"

    print("  -> Submission file validation passed.")
    print("\n==== DEMO COMPLETED SUCCESSFULLY ====")


if __name__ == "__main__":
    run_demo()
