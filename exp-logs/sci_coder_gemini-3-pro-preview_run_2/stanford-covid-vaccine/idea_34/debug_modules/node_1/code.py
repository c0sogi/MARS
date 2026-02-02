import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# 1. Patch Configuration for Speed and Demo constraints
# We must import config and modify it BEFORE importing other library modules
# because they import constants from config using 'from library.config import ...'
import library.config

print("Configuring environment for demo...")

# Redirect paths to working directory
library.config.WORKING_DIR = "./working"
library.config.TRAIN_CACHE = os.path.join(library.config.WORKING_DIR, "demo_train.npz")
library.config.VAL_CACHE = os.path.join(library.config.WORKING_DIR, "demo_val.npz")
library.config.TEST_CACHE = os.path.join(library.config.WORKING_DIR, "demo_test.npz")
library.config.MODEL_SAVE_PATH = os.path.join(
    library.config.WORKING_DIR, "demo_model.pth"
)
library.config.SUBMISSION_PATH = os.path.join(
    library.config.WORKING_DIR, "demo_submission.csv"
)

# Reduce Model Complexity for speed
library.config.NUM_LAYERS = 2
library.config.DILATIONS = [
    1,
    2,
]  # Must match NUM_LAYERS length logic if used strictly, though model handles list len
library.config.HIDDEN_DIM = 32
library.config.FEEDBACK_DIM = 16
library.config.RNN_HIDDEN_DIM = 48  # (32 + 16)

# Reduce Training Loop
library.config.EPOCHS = 1
library.config.BATCH_SIZE = 16
library.config.PATIENCE = 1
library.config.NUM_WORKERS = 0  # Use 0 for safer execution in some envs, or keep 2

# 2. Import other library modules
# These will now pick up the modified values from library.config
import library.utils
import library.data
import library.model
import library.train


def run_demo():
    print("\n=== Starting RNA Degradation Model Demo ===\n")

    # --- Step 1: Verify Utility Functions ---
    print("[1/5] Verifying Utility Functions...")

    # Test MCRMSE Loss
    # Create dummy prediction (zeros) and target (ones)
    # Scored columns are 0, 1, 3.
    # Diff is 1.0 for all. RMSE is 1.0. Mean is 1.0.
    pred = torch.zeros(2, 10, 5)
    target = torch.ones(2, 10, 5)
    loss = library.utils.mcrmse_loss(pred, target)

    assert torch.isclose(
        loss, torch.tensor(1.0)
    ), f"MCRMSE Loss calculation incorrect. Got {loss.item()}"
    print("   -> MCRMSE Loss verified.")

    # --- Step 2: Verify Data Processing Logic ---
    print("[2/5] Verifying Data Processing Logic...")

    # Test Partner Indices Extraction
    # Structure: "((..))" -> Indices: 0,1,2,3,4,5
    # Pairs: (0,5), (1,4). Unpaired: 2,3
    structure = "((..))"
    expected_indices = np.array([5, 4, -1, -1, 1, 0])
    calculated_indices = library.data.get_partner_indices(structure)

    np.testing.assert_array_equal(
        calculated_indices,
        expected_indices,
        err_msg="Partner indices logic is incorrect",
    )
    print("   -> Partner indices logic verified.")

    # --- Step 3: Verify Model Architecture ---
    print("[3/5] Verifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = library.model.LFDCN().to(device)

    # Create dummy inputs matching the dimensions
    # Batch=2, Seq=107, Features=18
    dummy_input = torch.randn(2, 107, 18).to(device)
    dummy_partners = torch.zeros(2, 107).long().to(device)
    dummy_targets = torch.randn(2, 107, 5).to(device)

    # Run forward pass (Training mode with targets)
    loss_out, pred_out = model(dummy_input, dummy_partners, targets=dummy_targets)

    # Check output shapes
    assert pred_out.shape == (2, 107, 5), f"Prediction shape mismatch: {pred_out.shape}"
    assert loss_out.ndim == 0, "Loss should be a scalar"
    print(f"   -> Model forward pass successful. Output shape: {pred_out.shape}")

    # --- Step 4: Run Training Loop ---
    print("[4/5] Running Training Loop (1 Epoch)...")

    # This function handles data loading, training loop, and saving the model
    library.train.train_model()

    if not os.path.exists(library.config.MODEL_SAVE_PATH):
        raise FileNotFoundError("Model file was not saved after training.")

    print("   -> Training completed. Model saved.")

    # --- Step 5: Generate Submission ---
    print("[5/5] Generating Submission...")

    # This function loads the saved model and generates predictions for test.json
    library.model.generate_submission()

    if not os.path.exists(library.config.SUBMISSION_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    # Verify Submission Content
    df_sub = pd.read_csv(library.config.SUBMISSION_PATH)

    # Expected rows: 240 test samples * 107 positions = 25680
    expected_rows = 240 * 107
    if len(df_sub) != expected_rows:
        raise AssertionError(
            f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"
        )

    # Expected columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    if list(df_sub.columns) != expected_cols:
        raise AssertionError(
            f"Submission columns mismatch.\nExpected: {expected_cols}\nGot: {list(df_sub.columns)}"
        )

    print(f"   -> Submission verified. Shape: {df_sub.shape}")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    # Set seed for reproducibility
    library.utils.set_seed(42)
    run_demo()
