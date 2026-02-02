import os
import sys
import numpy as np
import pandas as pd
import torch
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# 1. Disable tqdm progress bars as per requirements
# We monkey-patch tqdm before importing the library modules that use it.
import tqdm


def noop_tqdm(*args, **kwargs):
    if args:
        return args[0]
    return []


tqdm.tqdm = noop_tqdm

# 2. Import Library Modules
from library.config import Config, set_seed
from library.data import (
    parse_structure_pairs,
    process_dataframe,
    get_dataloaders,
    SEQ_MAP,
)
from library.model import RNAModel
from library.loss import MaskedMCRMSELoss
from library.utils import GlobalMCRMSE
from library.train import run_training


def verify_data_processing_logic():
    print("\n=== Verifying Data Processing Logic ===")

    # Test Case: Simple hairpin structure
    # Sequence: G G A A ( 4 bases )
    # Structure: ( . . )
    # Indices:   0 1 2 3
    # Pairings: 0-3, 1-unpaired, 2-unpaired, 3-0

    structure = "(..)"
    expected_pairs = np.array([3, -1, -1, 0], dtype=np.int32)

    computed_pairs = parse_structure_pairs(structure)

    print(f"Structure: {structure}")
    print(f"Expected Pairs: {expected_pairs}")
    print(f"Computed Pairs: {computed_pairs}")

    np.testing.assert_array_equal(
        computed_pairs, expected_pairs, err_msg="Structure parsing logic failed."
    )
    print("✓ Structure parsing logic verified.")


def verify_model_architecture():
    print("\n=== Verifying Model Architecture ===")

    # Initialize model
    device = torch.device("cpu")
    model = RNAModel().to(device)
    model.eval()

    # Create dummy batch
    # Batch Size = 2, Seq Len = 107 (Config.SEQ_LENGTH)
    B, L = 2, Config.SEQ_LENGTH
    # Input dim is 18 (4 seq + 3 struct + 7 loop + 4 partner)
    input_dim = 18

    dummy_input = torch.randn(B, L, input_dim).to(device)
    dummy_partners = torch.full((B, L), -1, dtype=torch.long).to(device)

    print(f"Input Shape: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input, dummy_partners)

    print(f"Output Shape: {output.shape}")

    # Expected Output: (Batch, SeqLen, 5)
    expected_shape = (B, L, 5)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print("✓ Model forward pass and output shape verified.")


def verify_loss_function():
    print("\n=== Verifying Masked MCRMSE Loss ===")

    criterion = MaskedMCRMSELoss()

    # Config.SCORED_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    # Config.ALL_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Indices: 0, 1, 3 are scored. 2 and 4 are ignored.

    # Create dummy predictions and targets
    # Shape: (1, 1, 5) for simplicity
    preds = torch.zeros(1, 1, 5)
    targets = torch.zeros(1, 1, 5)

    # Set errors
    # Col 0 (Scored): Error = 1.0
    preds[0, 0, 0] = 1.0
    targets[0, 0, 0] = 0.0

    # Col 1 (Scored): Error = 0.0

    # Col 2 (Ignored): Error = 100.0 (Should not affect loss)
    preds[0, 0, 2] = 100.0

    # Col 3 (Scored): Error = 1.0
    preds[0, 0, 3] = 1.0

    # Col 4 (Ignored): Error = 50.0
    preds[0, 0, 4] = 50.0

    # Manual Calculation:
    # Scored Cols: 0, 1, 3
    # Col 0: RMSE = sqrt((1-0)^2) = 1.0
    # Col 1: RMSE = sqrt((0-0)^2) = 0.0
    # Col 3: RMSE = sqrt((1-0)^2) = 1.0
    # MCRMSE = (1.0 + 0.0 + 1.0) / 3 = 0.6666...

    loss = criterion(preds, targets)
    val = loss.item()

    print(f"Calculated Loss: {val:.6f}")
    expected_val = 2.0 / 3.0
    print(f"Expected Loss:   {expected_val:.6f}")

    assert abs(val - expected_val) < 1e-5, "Loss function calculation incorrect."
    print("✓ Masked MCRMSE Loss logic verified.")


def run_demo_pipeline():
    print("\n=== Running Training Pipeline (Demo) ===")

    # 1. Modify Config for Demo
    # Change cache version to avoid loading existing full datasets
    Config.CACHE_VERSION = "demo_execution_v1"
    # Ensure working directory exists for this version
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Cache Version set to: {Config.CACHE_VERSION}")
    print("Starting run_training with debug=True and epochs=1...")

    # 2. Run Training
    # This function encapsulates data loading, model init, training loop, validation, and submission generation.
    run_training(debug=True, epochs=1)

    # 3. Verify Submission
    submission_path = Config.SUBMISSION_PATH
    if os.path.exists(submission_path):
        df_sub = pd.read_csv(submission_path)
        print(f"\nSubmission file generated at: {submission_path}")
        print(f"Submission Shape: {df_sub.shape}")
        print("First 3 rows:")
        print(df_sub.head(3))

        # Verify columns
        expected_cols = [
            "id_seqpos",
            "reactivity",
            "deg_Mg_pH10",
            "deg_pH10",
            "deg_Mg_50C",
            "deg_50C",
        ]
        if list(df_sub.columns) == expected_cols:
            print("✓ Submission columns verified.")
        else:
            raise AssertionError(
                f"Submission columns mismatch. Got {list(df_sub.columns)}"
            )
    else:
        raise FileNotFoundError("Submission file was not generated.")


if __name__ == "__main__":
    # Set global seed for reproducibility
    set_seed(42)

    try:
        # Run Verification Steps
        verify_data_processing_logic()
        verify_model_architecture()
        verify_loss_function()

        # Run Execution Pipeline
        run_demo_pipeline()

        print("\nAll demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\n[FAILED] Assertion Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAILED] An error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
