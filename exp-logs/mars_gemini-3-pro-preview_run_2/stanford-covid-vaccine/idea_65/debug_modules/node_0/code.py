import sys
import os
import types
import pandas as pd
import torch
import numpy as np


# ==================================================================================
# 1. PATCH LIBRARY.CONFIG TO PREVENT AUTO-EXECUTION
# ==================================================================================
# The provided library/config.py executes run_pipeline() at the global scope.
# We must prevent this to run our own optimized demo.
def patch_config_module():
    config_path = "./library/config.py"
    with open(config_path, "r") as f:
        source = f.read()

    # Remove the line that triggers the pipeline
    # We look for the exact call "run_pipeline()"
    source_lines = source.splitlines()
    patched_lines = [line for line in source_lines if "run_pipeline()" not in line]
    patched_source = "\n".join(patched_lines)

    # Create the module dynamically
    module_name = "library.config"
    mod = types.ModuleType(module_name)

    # Execute the patched source in the module's namespace
    exec(patched_source, mod.__dict__)

    # Register in sys.modules so other imports use this version
    sys.modules[module_name] = mod


print("Patching library.config to prevent automatic training run...")
patch_config_module()

# ==================================================================================
# 2. IMPORTS FROM LIBRARY
# ==================================================================================
# Now it is safe to import the rest of the library
import library.config as config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import SSRFN
from library.engine import fit, inference

# ==================================================================================
# 3. DEMONSTRATION SCRIPT
# ==================================================================================


def run_demo():
    # Setup
    DEMO_WORKING_DIR = "./working/demo_task"
    DEMO_SUBMISSION_PATH = os.path.join(DEMO_WORKING_DIR, "submission.csv")
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    print(f"Starting Demo in {DEMO_WORKING_DIR}")
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------------------
    # A. Data Loading & Verification
    # ------------------------------------------------------------------------------
    print("\n[1/4] Loading Data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        working_dir=DEMO_WORKING_DIR,
        batch_size=8,  # Small batch size for speed
        load_cached_data=True,  # Use cache if available
    )

    # Verify Train Batch
    sample_inputs, sample_targets = next(iter(train_loader))

    # Assert Shapes
    # Inputs: seq (B, 107), struct (B, 107), loop (B, 107), pid (B, 107), partner_idx (B, 107)
    B, L = sample_inputs["seq"].shape
    assert L == 107, f"Expected sequence length 107, got {L}"
    assert sample_targets.shape == (
        B,
        107,
        5,
    ), f"Expected targets (B, 107, 5), got {sample_targets.shape}"

    print(f"Data Loaded Successfully. Batch Shape: {sample_inputs['seq'].shape}")

    # ------------------------------------------------------------------------------
    # B. Model Instantiation & Logic Verification
    # ------------------------------------------------------------------------------
    print("\n[2/4] Verifying Model Logic...")
    model = SSRFN().to(device)

    # Move sample to device
    sample_inputs_dev = {k: v.to(device) for k, v in sample_inputs.items()}

    # Test Forward Pass (Training Mode: returns y2, y1)
    model.train()
    y2, y1 = model(sample_inputs_dev)
    assert y2.shape == (B, 107, 5), f"Model output shape mismatch: {y2.shape}"
    assert y1.shape == (B, 107, 5), f"Model auxiliary output shape mismatch: {y1.shape}"

    # Test Forward Pass (Eval Mode: returns y2 only)
    model.eval()
    with torch.no_grad():
        preds = model(sample_inputs_dev)
    assert preds.shape == (B, 107, 5), "Eval mode output shape mismatch"

    print("Model Logic Verified.")

    # ------------------------------------------------------------------------------
    # C. Training Loop (Optimized for Speed)
    # ------------------------------------------------------------------------------
    print("\n[3/4] Running Training Loop (1 Epoch)...")

    # We use the fit function from library.engine
    # We pass epochs=1 to ensure it finishes quickly
    best_model_path, _ = fit(
        epochs=1, batch_size=16, working_dir=DEMO_WORKING_DIR, load_cached_data=True
    )

    assert os.path.exists(best_model_path), "Best model file was not saved."
    print(f"Training Complete. Model saved to {best_model_path}")

    # ------------------------------------------------------------------------------
    # D. Inference & Submission
    # ------------------------------------------------------------------------------
    print("\n[4/4] Generating Submission...")

    inference(
        model_path=best_model_path,
        test_loader=test_loader,
        submission_path=DEMO_SUBMISSION_PATH,
    )

    # Verify Submission
    assert os.path.exists(DEMO_SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(DEMO_SUBMISSION_PATH)

    # Expected rows: 240 samples * 107 positions = 25680 rows
    # Note: The test set in metadata/test.csv has 240 rows.
    expected_rows = 240 * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df_sub)}"

    # Check columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch."

    # Check values are numeric
    assert (
        pd.to_numeric(df_sub["reactivity"], errors="coerce").notnull().all()
    ), "Non-numeric predictions found."

    print(f"Submission Verified. Shape: {df_sub.shape}")
    print("\n=== DEMO COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    run_demo()
