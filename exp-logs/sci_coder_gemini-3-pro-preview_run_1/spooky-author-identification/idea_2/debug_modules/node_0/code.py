import os
import sys
import numpy as np
import pandas as pd
import warnings
import torch

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
# Disable tokenizers parallelism to avoid deadlocks in some environments
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import library modules
from library.config import Config
from library.utils import set_seed, format_submission
from library.linear_branch import run_linear_branch
from library.transformer_branch import run_transformer_branch
from library.ensemble import run_ensemble
from library.data_loader import load_data


def main():
    print("=== Starting Author Identification Pipeline Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed
    # -------------------------------------------------------------------------
    print("Configuring environment and hyperparameters...")

    # Override Config attributes to ensure the demo runs quickly
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.MAX_FEATURES_WORD = 5000  # Limit vocabulary size for speed
    Config.MAX_FEATURES_CHAR = 5000

    # Update working directories to isolate this run
    Config.WORKING_DIR = "./working/demo_run"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.TRANSFORMER_MODEL_DIR = os.path.join(Config.WORKING_DIR, "transformer_model")
    Config.LINEAR_MODEL_PATH = os.path.join(Config.WORKING_DIR, "linear_model.joblib")

    # Initialize directories
    Config.setup()

    # Set global seeds for reproducibility
    set_seed(Config.SEED)

    print(f"Runtime Device: {Config.get_device()}")
    print(f"Training Epochs: {Config.EPOCHS}")

    # -------------------------------------------------------------------------
    # 2. Linear Model Branch
    # -------------------------------------------------------------------------
    print("\n--- [Step 1] Executing Linear Branch ---")
    # Force retraining to demonstrate the pipeline
    val_probs_lin, test_probs_lin, y_val = run_linear_branch(load_cached_data=False)

    # Verification
    assert isinstance(
        val_probs_lin, np.ndarray
    ), "Linear validation probs must be a numpy array"
    assert val_probs_lin.shape[1] == 3, "Linear output must have 3 classes"
    assert not np.isnan(val_probs_lin).any(), "Linear validation probs contain NaNs"
    print("Linear branch outputs verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Transformer Model Branch
    # -------------------------------------------------------------------------
    print("\n--- [Step 2] Executing Transformer Branch ---")
    # Train transformer (RoBERTa) for 1 epoch
    val_probs_trans, test_probs_trans, y_val_check = run_transformer_branch(
        load_cached_data=False
    )

    # Verification
    assert np.array_equal(
        y_val, y_val_check
    ), "Ground truth labels mismatch between branches"
    assert (
        val_probs_trans.shape == val_probs_lin.shape
    ), "Shape mismatch between linear and transformer outputs"
    assert not np.isnan(test_probs_trans).any(), "Transformer test probs contain NaNs"
    print("Transformer branch outputs verified successfully.")

    # -------------------------------------------------------------------------
    # 4. Ensemble
    # -------------------------------------------------------------------------
    print("\n--- [Step 3] Executing Ensemble ---")
    # Optimize weights and blend predictions
    final_test_probs = run_ensemble(
        val_probs_lin, test_probs_lin, val_probs_trans, test_probs_trans, y_val
    )

    # Verification
    assert (
        final_test_probs.shape == test_probs_lin.shape
    ), "Final predictions shape mismatch"
    print("Ensemble outputs verified successfully.")

    # -------------------------------------------------------------------------
    # 5. Submission Generation
    # -------------------------------------------------------------------------
    print("\n--- [Step 4] Generating Submission File ---")

    # Load test metadata to retrieve IDs
    df_test = load_data("test")
    test_ids = df_test["id"].values

    # Ensure alignment
    if len(test_ids) != len(final_test_probs):
        raise AssertionError(
            f"ID count ({len(test_ids)}) does not match prediction count ({len(final_test_probs)})"
        )

    # Format the submission dataframe
    submission_df = format_submission(
        test_ids, final_test_probs, columns=["EAP", "HPL", "MWS"]
    )

    # Save to the designated submission path defined in Config
    save_path = Config.SUBMISSION_FILE_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    submission_df.to_csv(save_path, index=False)

    print(f"Submission saved to: {save_path}")

    # Final sanity check of the saved file
    saved_df = pd.read_csv(save_path)
    print(f"Saved file shape: {saved_df.shape}")
    print("First 3 rows of submission:")
    print(saved_df.head(3))

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
