import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, mcrmse
from library.data import get_sinusoidal_encoding_np, get_dataloaders
from library.model import RNARegressor
from library.train import run_training

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def verify_mcrmse_logic():
    """
    Verifies the MCRMSE metric calculation with a manual example.
    """
    print("Verifying MCRMSE metric logic...")
    # Create synthetic ground truth and predictions
    # Shape: (N=2, L=3, C=2)
    y_true = np.array(
        [[[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], [[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]]]
    )

    # Predictions are off by 1.0 for the first column, and 2.0 for the second column
    y_pred = np.array(
        [[[2.0, 4.0], [2.0, 4.0], [2.0, 4.0]], [[4.0, 6.0], [4.0, 6.0], [4.0, 6.0]]]
    )

    # Manual Calculation:
    # Error Diff:
    # Col 0: (1-2)=-1, (3-4)=-1 -> Squared: 1, 1. Mean: 1. RMSE: 1.
    # Col 1: (2-4)=-2, (4-6)=-2 -> Squared: 4, 4. Mean: 4. RMSE: 2.
    # MCRMSE = (1 + 2) / 2 = 1.5

    calculated_score = mcrmse(y_true, y_pred)
    expected_score = 1.5

    assert np.isclose(
        calculated_score, expected_score
    ), f"MCRMSE verification failed. Expected {expected_score}, got {calculated_score}"
    print("MCRMSE logic verified.")


def verify_embedding_logic():
    """
    Verifies the shape and basic properties of the sinusoidal encoding.
    """
    print("Verifying Sinusoidal Embedding logic...")
    batch_size = 4
    seq_len = 10
    d_model = 32

    positions = np.random.randint(0, 100, size=(batch_size, seq_len))
    embeddings = get_sinusoidal_encoding_np(positions, d_model)

    # Check shape
    assert embeddings.shape == (
        batch_size,
        seq_len,
        d_model,
    ), f"Embedding shape mismatch. Expected {(batch_size, seq_len, d_model)}, got {embeddings.shape}"

    # Check values range [-1, 1]
    assert np.all(embeddings >= -1.0) and np.all(
        embeddings <= 1.0
    ), "Embedding values out of range [-1, 1]"
    print("Sinusoidal Embedding logic verified.")


def run_pipeline_demo():
    """
    Runs the training pipeline in debug mode and validates outputs.
    """
    print("\n" + "=" * 40)
    print("Running Training Pipeline Demo")
    print("=" * 40)

    # 1. Setup Demo Directory
    demo_dir = "./demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # 2. Monkey-patch Config to use demo directory
    # This ensures we don't overwrite the main working directory logic
    # and keeps the demo self-contained.
    print(f"Redirecting outputs to {demo_dir}...")
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir
    Config.MODEL_PATH = os.path.join(demo_dir, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(demo_dir, "submission.csv")

    # We also need to update cache paths implicitly used by get_dataloaders
    # Since get_dataloaders constructs paths based on Config.WORKING_DIR,
    # setting Config.WORKING_DIR above is sufficient.

    # 3. Execute Training
    # debug=True triggers subsampling (100 train, 50 test) and reduces epochs to 2
    try:
        run_training(debug=True)
    except Exception as e:
        raise RuntimeError(f"Training pipeline failed with error: {e}")

    # 4. Validate Outputs
    print("\nValidating pipeline outputs...")

    # Check Model File
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_PATH}")
    print(f"Model file generated: {os.path.getsize(Config.MODEL_PATH) / 1e6:.2f} MB")

    # Check Submission File
    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_FILE}"
        )

    # Validate Submission Content
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission shape: {df_sub.shape}")

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
    ), f"Submission columns mismatch. Got {list(df_sub.columns)}"

    # Expected rows:
    # In debug mode, test set is subsampled to 50 samples.
    # Each sample has 107 positions.
    # Total rows = 50 * 107 = 5350
    expected_rows = 50 * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Check for NaNs
    assert not df_sub.isnull().values.any(), "Submission contains NaN values."

    print("Pipeline demo completed and validated successfully.")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(42)

    # Run verifications
    verify_mcrmse_logic()
    verify_embedding_logic()

    # Run main pipeline
    run_pipeline_demo()
