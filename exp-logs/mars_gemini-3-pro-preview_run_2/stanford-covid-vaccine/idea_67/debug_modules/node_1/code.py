import os
import sys
import torch
import numpy as np
import pandas as pd

# Ensure the library modules can be imported
sys.path.append(".")

from library.utils import seed_everything, metric_mcrmse, MCRMSELoss
from library.model import AHS_DFN
from library.train import train_and_predict
from library.config import SUBMISSION_FILE_PATH, SEQ_LEN, NUM_TARGETS, SCORED_LEN


def verify_model_architecture():
    """
    Instantiates the model and performs a forward pass with dummy data
    to verify output shapes and graph execution.
    """
    print("Verifying Model Architecture...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AHS_DFN().to(device)
    model.eval()

    # Create dummy inputs
    # Batch size = 2, Sequence Length = 107, Channels = 18 (as defined in AHS_DFN)
    batch_size = 2
    dummy_input = torch.randn(batch_size, SEQ_LEN, 18).to(device)

    # Partner indices: -1 for unpaired, or indices 0-106.
    # For simplicity, let's assume all unpaired (-1)
    dummy_partners = torch.full((batch_size, SEQ_LEN), -1, dtype=torch.long).to(device)

    with torch.no_grad():
        # The model returns a tuple: (y_hat_1, y_hat_2)
        y1, y2 = model(dummy_input, dummy_partners)

    # Check shapes
    expected_shape = (batch_size, SEQ_LEN, NUM_TARGETS)

    assert (
        y1.shape == expected_shape
    ), f"Pass 1 output shape mismatch. Expected {expected_shape}, got {y1.shape}"
    assert (
        y2.shape == expected_shape
    ), f"Pass 2 output shape mismatch. Expected {expected_shape}, got {y2.shape}"

    print("Model architecture verification passed.")


def verify_metric_logic():
    """
    Verifies the MCRMSE metric calculation with known values.
    """
    print("Verifying Metric Logic...")

    # Create synthetic ground truth and predictions
    # Shape: (N=1, Seq=107, Targets=5)
    # We only care about the first 68 positions (SCORED_LEN) and indices [0, 1, 3]

    targets = np.zeros((1, SEQ_LEN, NUM_TARGETS))
    preds = np.zeros((1, SEQ_LEN, NUM_TARGETS))

    # Set a known error for a scored position and scored column
    # Index 0 (reactivity) at seq pos 0
    targets[0, 0, 0] = 1.0
    preds[0, 0, 0] = 0.0  # Error = 1.0, Squared Error = 1.0

    # Set a known error for an unscored column (e.g., index 2: deg_pH10)
    targets[0, 0, 2] = 1.0
    preds[0, 0, 2] = 0.0  # Should be ignored

    # Set a known error for an unscored position (e.g., index 70)
    targets[0, 70, 0] = 1.0
    preds[0, 70, 0] = 0.0  # Should be ignored

    # Calculation:
    # Scored columns: 0, 1, 3.
    # Col 0 MSE: (1.0^2) / 68 (averaged over scored len) = 1/68
    # Col 1 MSE: 0.0
    # Col 3 MSE: 0.0
    # RMSEs: sqrt(1/68), 0, 0
    # MCRMSE: (sqrt(1/68) + 0 + 0) / 3

    expected_mcrmse = (np.sqrt(1.0 / SCORED_LEN)) / 3.0
    calculated_mcrmse = metric_mcrmse(preds, targets)

    assert np.isclose(
        calculated_mcrmse, expected_mcrmse, atol=1e-6
    ), f"Metric calculation mismatch. Expected {expected_mcrmse:.6f}, got {calculated_mcrmse:.6f}"

    print("Metric logic verification passed.")


def verify_submission_format(debug_size=100):
    """
    Checks if the generated submission file exists and has the correct shape.
    """
    print("Verifying Submission File...")

    if not os.path.exists(SUBMISSION_FILE_PATH):
        raise FileNotFoundError(f"Submission file not found at {SUBMISSION_FILE_PATH}")

    df = pd.read_csv(SUBMISSION_FILE_PATH)

    # Expected rows: debug_size * SEQ_LEN
    expected_rows = debug_size * SEQ_LEN

    assert (
        len(df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df)}"

    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert (
        list(df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df.columns)}"

    print("Submission format verification passed.")


if __name__ == "__main__":
    # 1. Set Random Seeds for Reproducibility
    seed_everything(42)

    # 2. Verify Components
    verify_model_architecture()
    verify_metric_logic()

    # 3. Run Training and Prediction Pipeline
    # We use debug=True to limit the dataset to 100 samples.
    # We use epochs=2 to ensure the training loop completes quickly.
    print("\nStarting Training Pipeline (Debug Mode)...")
    train_and_predict(debug=True, epochs=2)

    # 4. Verify Output
    verify_submission_format(debug_size=100)

    print("\nDemonstration completed successfully.")
