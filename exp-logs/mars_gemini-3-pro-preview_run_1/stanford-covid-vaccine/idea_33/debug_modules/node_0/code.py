import os
import shutil
import torch
import numpy as np
import pandas as pd

# Import classes and functions from the provided library files
from library.config import Config
from library.dataset import get_dataset
from library.model import RNAModel
from library.engine import run_training, run_inference
from library.utils import seed_everything, mcrmse_loss, format_submission

# ==================================================================================
# DEMO CONFIGURATION
# ==================================================================================


class DemoConfig(Config):
    """
    Configuration optimized for a quick demonstration run.
    Reduces model size and training duration.
    """

    # Use a specific subdirectory for demo outputs
    WORKING_DIR = "./working/demo_run"
    SUBMISSION_PATH = os.path.join(WORKING_DIR, "submission.csv")

    # Reduced Model Hyperparameters for speed
    EMBED_DIM = 32
    HIDDEN_DIM = 64
    N_LAYERS = 2
    DROPOUT = 0.0

    # Minimal Training Settings
    EPOCHS = 2
    BATCH_SIZE = 4
    NUM_WORKERS = 0  # Disable multiprocessing for small data to avoid overhead

    # Hardware
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================================================
# VERIFICATION FUNCTIONS
# ==================================================================================


def clean_working_dir(path):
    """Cleans the working directory to ensure a fresh run."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def test_dataset_loading():
    print("\n=== Testing Dataset Loading ===")
    # Load a tiny subset (10 samples) of the training data
    # This verifies that the parquet reading and processing pipeline works
    ds = get_dataset("train", DemoConfig, load_cached_data=False, num_samples=10)

    # Assertions
    assert len(ds) == 10, f"Expected 10 samples, got {len(ds)}"

    item = ds[0]
    required_keys = ["sequence", "loop_type", "pair_index", "distance", "targets"]
    for k in required_keys:
        assert k in item, f"Missing key {k} in dataset item"

    # Check data shapes
    # Sequence length is fixed at 107
    seq_len = item["sequence"].shape[0]
    assert seq_len == 107, f"Expected sequence length 107, got {seq_len}"
    # Targets should be (107, 3)
    assert item["targets"].shape == (
        107,
        3,
    ), f"Expected targets shape (107, 3), got {item['targets'].shape}"

    print("Dataset loading verification passed.")


def test_model_forward():
    print("\n=== Testing Model Forward Pass ===")
    # Instantiate model with reduced config
    model = RNAModel(DemoConfig).to(DemoConfig.DEVICE)
    model.eval()

    # Create a dummy batch
    B, L = 2, 107
    dummy_batch = {
        "sequence": torch.randint(0, 4, (B, L)).to(DemoConfig.DEVICE),
        "loop_type": torch.randint(0, 7, (B, L)).to(DemoConfig.DEVICE),
        "pair_index": torch.full((B, L), -1)
        .long()
        .to(DemoConfig.DEVICE),  # All unpaired
        "distance": torch.zeros((B, L)).float().to(DemoConfig.DEVICE),
        "targets": torch.zeros((B, L, 3)).float().to(DemoConfig.DEVICE),
    }

    # Run forward pass
    with torch.no_grad():
        output = model(dummy_batch)

    # Check output shape: [Batch, Seq_Len, N_Targets]
    assert output.shape == (
        B,
        L,
        3,
    ), f"Expected output shape ({B}, {L}, 3), got {output.shape}"
    print("Model forward pass verification passed.")


def test_metric_logic():
    print("\n=== Testing Metric Logic (MCRMSE) ===")
    # Create dummy predictions and targets
    # Shape: (N, 3)
    y_true = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    y_pred = torch.tensor([[1.1, 2.0, 3.0], [0.9, 2.0, 3.0]])

    # Manual Calculation:
    # Col 1 (Reactivity): |1.0-1.1|=0.1, |1.0-0.9|=0.1. MSE = 0.01. RMSE = 0.1
    # Col 2: Error 0. RMSE = 0
    # Col 3: Error 0. RMSE = 0
    # MCRMSE = (0.1 + 0 + 0) / 3 = 0.0333...

    loss = mcrmse_loss(y_true, y_pred)
    expected = 0.1 / 3.0

    assert torch.isclose(
        loss, torch.tensor(expected), atol=1e-5
    ), f"Expected loss {expected}, got {loss.item()}"

    print("Metric logic verification passed.")


def test_submission_formatting():
    print("\n=== Testing Submission Formatting ===")
    ids = ["id_001", "id_002"]
    # 2 samples, 107 length, 3 predicted targets
    preds = np.zeros((2, 107, 3))
    preds[:, :, 0] = 0.1  # reactivity
    preds[:, :, 1] = 0.2  # deg_Mg_pH10
    preds[:, :, 2] = 0.3  # deg_Mg_50C

    df = format_submission(ids, preds, seq_length=107)

    # Check total rows: 2 samples * 107 positions = 214 rows
    assert len(df) == 214, f"Expected 214 rows, got {len(df)}"

    # Check columns
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
    ), f"Columns mismatch. Got {list(df.columns)}"

    # Check value mapping
    # Unscored columns should be 0.0
    assert df["deg_pH10"].sum() == 0
    assert df["deg_50C"].sum() == 0
    # Check a specific value
    assert df["reactivity"].iloc[0] == 0.1

    print("Submission formatting verification passed.")


def run_full_pipeline():
    print("\n=== Running Full Training & Inference Pipeline ===")

    # 1. Training
    # We use a small subset (32 samples) to simulate a training run quickly
    print("Starting Training (2 Epochs, 32 samples)...")
    best_model_path = run_training(DemoConfig, debug=True, num_samples=32)

    assert os.path.exists(best_model_path), "Best model file was not created."
    print(f"Training finished. Model saved at {best_model_path}")

    # 2. Inference
    # Runs inference on the public test set (240 samples) using the trained model
    print("Starting Inference on Test Set...")
    run_inference(DemoConfig)

    assert os.path.exists(
        DemoConfig.SUBMISSION_PATH
    ), "Submission file was not created."

    # Validate submission file
    df_sub = pd.read_csv(DemoConfig.SUBMISSION_PATH)
    # Test set has 240 samples. 240 * 107 = 25680 rows.
    assert len(df_sub) == 25680, f"Expected 25680 rows in submission, got {len(df_sub)}"

    print(
        f"Pipeline execution successful. Submission saved to {DemoConfig.SUBMISSION_PATH}"
    )


# ==================================================================================
# MAIN EXECUTION
# ==================================================================================

if __name__ == "__main__":
    # Setup
    clean_working_dir(DemoConfig.WORKING_DIR)
    seed_everything(DemoConfig.SEED)

    # Run Unit Verifications
    test_dataset_loading()
    test_model_forward()
    test_metric_logic()
    test_submission_formatting()

    # Run Integration Pipeline
    run_full_pipeline()

    print("\nAll demonstrations completed successfully.")
