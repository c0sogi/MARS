import os
import shutil
import torch
import pandas as pd
import numpy as np

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.data import get_dataloaders
from library.model import SDBR_BiGRU
from library.loss import MCRMSELoss
from library.engine import train_pipeline, predict_and_submit


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("Initializing Demo Configuration...")

    # Set seed for reproducibility
    seed_everything(42)

    # Define a demo-specific working directory to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)  # Clean start
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths and parameters for the demo
    # We modify the class attributes directly so they propagate to other modules
    Config.WORKING_DIR = DEMO_DIR
    Config.CACHE_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission_demo.csv")
    Config.BEST_MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")

    # Optimize for speed: 1 epoch, small batch
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set device dynamically
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {Config.DEVICE}")

    # Ensure directories exist
    Config.setup_directories()

    # ==========================================
    # 2. Verify Data Processing & Loading
    # ==========================================
    print("\n[Verification] Data Loading...")

    # Load dataloaders
    # load_cached_data=True allows using cache if it exists, but since DEMO_DIR is new,
    # it will process from parquet files.
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Fetch a single batch to verify structure
    batch = next(iter(train_loader))
    inputs = batch["inputs"]
    targets = batch["targets"]
    pair_index = batch["pair_index"]
    pair_mask = batch["pair_mask"]

    # Assertions
    # Inputs: (Batch, SeqLen, Channels) -> (16, 107, 14)
    assert inputs.shape == (
        Config.BATCH_SIZE,
        107,
        14,
    ), f"Input shape incorrect. Expected ({Config.BATCH_SIZE}, 107, 14), got {inputs.shape}"

    # Targets: (Batch, ScoredSeqLen, NumTargets) -> (16, 68, 5)
    assert targets.shape == (
        Config.BATCH_SIZE,
        68,
        5,
    ), f"Target shape incorrect. Expected ({Config.BATCH_SIZE}, 68, 5), got {targets.shape}"

    print("Data loading verified successfully.")

    # ==========================================
    # 3. Verify Model & Loss Logic
    # ==========================================
    print("\n[Verification] Model & Loss...")

    device = torch.device(Config.DEVICE)
    model = SDBR_BiGRU().to(device)
    criterion = MCRMSELoss()

    # Move batch to device
    b_inputs = inputs.to(device)
    b_pair_index = pair_index.to(device)
    b_pair_mask = pair_mask.to(device)
    b_targets = targets.to(device)

    # Forward Pass
    preds = model(b_inputs, b_pair_index, b_pair_mask)

    # Check Output Shape: (Batch, SeqLen, NumTargets) -> (16, 107, 5)
    # Note: Model outputs predictions for the full sequence length (107)
    assert preds.shape == (
        Config.BATCH_SIZE,
        107,
        5,
    ), f"Model output shape incorrect. Expected ({Config.BATCH_SIZE}, 107, 5), got {preds.shape}"

    # Loss Calculation
    # Must slice predictions to match targets (first 68 positions)
    preds_sliced = preds[:, : Config.SEQ_SCORED, :]
    loss = criterion(preds_sliced, b_targets)

    # Check Loss validity
    assert not torch.isnan(loss), "Loss returned NaN."
    assert loss.item() > 0, "Loss should be positive."

    print(
        f"Model forward pass and loss calculation verified. Initial Loss: {loss.item():.4f}"
    )

    # ==========================================
    # 4. Execute Training Pipeline
    # ==========================================
    print("\n[Execution] Starting Training Pipeline (1 Epoch)...")

    # Run training for 1 epoch
    # We pass explicit arguments to ensure our runtime config overrides defaults
    train_pipeline(epochs=1, batch_size=Config.BATCH_SIZE, debug=False)

    # Verify checkpoint creation
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), f"Model checkpoint not found at {Config.BEST_MODEL_PATH}"

    print("Training pipeline completed successfully.")

    # ==========================================
    # 5. Execute Prediction Pipeline
    # ==========================================
    print("\n[Execution] Starting Prediction Pipeline...")

    predict_and_submit(batch_size=Config.BATCH_SIZE)

    # Verify submission file existence
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Verify submission format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check dimensions
    # 240 test samples * 107 positions = 25680 rows
    expected_rows = 240 * 107
    assert (
        len(sub_df) == expected_rows
    ), f"Submission row count incorrect. Expected {expected_rows}, got {len(sub_df)}"

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
        list(sub_df.columns) == expected_cols
    ), f"Submission columns incorrect. Expected {expected_cols}, got {list(sub_df.columns)}"

    # Check for NaNs
    assert not sub_df.isnull().values.any(), "Submission contains NaN values."

    print("Prediction pipeline completed and verified.")
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")
    print("\nDemo execution finished successfully.")


if __name__ == "__main__":
    run_demo()
