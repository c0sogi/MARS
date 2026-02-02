import os
import sys
import torch
import numpy as np
import pandas as pd

# Append current directory to system path to ensure imports work
sys.path.append(os.getcwd())

# Import from provided library files
from library.config import Config
from library.utils import set_seed, mcrmse
from library.dataset import RNADataset, get_dataloader
from library.model import RNAModel
from library.engine import run_training, predict_and_submit, masked_mse_loss


def main():
    print("Starting demonstration script...")

    # =========================================================================
    # 1. Configuration Override for Speed and Demonstration
    # =========================================================================
    print("Configuring parameters for rapid execution...")

    # Enable Debug mode to use a small subset of data (e.g., 50 samples)
    Config.DEBUG = True
    Config.SUBSET_SIZE = 50

    # Reduce training duration
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8  # Small batch size

    # Reduce Model Complexity for speed
    Config.EMBED_DIM = 16
    Config.HIDDEN_DIM = 32  # Width will be 64
    Config.NUM_LAYERS = 2

    # Ensure directories exist (Config usually handles this, but good to be explicit)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(Config.SEED)

    # =========================================================================
    # 2. Data Loading Verification
    # =========================================================================
    print("\n--- Verifying Data Loading ---")

    # Initialize Dataset (Train)
    # forcing load_cached_data=False to demonstrate processing logic from Parquet
    train_dataset = RNADataset(mode="train", load_cached_data=False)
    print(f"Train Dataset initialized. Size (Debug Mode): {len(train_dataset)}")

    # Verify subset size
    assert (
        len(train_dataset) == Config.SUBSET_SIZE
    ), f"Expected {Config.SUBSET_SIZE} samples in debug mode, got {len(train_dataset)}"

    # Initialize DataLoader
    train_loader = get_dataloader(
        mode="train", batch_size=Config.BATCH_SIZE, shuffle=True, load_cached_data=True
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    seq, loop, dist, targets, ids = batch

    print(f"Batch loaded. IDs: {ids[:2]} ...")
    print(
        f"Shapes -> Seq: {seq.shape}, Loop: {loop.shape}, Dist: {dist.shape}, Targets: {targets.shape}"
    )

    # Verify Shapes
    # Seq/Loop/Dist: (Batch, 107)
    assert seq.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH)
    assert loop.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH)
    assert dist.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH)
    # Targets: (Batch, 68, 3)
    assert targets.shape == (Config.BATCH_SIZE, Config.PRED_LENGTH, Config.NUM_TARGETS)

    print("Data loading logic verified.")

    # =========================================================================
    # 3. Model Architecture Verification
    # =========================================================================
    print("\n--- Verifying Model Architecture ---")

    device = torch.device(Config.DEVICE)
    model = RNAModel().to(device)

    # Move batch to device
    seq = seq.to(device)
    loop = loop.to(device)
    dist = dist.to(device)

    # Forward Pass
    outputs = model(seq, loop, dist)
    print(f"Model output shape: {outputs.shape}")

    # Verify Output Shape: (Batch, 107, 3)
    # Note: Model outputs predictions for the full sequence length (107),
    # even though we only train on the first 68.
    assert outputs.shape == (Config.BATCH_SIZE, Config.SEQ_LENGTH, Config.NUM_TARGETS)

    print("Model forward pass verified.")

    # =========================================================================
    # 4. Loss and Metric Verification
    # =========================================================================
    print("\n--- Verifying Loss and Metric ---")

    # Test Loss Function
    targets = targets.to(device)
    loss = masked_mse_loss(outputs, targets)
    print(f"Calculated Loss: {loss.item():.6f}")

    # Verify Loss is a scalar and not NaN
    assert loss.dim() == 0
    assert not torch.isnan(loss)

    # Test MCRMSE Metric
    # Create dummy ground truth and predictions
    # Shape: (N, 3)
    dummy_true = np.array([[1.0, 0.5, 0.2], [0.5, 0.5, 0.5]])
    dummy_pred = np.array([[1.1, 0.5, 0.2], [0.5, 0.6, 0.5]])

    metric_val = mcrmse(dummy_true, dummy_pred)
    print(f"Calculated MCRMSE (Dummy): {metric_val:.6f}")
    assert metric_val > 0

    print("Loss and Metric functions verified.")

    # =========================================================================
    # 5. Full Training Cycle Demonstration
    # =========================================================================
    print("\n--- Running Training Cycle (Engine) ---")

    # This runs the training loop using the modified Config
    best_score = run_training()

    print(f"Training finished. Best Validation Score: {best_score}")

    # Verify Model Checkpoint exists
    assert os.path.exists(Config.MODEL_PATH), "Best model file was not saved."
    print(f"Model saved at {Config.MODEL_PATH}")

    # =========================================================================
    # 6. Inference and Submission Demonstration
    # =========================================================================
    print("\n--- Running Inference and Submission (Engine) ---")

    # This generates the submission file
    predict_and_submit()

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated."

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission loaded. Shape: {sub_df.shape}")
    print(sub_df.head(3))

    # Verify Submission Content
    # Expected rows: SUBSET_SIZE (50) * SEQ_LENGTH (107) = 5350
    expected_rows = Config.SUBSET_SIZE * Config.SEQ_LENGTH
    assert (
        len(sub_df) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(sub_df)}"

    # Verify Columns
    expected_cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    assert list(sub_df.columns) == expected_cols

    # Verify Unscored Columns are 0.0 (deg_pH10, deg_50C)
    assert (sub_df["deg_pH10"] == 0.0).all()
    assert (sub_df["deg_50C"] == 0.0).all()

    print("Submission format verified.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
