import os
import torch
import numpy as np
import pandas as pd
import shutil
import sys

# Import library components
from library.config import Config
from library.utils import seed_all, mcrmse, get_device
from library.dataset import parse_structure_to_distance, get_dataloader
from library.model import RNAModel, loss_fn
from library.engine import run_training


def main():
    print("Initializing Demo Script...")

    # ---------------------------------------------------------
    # 1. Configure for Demo (Speed & Isolation)
    # ---------------------------------------------------------
    # We modify the Config class attributes directly to isolate the demo run
    # and speed up execution by using smaller models and fewer epochs.

    # Paths
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Create directories
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Use main process to avoid overhead
    Config.HIDDEN_DIM = 64  # Reduce model size
    Config.NUM_LAYERS = 2  # Reduce depth
    Config.EMBEDDING_DIM = 32
    Config.LOOP_EMBEDDING_DIM = 16
    Config.PAIR_EMBEDDING_DIM = 16

    # Reproducibility
    seed_all(Config.SEED)
    device = get_device()
    print(f"Configuration set. Working directory: {Config.WORKING_DIR}")
    print(f"Device: {device}")

    # ---------------------------------------------------------
    # 2. Verify Utility Functions
    # ---------------------------------------------------------
    print("\n--- Verifying Utility Functions ---")

    # Test MCRMSE Metric
    # Scenario: 1 sample, 2 columns.
    # True: [[0, 0], [1, 1]]
    # Pred: [[1, 0], [1, 2]]
    # Diff: [[-1, 0], [0, -1]] -> Sq: [[1, 0], [0, 1]]
    # Col 0 MSE: (1+0)/2 = 0.5 -> RMSE: sqrt(0.5) ≈ 0.7071
    # Col 1 MSE: (0+1)/2 = 0.5 -> RMSE: sqrt(0.5) ≈ 0.7071
    # MCRMSE: (0.7071 + 0.7071) / 2 = 0.7071
    y_t = np.array([[[0, 0], [1, 1]]])  # Shape (1, 2, 2)
    y_p = np.array([[[1, 0], [1, 2]]])

    score = mcrmse(y_t, y_p)
    expected_score = np.sqrt(0.5)
    assert (
        abs(score - expected_score) < 1e-5
    ), f"MCRMSE logic error. Expected {expected_score}, got {score}"
    print("MCRMSE function verified.")

    # ---------------------------------------------------------
    # 3. Verify Data Processing Logic
    # ---------------------------------------------------------
    print("\n--- Verifying Data Processing ---")

    # Test Structure Parsing
    # Structure: "(..)" -> Indices 0, 1, 2, 3. Pair (0, 3).
    # Distances: 0->(3-0)=3, 1->0, 2->0, 3->(0-3)=-3
    struct_str = "(..)"
    dists = parse_structure_to_distance(struct_str)
    expected_dists = np.array([3.0, 0.0, 0.0, -3.0], dtype=np.float32)
    assert np.allclose(dists, expected_dists), f"Structure parsing error. Got {dists}"
    print("Structure parsing verified.")

    # ---------------------------------------------------------
    # 4. Verify Dataset and DataLoader
    # ---------------------------------------------------------
    print("\n--- Verifying Dataset & DataLoader ---")
    # We load the training set. Since we changed WORKING_DIR, it will process from source and cache it.
    # We disable loading existing cache just in case, though the dir is new.
    train_loader = get_dataloader(
        "train", batch_size=Config.BATCH_SIZE, shuffle=True, load_cached_data=False
    )

    # Fetch one batch
    batch = next(iter(train_loader))
    seq = batch["sequence"]
    target = batch["target"]
    dist = batch["pairing_distance"]

    # Check shapes
    # Seq: (B, 107)
    assert seq.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Sequence shape incorrect: {seq.shape}"
    # Target: (B, 68, 3)
    assert target.shape == (
        Config.BATCH_SIZE,
        Config.PRED_LEN,
        Config.NUM_CLASSES,
    ), f"Target shape incorrect: {target.shape}"
    # Dist: (B, 107, 1)
    assert dist.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        1,
    ), f"Distance shape incorrect: {dist.shape}"

    print("DataLoader shapes verified.")

    # ---------------------------------------------------------
    # 5. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n--- Verifying Model Architecture ---")
    model = RNAModel(Config).to(device)

    # Forward pass with batch
    seq = seq.to(device)
    loop = batch["loop_type"].to(device)
    dist = dist.to(device)
    target = target.to(device)

    preds = model(seq, loop, dist)

    # Output should be (B, 107, 3)
    assert preds.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_CLASSES,
    ), f"Model output shape incorrect: {preds.shape}"
    print("Model forward pass verified.")

    # ---------------------------------------------------------
    # 6. Verify Loss Function
    # ---------------------------------------------------------
    print("\n--- Verifying Loss Function ---")
    loss = loss_fn(preds, target)
    # Should be scalar
    assert loss.dim() == 0, "Loss is not a scalar."
    assert not torch.isnan(loss), "Loss is NaN."
    print(f"Loss calculation verified. Value: {loss.item():.5f}")

    # ---------------------------------------------------------
    # 7. Integration Test: Full Training Loop
    # ---------------------------------------------------------
    print("\n--- Running Integration Test (Training Pipeline) ---")
    # This runs training for 1 epoch (limited batches via debug=True)
    # and then generates submission for the test set.
    run_training(Config, epochs=1, patience=1, debug=True)

    # ---------------------------------------------------------
    # 8. Verify Submission Output
    # ---------------------------------------------------------
    print("\n--- Verifying Submission ---")
    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_FILE}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)

    # Check dimensions
    # Test set has 240 samples. Each has 107 positions. Total rows = 240 * 107 = 25680.
    expected_rows = 240 * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

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
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(df_sub.columns)}"

    # Check content (no NaNs)
    assert not df_sub.isnull().values.any(), "Submission contains NaN values."

    print("Submission file verified successfully.")
    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
