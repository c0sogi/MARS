import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import shutil

# Import from the provided library
from library.config import Config
from library.utils import set_seed, MCRMSE
from library.dataset import get_dataloaders, compute_pair_dist
from library.model import RNAModel
from library.engine import train_one_epoch, validate, predict_and_submit


def run_demo():
    print("--- Starting Library Usage Demo ---")

    # 1. Configure for Speed and Reproducibility
    # We modify the Config class directly to run a lightweight version of the task
    set_seed(42)
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 20  # Use only 20 samples
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for this small demo

    # Initialize directories
    Config.setup()

    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Verify Metric Logic (MCRMSE)
    print("\n[1/5] Verifying Metric (MCRMSE)...")
    # Case: 2 samples, 1 column.
    # Sample 1: True=0, Pred=0 -> SqErr=0
    # Sample 2: True=0, Pred=2 -> SqErr=4
    # MSE = (0+4)/2 = 2. RMSE = sqrt(2) ~= 1.414
    y_true = torch.tensor([[0.0], [0.0]])
    y_pred = torch.tensor([[0.0], [2.0]])
    score = MCRMSE(y_true, y_pred)
    expected = np.sqrt(2.0)

    assert (
        abs(score.item() - expected) < 1e-5
    ), f"MCRMSE calculation mismatch. Got {score.item()}, expected {expected}"
    print("MCRMSE logic verified.")

    # 3. Verify Data Processing Logic
    print("\n[2/5] Verifying Data Processing (compute_pair_dist)...")
    # Structure: ((..)) -> Indices: 012345
    # Pairs: (0, 5), (1, 4). Unpaired: 2, 3
    # Dist at 0: 5-0 = 5. Dist at 5: 0-5 = -5
    # Dist at 1: 4-1 = 3. Dist at 4: 1-4 = -3
    test_struct = "((..))"
    test_len = 6
    dists = compute_pair_dist(test_struct, test_len)
    expected_dists = np.array([5.0, 3.0, 0.0, 0.0, -3.0, -5.0], dtype=np.float32)

    assert np.allclose(
        dists, expected_dists
    ), f"Pair distance calculation mismatch.\nGot: {dists}\nExp: {expected_dists}"
    print("compute_pair_dist logic verified.")

    # 4. Verify Data Loading and Shapes
    print("\n[3/5] Verifying DataLoaders and Batch Shapes...")
    # We force reload of cache to ensure debug subset is used
    cache_files = [
        os.path.join(Config.CACHE_DIR, "train_data_debug.npz"),
        os.path.join(Config.CACHE_DIR, "val_data_debug.npz"),
        os.path.join(Config.CACHE_DIR, "test_data_debug.npz"),
    ]
    for f in cache_files:
        if os.path.exists(f):
            os.remove(f)

    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.DEBUG, load_cached_data=False
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    # Check keys
    required_keys = ["sequence", "loop_type", "pair_dist", "position", "targets", "id"]
    for key in required_keys:
        assert key in batch, f"Batch missing key: {key}"

    # Check Shapes
    # Sequence: (Batch, 107)
    assert batch["sequence"].shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
    ), f"Sequence shape incorrect: {batch['sequence'].shape}"

    # Targets: (Batch, 68, 3) -> Only first 68 positions have targets
    assert batch["targets"].shape == (
        Config.BATCH_SIZE,
        Config.SEQ_SCORED,
        Config.NUM_TARGETS,
    ), f"Target shape incorrect: {batch['targets'].shape}"

    print("DataLoader shapes verified.")

    # 5. Verify Model Architecture
    print("\n[4/5] Verifying Model Architecture...")
    model = RNAModel().to(device)

    # Move batch to device
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    # Forward pass
    # Model outputs predictions for the full sequence length (107)
    preds = model(batch)

    assert preds.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LENGTH,
        Config.NUM_TARGETS,
    ), f"Model output shape incorrect: {preds.shape}"

    print("Model forward pass verified.")

    # 6. Verify Training and Inference Engine
    print("\n[5/5] Verifying Training and Inference Engine...")

    optimizer = optim.AdamW(model.parameters(), lr=Config.LR)
    criterion = nn.MSELoss()

    # A. Train one epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Training check passed. Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # B. Validate
    val_score = validate(model, val_loader, device)
    print(f"Validation check passed. MCRMSE: {val_score:.4f}")
    assert not np.isnan(val_score), "Validation score is NaN"

    # Save this model as 'best_model' for the prediction step
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)

    # C. Predict and Submit
    # This reads the best model from disk and creates submission.csv
    predict_and_submit(test_loader, device)

    # Verify submission file exists and has correct format
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file loaded. Rows: {len(df_sub)}")

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
    ), f"Submission columns mismatch: {df_sub.columns}"

    # Check row count: (Num Test Samples * Seq Length)
    # In debug mode, test set size is Config.DEBUG_SUBSET_SIZE (20)
    # Seq Length is 107. Total rows = 20 * 107 = 2140
    expected_rows = Config.DEBUG_SUBSET_SIZE * Config.SEQ_LENGTH
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    print("Inference and Submission verified.")
    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
