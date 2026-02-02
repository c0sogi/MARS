import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import warnings


# 1. Suppress TQDM and Warnings
# Monkeypatch tqdm to be silent before importing library modules that use it
def silent_tqdm(iterable, *args, **kwargs):
    return iterable


import tqdm

tqdm.tqdm = silent_tqdm

warnings.filterwarnings("ignore")

# 2. Import Library Modules
from library.config import Config
from library.utils import seed_everything, get_couples, scored_mcrmse
from library.dataset import get_loaders, RNADataset
from library.model import SDIN_CG_BiGRU
from library.train import train_fn, eval_fn, inference


def main():
    print("=== Starting RNA Degradation Prediction Demo ===")

    # 3. Setup & Configuration Override
    seed_everything(42)

    # Define temporary paths in working directory
    demo_dir = os.path.join(Config.WORKING_DIR, "demo_run")
    os.makedirs(demo_dir, exist_ok=True)

    demo_train_path = os.path.join(demo_dir, "train_subset.parquet")
    demo_val_path = os.path.join(demo_dir, "val_subset.parquet")
    demo_test_path = os.path.join(demo_dir, "test_subset.parquet")

    # Override Config for speed
    print("Configuring demo parameters...")
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = demo_dir  # Use demo dir for cache to avoid conflicts
    Config.TRAIN_DATA_PATH = demo_train_path
    Config.VAL_DATA_PATH = demo_val_path
    Config.TEST_DATA_PATH = demo_test_path
    Config.MODEL_SAVE_PATH = os.path.join(demo_dir, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "demo_submission.csv")

    # Hyperparameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.HIDDEN_DIM = 64  # Reduced from 384
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.VERBOSE = False

    # 4. Create Data Subsets
    print("Creating data subsets...")
    # Load original metadata
    orig_train_path = "./metadata/train.parquet"
    orig_val_path = "./metadata/val.parquet"
    orig_test_path = "./metadata/test.parquet"

    # Read top 20 rows and save to demo paths
    pd.read_parquet(orig_train_path).head(20).to_parquet(demo_train_path)
    pd.read_parquet(orig_val_path).head(10).to_parquet(demo_val_path)
    pd.read_parquet(orig_test_path).head(5).to_parquet(demo_test_path)

    # 5. Verify Utils Logic
    print("Verifying utility functions...")
    structure = "((..))"
    couples = get_couples(structure)
    # Expected: 0 pairs with 5, 1 pairs with 4, 2 and 3 are -1
    expected_couples = np.array([5, 4, -1, -1, 1, 0])
    # Note: get_couples returns indices. Index 0 '(' pairs with index 5 ')'.
    # Index 1 '(' pairs with index 4 ')'.
    assert np.array_equal(couples, expected_couples), f"get_couples failed: {couples}"
    print("  > get_couples: OK")

    # 6. Data Loading & Dataset Verification
    print("Initializing DataLoaders...")
    # Force reload by ignoring existing cache if any (though we changed cache dir)
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    # Check Batch Structure
    batch = next(iter(train_loader))
    features = batch["features"]
    indices = batch["indices"]
    mask = batch["mask"]
    targets = batch["targets"]

    # Assertions
    assert features.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_DIM,
    ), f"Feature shape mismatch: {features.shape}"
    assert indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Indices shape mismatch: {indices.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), f"Targets shape mismatch: {targets.shape}"
    print("  > DataLoader shapes: OK")

    # 7. Model Initialization & Forward Pass
    print("Initializing Model...")
    device = Config.DEVICE
    model = SDIN_CG_BiGRU().to(device)

    print("Running Forward Pass...")
    features = features.to(device)
    indices = indices.to(device)
    mask = mask.to(device)
    targets = targets.to(device)

    outputs = model(features, indices, mask)

    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        5,
    ), f"Model output shape mismatch: {outputs.shape}"
    print("  > Forward Pass: OK")

    # 8. Metric Verification
    print("Verifying Metric (MCRMSE)...")
    # Calculate metric on this dummy batch
    loss_val = scored_mcrmse(outputs, targets)
    assert isinstance(loss_val.item(), float), "Metric did not return a float"
    assert loss_val.item() >= 0, "Metric cannot be negative"
    print(f"  > MCRMSE Calculation: OK (Value: {loss_val.item():.4f})")

    # 9. Training Loop Simulation
    print("Simulating Training Loop (1 Epoch)...")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    criterion = nn.MSELoss()

    # Train
    train_loss = train_fn(model, train_loader, optimizer, criterion, device, scheduler)
    print(f"  > Train Epoch Loss: {train_loss:.6f}")

    # Eval
    val_mcrmse = eval_fn(model, val_loader, device)
    print(f"  > Val MCRMSE: {val_mcrmse:.6f}")

    # Save Model (simulating checkpointing)
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint not saved"

    # 10. Inference & Submission
    print("Running Inference on Test Set...")
    # Load model state
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    preds, ids = inference(model, test_loader, device)

    assert len(preds) == 5, "Prediction count mismatch (expected 5 from subset)"
    assert preds.shape == (5, Config.SEQ_LEN, 5), "Prediction tensor shape mismatch"

    print("Generating Submission DataFrame...")
    submission_rows = []
    target_cols = Config.TARGET_COLS

    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]
        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos].tolist()

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_values[col_idx]

            submission_rows.append(row_dict)

    submission_df = pd.DataFrame(submission_rows)

    # Check submission format
    expected_cols = ["id_seqpos"] + target_cols
    assert list(submission_df.columns) == expected_cols, "Submission columns mismatch"
    assert len(submission_df) == 5 * Config.SEQ_LEN, "Submission row count mismatch"

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"  > Submission saved to {Config.SUBMISSION_PATH}")
    print(f"  > Submission Shape: {submission_df.shape}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
