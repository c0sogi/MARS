import os
import torch
import numpy as np
import pandas as pd
import shutil
import time

# Import provided library modules
from library.config import Config
from library.utils import set_seed, save_submission
from library.dataset import get_dataloaders
from library.model import SpatiallyAugmentedBiGRU
from library.loss import WeightedMCRMSELoss
from library.train import Trainer


def main():
    print("==== RNA Degradation Prediction Demo ====")

    # ---------------------------------------------------------
    # 1. Configuration Override for Demo
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for fast execution...")

    # Set a specific working directory for this demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_execution/"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Optimize for speed
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples
    Config.EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Setup system (creates directories, sets seeds)
    Config.setup_system()
    set_seed(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Subset Size: {Config.DEBUG_SUBSET_SIZE}")

    # ---------------------------------------------------------
    # 2. Data Loading & Verification
    # ---------------------------------------------------------
    print("\n[2] Loading data and verifying shapes...")

    # Get dataloaders
    loaders = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,  # Force re-creation for the subset
    )

    train_loader = loaders["train"]
    test_loader = loaders["test"]

    # Verify Train Batch
    X_train, y_train, w_train = next(iter(train_loader))

    print(f"Train Batch - X shape: {X_train.shape}")
    print(f"Train Batch - y shape: {y_train.shape}")
    print(f"Train Batch - w shape: {w_train.shape}")

    # Assertions for Train Data
    # X: (Batch, Seq_Len=107, Feats=28)
    assert (
        X_train.shape[1] == 107
    ), f"Expected sequence length 107, got {X_train.shape[1]}"
    assert X_train.shape[2] == 28, f"Expected feature dim 28, got {X_train.shape[2]}"
    # y: (Batch, Seq_Scored=68, Targets=5)
    assert (
        y_train.shape[1] == 68
    ), f"Expected scored sequence length 68, got {y_train.shape[1]}"
    assert y_train.shape[2] == 5, f"Expected 5 targets, got {y_train.shape[2]}"
    # w: (Batch, Seq_Scored=68, Targets=5)
    assert w_train.shape == y_train.shape, "Weights shape mismatch"

    # Verify Test Batch
    X_test, ids_test = next(iter(test_loader))
    print(f"Test Batch - X shape: {X_test.shape}")
    print(f"Test Batch - IDs length: {len(ids_test)}")

    assert X_test.shape[1] == 107, "Test sequence length mismatch"
    assert len(ids_test) == X_test.shape[0], "Test IDs count mismatch"

    # ---------------------------------------------------------
    # 3. Model Verification
    # ---------------------------------------------------------
    print("\n[3] Initializing model and verifying forward pass...")

    device = torch.device(
        "cpu"
    )  # Use CPU for simple verification to avoid GPU overhead if busy
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA.")

    model = SpatiallyAugmentedBiGRU().to(device)

    # Move dummy batch to device
    X_train_dev = X_train.to(device)

    # Forward pass
    preds = model(X_train_dev)
    print(f"Model Output Shape: {preds.shape}")

    # Assert Output Shape: (Batch, 107, 5)
    # Note: Model outputs full sequence length (107), targets are only 68.
    assert preds.shape == (
        X_train.shape[0],
        107,
        5,
    ), f"Expected output (B, 107, 5), got {preds.shape}"

    # ---------------------------------------------------------
    # 4. Loss Function Verification
    # ---------------------------------------------------------
    print("\n[4] Verifying Loss Function...")

    criterion = WeightedMCRMSELoss()
    y_train_dev = y_train.to(device)
    w_train_dev = w_train.to(device)

    loss = criterion(preds, y_train_dev, w_train_dev)
    print(f"Calculated Loss: {loss.item()}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    # ---------------------------------------------------------
    # 5. Training Loop Execution
    # ---------------------------------------------------------
    print("\n[5] Running Training Loop...")

    val_loader = loaders["val"]
    trainer = Trainer(model, train_loader, val_loader, device)

    # Run fit
    trainer.fit(epochs=Config.EPOCHS)

    # Verify model artifact
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Best model file was not saved."
    print("Training completed successfully.")

    # ---------------------------------------------------------
    # 6. Inference & Submission Generation
    # ---------------------------------------------------------
    print("\n[6] Running Inference and Generating Submission...")

    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for X_batch, ids_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)  # (B, 107, 5)

            all_preds.append(outputs.cpu().numpy())
            all_ids.extend(ids_batch)

    all_preds = np.concatenate(all_preds, axis=0)

    print(f"Total Predictions Shape: {all_preds.shape}")
    print(f"Total IDs: {len(all_ids)}")

    # Generate Submission
    save_submission(all_preds, all_ids, save_path=Config.SUBMISSION_PATH)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission DataFrame Shape: {df_sub.shape}")

    # Expected rows: N_test_samples * 107
    # N_test_samples is 240 in full set, but we handle whatever is in test_loader.
    # The test loader loads from metadata/test.parquet which has 240 rows.
    # Since we didn't subset test data in Config (only train), it should be 240 * 107 = 25680.
    # However, if the test set in metadata is small, we check logic.
    expected_rows = len(all_ids) * 107
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows, got {len(df_sub)}"

    # Check columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert list(df_sub.columns) == expected_cols, "Submission columns mismatch"

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    main()
