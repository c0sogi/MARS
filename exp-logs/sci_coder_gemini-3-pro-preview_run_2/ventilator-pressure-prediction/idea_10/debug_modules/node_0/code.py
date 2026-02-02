import os
import shutil
import torch
import pandas as pd
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, WeightedL1Loss
from library.dataset import load_and_preprocess_data
from library.model import RGIBiLSTM
from library.train import train_one_epoch, validate_one_epoch
from library.inference import predict


def main():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print(">>> 1. Configuring environment for demonstration...")

    # Override Config for speed and debugging
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # 50 breaths for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.NUM_WORKERS = 0  # Disable multiprocessing for small data

    # Define a custom working directory for this execution
    Config.WORKING_DIR = "./working/demo_execution_script"
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update derived paths in Config
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_processed.parquet")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_processed.parquet")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_processed.parquet")
    Config.MODEL_CHECKPOINT = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Set seed
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # ==========================================
    # 2. Prepare Debug Data (Patching Test CSV)
    # ==========================================
    # The inference pipeline reads Config.TEST_CSV directly for ID alignment.
    # In DEBUG mode, we must ensure Config.TEST_CSV contains only the breaths used in the debug subset.
    print("\n>>> 2. Preparing debug datasets...")

    # Read Test Metadata to identify debug breaths
    test_meta = pd.read_csv(Config.TEST_METADATA)
    unique_breaths = test_meta[Config.BREATH_ID_COL].unique()
    debug_breaths = unique_breaths[: Config.DEBUG_SAMPLE_SIZE]

    # Create a subset of the raw test CSV
    full_test_csv_path = os.path.join(Config.INPUT_DIR, "test.csv")
    df_test_full = pd.read_csv(full_test_csv_path)
    df_test_debug = df_test_full[
        df_test_full[Config.BREATH_ID_COL].isin(debug_breaths)
    ].copy()

    # Save to working dir and update Config
    debug_test_csv_path = os.path.join(Config.WORKING_DIR, "test_debug.csv")
    df_test_debug.to_csv(debug_test_csv_path, index=False)
    Config.TEST_CSV = debug_test_csv_path
    print(
        f"    Created debug test CSV at {Config.TEST_CSV} with {len(df_test_debug)} rows."
    )

    # ==========================================
    # 3. Data Loading & Preprocessing
    # ==========================================
    print("\n>>> 3. Loading and preprocessing data...")
    # Force reprocessing (load_cached_data=False) to ensure we use our debug settings
    train_ds, val_ds, test_ds = load_and_preprocess_data(load_cached_data=False)

    # Validation
    print("    Verifying dataset integrity...")
    assert len(train_ds) > 0, "Training dataset is empty."
    assert len(val_ds) > 0, "Validation dataset is empty."
    # Check shape of a single sample
    sample = train_ds[0]
    # Expected shape: (Breath_Len, Features) -> (80, 12)
    assert sample["X"].shape == (
        80,
        Config.INPUT_DIM,
    ), f"Feature shape mismatch. Expected (80, {Config.INPUT_DIM}), got {sample['X'].shape}"
    assert sample["u_out"].shape == (80,), "u_out shape mismatch."
    assert sample["y"].shape == (80,), "Target shape mismatch."

    # Create Loaders
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False)

    print(f"    Train size: {len(train_ds)} breaths")
    print(f"    Val size: {len(val_ds)} breaths")

    # ==========================================
    # 4. Model Initialization
    # ==========================================
    print("\n>>> 4. Initializing Model...")
    model = RGIBiLSTM().to(device)

    # Validation: Forward pass
    print("    Verifying model forward pass...")
    dummy_x = torch.randn(2, 80, Config.INPUT_DIM).to(device)
    with torch.no_grad():
        dummy_out = model(dummy_x)

    assert dummy_out.shape == (
        2,
        80,
    ), f"Model output shape mismatch. Expected (2, 80), got {dummy_out.shape}"
    print("    Model initialized and verified successfully.")

    # ==========================================
    # 5. Training Loop
    # ==========================================
    print("\n>>> 5. Running Training Loop...")
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = WeightedL1Loss()

    # Baseline validation
    val_mae_pre = validate_one_epoch(model, val_loader, device)
    print(f"    Pre-train Val MAE: {val_mae_pre:.4f}")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, Config.MAX_GRAD_NORM
        )
        val_mae = validate_one_epoch(model, val_loader, device)
        print(
            f"    Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f}"
        )

        # Simple assertion to ensure loss is valid
        assert not np.isnan(train_loss), "Training loss is NaN!"

    # Save Model
    print(f"    Saving model to {Config.MODEL_CHECKPOINT}...")
    torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)
    assert os.path.exists(
        Config.MODEL_CHECKPOINT
    ), "Model checkpoint file was not created."

    # ==========================================
    # 6. Inference
    # ==========================================
    print("\n>>> 6. Running Inference...")
    # Run predict function (it will reload data and model internally)
    predict(batch_size=Config.BATCH_SIZE, num_workers=0)

    # Validation
    print("    Verifying submission file...")
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"    Submission shape: {df_sub.shape}")

    # Check columns
    assert list(df_sub.columns) == [
        "id",
        "pressure",
    ], "Submission columns are incorrect."

    # Check row count matches the debug test set
    # Note: df_test_debug was created in step 2
    expected_rows = len(df_test_debug)
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}."

    # Check for NaN values
    assert not df_sub["pressure"].isnull().any(), "Submission contains NaN values."

    print("\n>>> Demonstration completed successfully!")


if __name__ == "__main__":
    main()
