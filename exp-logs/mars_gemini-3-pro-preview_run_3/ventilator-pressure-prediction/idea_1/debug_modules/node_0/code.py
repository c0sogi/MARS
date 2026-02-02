import os
import torch
import pandas as pd
import numpy as np
import sys

# Add the current directory to sys.path to ensure library imports work correctly
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import PhysicsGRU
from library.trainer import Trainer


def main():
    print("=== Starting Demonstration of Ventilator Pressure Prediction Pipeline ===\n")

    # ------------------------------------------------------------------------
    # 1. Configuration & Setup
    # ------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Modify Config for a fast, debug run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200  # Use 200 breaths for quick processing
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 32  # Small batch size
    Config.HIDDEN_DIM = 64  # Smaller model for speed
    Config.WORKING_DIR = "./working/demo"  # Isolate demo outputs
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "model.pth")

    # Update cache paths to point to the demo working directory
    # This prevents reading potential full-dataset caches from previous runs
    Config.TRAIN_CACHE_DATA = os.path.join(Config.WORKING_DIR, "train_data.npy")
    Config.TRAIN_CACHE_TARGET = os.path.join(Config.WORKING_DIR, "train_targets.npy")
    Config.VAL_CACHE_DATA = os.path.join(Config.WORKING_DIR, "val_data.npy")
    Config.VAL_CACHE_TARGET = os.path.join(Config.WORKING_DIR, "val_targets.npy")
    Config.TEST_CACHE_DATA = os.path.join(Config.WORKING_DIR, "test_data.npy")
    Config.STATS_CACHE = os.path.join(Config.WORKING_DIR, "stats.npy")

    # Create the working directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print("    Configuration complete. Debug mode enabled.")

    # ------------------------------------------------------------------------
    # 2. Data Loading
    # ------------------------------------------------------------------------
    print("\n[2] Loading and processing data...")

    # get_dataloaders handles loading, feature engineering, normalization, and caching
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verification: Check Data Shapes
    print("    Verifying data shapes...")
    sample_x, sample_y = next(iter(train_loader))

    # Expected shape: (Batch, Seq_Len, Features)
    expected_x_shape = (Config.BATCH_SIZE, Config.SEQ_LEN, len(Config.FEATURE_COLS))
    # Expected target shape: (Batch, Seq_Len)
    expected_y_shape = (Config.BATCH_SIZE, Config.SEQ_LEN)

    # Handle last batch potentially being smaller
    current_batch_size = sample_x.size(0)

    assert sample_x.dim() == 3, f"Input tensor should be 3D, got {sample_x.dim()}"
    assert (
        sample_x.size(1) == Config.SEQ_LEN
    ), f"Sequence length mismatch. Expected {Config.SEQ_LEN}, got {sample_x.size(1)}"
    assert sample_x.size(2) == len(
        Config.FEATURE_COLS
    ), f"Feature dim mismatch. Expected {len(Config.FEATURE_COLS)}, got {sample_x.size(2)}"
    assert sample_y.size() == (
        current_batch_size,
        Config.SEQ_LEN,
    ), f"Target shape mismatch. Got {sample_y.size()}"

    print(
        f"    Data verification passed. Input shape: {sample_x.shape}, Target shape: {sample_y.shape}"
    )

    # ------------------------------------------------------------------------
    # 3. Model Initialization
    # ------------------------------------------------------------------------
    print("\n[3] Initializing PhysicsGRU model...")

    device = Config.DEVICE
    model = PhysicsGRU(
        input_dim=len(Config.FEATURE_COLS),
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        bidirectional=Config.BIDIRECTIONAL,
        dropout=Config.DROPOUT,
    ).to(device)

    # Verification: Dummy Forward Pass
    print("    Verifying model forward pass...")
    with torch.no_grad():
        dummy_out = model(sample_x.to(device))

    assert dummy_out.shape == (
        current_batch_size,
        Config.SEQ_LEN,
    ), f"Model output shape mismatch. Expected {(current_batch_size, Config.SEQ_LEN)}, got {dummy_out.shape}"

    print("    Model verification passed.")

    # ------------------------------------------------------------------------
    # 4. Training
    # ------------------------------------------------------------------------
    print("\n[4] Starting training loop...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    trainer = Trainer(model, optimizer, device)

    # Run training
    trainer.fit(train_loader, val_loader)

    # Verification: Check if model file exists
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file was not saved at {Config.MODEL_SAVE_PATH}")

    print(f"    Training complete. Model saved to {Config.MODEL_SAVE_PATH}")

    # ------------------------------------------------------------------------
    # 5. Inference & Submission Generation
    # ------------------------------------------------------------------------
    print("\n[5] Generating predictions on test set...")

    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    predictions = []

    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            preds = model(x)
            # Flatten predictions because submission format is 1 row per time step
            predictions.extend(preds.cpu().numpy().flatten())

    predictions = np.array(predictions)

    # In debug mode, we only processed a subset of breaths.
    # We need to match these predictions to the corresponding IDs in the test set.
    # We reload the test metadata, filter by the debug breath IDs, and align.

    test_meta = pd.read_csv(Config.TEST_PATH)
    if Config.DEBUG:
        # Re-apply the same filtering logic used in dataset.py
        unique_breaths = test_meta["breath_id"].unique()[: Config.DEBUG_SAMPLE_SIZE]
        test_meta = test_meta[test_meta["breath_id"].isin(unique_breaths)].copy()

    # Verification: Ensure number of predictions matches number of rows in filtered test set
    assert len(predictions) == len(
        test_meta
    ), f"Mismatch: {len(predictions)} predictions vs {len(test_meta)} test rows."

    # Create submission dataframe
    submission = pd.DataFrame({"id": test_meta["id"], "pressure": predictions})

    # Save submission
    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)

    print(f"    Submission saved to {submission_path}")
    print(f"    Submission shape: {submission.shape}")
    print(f"    First 5 rows:\n{submission.head().to_string()}")

    print("\n=== Demonstration Complete Successfully ===")


if __name__ == "__main__":
    main()
