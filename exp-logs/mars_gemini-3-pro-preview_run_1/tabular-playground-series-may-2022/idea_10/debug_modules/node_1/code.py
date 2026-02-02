import os
import sys
import torch
import pandas as pd
import numpy as np

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config, set_seed
from library.data_utils import get_dataloaders
from library.model_utils import SSDeGUT
from library.train_utils import Trainer


def main():
    # ------------------------------------------------------------------------
    # 1. Configuration Setup
    # ------------------------------------------------------------------------
    print("Setting up configuration for demonstration...")

    # We modify the Config class attributes directly because library functions
    # access Config class attributes statically.

    # Set specific paths for this execution to isolate outputs
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Enable Debug mode for speed (uses a small subset of data)
    Config.DEBUG = True
    Config.DEBUG_SAMPLES = 1000

    # Reduce Model complexity for rapid execution
    Config.D_MODEL = 32
    Config.N_LAYERS = 2
    Config.N_HEADS = 2
    Config.D_FF = 64

    # Optimize training hyperparameters for speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 128
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Initialize directories and seed
    Config.setup()
    set_seed(Config.SEED)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # ------------------------------------------------------------------------
    # 2. Data Loading
    # ------------------------------------------------------------------------
    print("\n--- [Step 1] Loading Data ---")
    # get_dataloaders handles preprocessing, caching, and loader creation
    train_loader, val_loader, test_loader = get_dataloaders(Config)

    # Verification: Check batch structure
    print("Verifying DataLoaders...")
    try:
        batch = next(iter(train_loader))
        x_num = batch["x_num"]
        x_seq = batch["x_seq"]
        target = batch["target"]

        # Verify dimensions
        assert x_num.dim() == 2, f"Expected x_num to be 2D, got {x_num.dim()}"
        assert x_seq.dim() == 2, f"Expected x_seq to be 2D, got {x_seq.dim()}"
        assert target.dim() == 1, f"Expected target to be 1D, got {target.dim()}"

        num_features = x_num.shape[1]
        print(
            f"Batch verified. Batch Size: {x_num.shape[0]}, Num Features: {num_features}"
        )
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # ------------------------------------------------------------------------
    # 3. Model Initialization
    # ------------------------------------------------------------------------
    print("\n--- [Step 2] Initializing Model ---")
    model = SSDeGUT(Config, num_numerical_features=num_features)
    model.to(Config.DEVICE)

    # Verification: Run a forward pass
    print("Verifying Forward Pass...")
    with torch.no_grad():
        # Move batch to device
        x_num_dev = x_num.to(Config.DEVICE)
        x_seq_dev = x_seq.to(Config.DEVICE)

        # Forward pass with masking enabled (training mode simulation)
        outputs = model(x_num_dev, x_seq_dev, mask_ratio=0.15)

        # Check output keys required by the loss function
        assert "logits" in outputs
        assert "recon_num" in outputs
        assert "recon_seq" in outputs
        assert "mask" in outputs

        # Check output shapes
        assert outputs["logits"].shape == (x_num.shape[0], 1)

    print("Forward pass successful.")

    # ------------------------------------------------------------------------
    # 4. Training
    # ------------------------------------------------------------------------
    print("\n--- [Step 3] Training Loop ---")
    trainer = Trainer(model, train_loader, val_loader, test_loader, Config)

    # Run training loop
    trainer.fit()

    # Verify model artifact creation
    if not os.path.exists(Config.MODEL_PATH):
        raise AssertionError(f"Model file not found at {Config.MODEL_PATH}")
    print("Training complete and best model saved.")

    # ------------------------------------------------------------------------
    # 5. Prediction
    # ------------------------------------------------------------------------
    print("\n--- [Step 4] Prediction ---")
    trainer.predict()

    # Verify submission artifact creation
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise AssertionError(f"Submission file not found at {Config.SUBMISSION_PATH}")

    # Verify submission content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission Preview:")
    print(df_sub.head())

    # Check columns
    assert "id" in df_sub.columns
    assert "target" in df_sub.columns

    # Check row count (should match DEBUG_SAMPLES in this demo)
    assert (
        len(df_sub) == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} predictions in debug mode, got {len(df_sub)}"

    # Check for missing values
    assert not df_sub.isnull().values.any(), "Submission contains NaN values."

    print("\n--- Success! Pipeline demonstration finished. ---")


if __name__ == "__main__":
    main()
