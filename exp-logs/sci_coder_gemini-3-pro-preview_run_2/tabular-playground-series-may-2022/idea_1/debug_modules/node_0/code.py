import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_dataloaders
from library.model import ShallowEmbeddingMLP
from library.trainer import run_training
from library.inference import run_inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("--- Starting Demonstration Script ---")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    # Modify Config for a fast demonstration (Debug Mode) to ensure quick execution
    print("1. Configuring environment for fast demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 2000  # Use only 2000 samples for this demo
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 256  # Set a reasonable batch size

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Ensure working directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    print("\n2. Loading Data (Debug Mode)...")
    # get_dataloaders handles preprocessing and caching.
    # debug=True forces it to load a subset defined by Config.DEBUG_SUBSET_SIZE
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=2,  # Reduced workers for this lightweight script
        debug=True,
    )

    # Verification: Check dataset sizes
    print("   Verifying dataset sizes...")
    expected_size = Config.DEBUG_SUBSET_SIZE

    assert (
        len(train_loader.dataset) == expected_size
    ), f"Train dataset size mismatch. Expected {expected_size}, got {len(train_loader.dataset)}"
    assert (
        len(val_loader.dataset) == expected_size
    ), f"Val dataset size mismatch. Expected {expected_size}, got {len(val_loader.dataset)}"
    assert (
        len(test_loader.dataset) == expected_size
    ), f"Test dataset size mismatch. Expected {expected_size}, got {len(test_loader.dataset)}"
    print("   Dataset sizes verified.")

    # Verification: Check batch shapes
    print("   Verifying batch shapes...")
    sample_batch = next(iter(train_loader))
    cont = sample_batch["continuous"]
    cat = sample_batch["categorical"]
    target = sample_batch["target"]

    # Continuous: (Batch, 30 features)
    assert cont.shape == (
        Config.BATCH_SIZE,
        30,
    ), f"Continuous shape mismatch: {cont.shape}"
    # Categorical: (Batch, 10 sequence length)
    assert cat.shape == (
        Config.BATCH_SIZE,
        10,
    ), f"Categorical shape mismatch: {cat.shape}"
    # Target: (Batch, 1)
    assert target.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Target shape mismatch: {target.shape}"
    print("   Batch shapes verified.")

    # --------------------------------------------------------------------------
    # 3. Model Initialization
    # --------------------------------------------------------------------------
    print("\n3. Initializing Model...")
    model = ShallowEmbeddingMLP()
    model.to(Config.DEVICE)

    # Verification: Forward pass
    print("   Verifying forward pass...")
    with torch.no_grad():
        # Move sample batch to device
        cont_dev = cont.to(Config.DEVICE)
        cat_dev = cat.to(Config.DEVICE)
        output = model(cont_dev, cat_dev)

    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Output shape mismatch: {output.shape}"
    assert (
        output.min() >= 0 and output.max() <= 1
    ), "Output probabilities out of range [0, 1]"
    print("   Forward pass verified.")

    # --------------------------------------------------------------------------
    # 4. Training Loop
    # --------------------------------------------------------------------------
    print("\n4. Running Training Loop...")
    # run_training handles the loop, validation, early stopping, and saving best_model.pth
    trained_model = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.EPOCHS,
        device=Config.DEVICE,
    )

    # Verification: Check if model checkpoint was created
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(
        checkpoint_path
    ), "Model checkpoint best_model.pth was not created."
    print(f"   Training complete. Checkpoint saved at {checkpoint_path}")

    # --------------------------------------------------------------------------
    # 5. Inference
    # --------------------------------------------------------------------------
    print("\n5. Running Inference...")
    # run_inference loads the best model weights and generates submission.csv
    # We must pass debug=True so it loads the corresponding test metadata subset
    submission_df = run_inference(
        checkpoint_path=checkpoint_path,
        device=Config.DEVICE,
        batch_size=Config.BATCH_SIZE,
        num_workers=2,
        debug=True,
    )

    # Verification: Submission file existence
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created."

    # Verification: Content
    print("   Verifying submission file content...")
    df_check = pd.read_csv(submission_path)

    # Check row count
    assert (
        len(df_check) == expected_size
    ), f"Submission rows mismatch. Expected {expected_size}, got {len(df_check)}"

    # Check columns
    assert list(df_check.columns) == [
        "id",
        "target",
    ], f"Submission columns mismatch. Expected ['id', 'target'], got {list(df_check.columns)}"

    # Check value ranges
    assert (
        df_check["target"].min() >= 0.0 and df_check["target"].max() <= 1.0
    ), "Submission target values out of probability range."

    print("   Submission verified.")

    print("\n--- Demonstration Complete Successfully ---")


if __name__ == "__main__":
    main()
