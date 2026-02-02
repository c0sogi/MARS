import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, MCRMSE
from library.data import get_dataloaders, RNADataset
from library.model import HC_WG_BiGRU
from library.train import MCRMSELoss, run_training


def setup_demo_config():
    """
    Overrides the default configuration for a fast demonstration run.
    """
    print("Setting up demo configuration...")

    # Use a separate working directory for the demo to avoid conflicts
    Config.WORKING_DIR = "./working/demo_run"
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update paths based on new working dir
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    # Reduce data size and training duration for speed
    Config.DEBUG = True
    Config.SUBSET_SIZE = 20  # Use only 20 samples
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 2  # Train for just 2 epochs
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure reproducibility
    seed_everything(Config.SEED)
    print(f"Demo Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}, Subset Size: {Config.SUBSET_SIZE}")


def verify_data_loading():
    """
    Demonstrates data loading and verifies batch structure.
    """
    print("\n=== Verifying Data Loading ===")

    # Load dataloaders
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify keys
    expected_keys = {"features", "pair_indices", "pair_mask", "targets"}
    assert expected_keys.issubset(
        batch.keys()
    ), f"Batch missing keys. Found: {batch.keys()}"

    # Verify shapes
    # Features: (Batch, Seq_Len, Input_Dim) -> (4, 107, 14)
    features = batch["features"]
    assert features.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.INPUT_DIM,
    ), f"Incorrect features shape: {features.shape}"

    # Targets: (Batch, Seq_Len, Num_Targets) -> (4, 107, 5)
    targets = batch["targets"]
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.NUM_TARGETS,
    ), f"Incorrect targets shape: {targets.shape}"

    # Pair Indices: (Batch, Seq_Len)
    pair_indices = batch["pair_indices"]
    assert pair_indices.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
    ), f"Incorrect pair_indices shape: {pair_indices.shape}"

    print("Data loading verification passed. Batch shapes are correct.")
    return train_loader, val_loader


def verify_model_forward(train_loader):
    """
    Demonstrates model instantiation and forward pass.
    """
    print("\n=== Verifying Model Forward Pass ===")

    device = Config.DEVICE
    model = HC_WG_BiGRU().to(device)
    model.eval()

    # Get a batch
    batch = next(iter(train_loader))
    features = batch["features"].to(device)
    pair_indices = batch["pair_indices"].to(device)
    pair_mask = batch["pair_mask"].to(device)

    # Forward pass
    with torch.no_grad():
        output = model(features, pair_indices, pair_mask)

    # Verify Output Shape: (Batch, Seq_Len, Num_Targets)
    expected_shape = (Config.BATCH_SIZE, Config.SEQ_LEN, Config.NUM_TARGETS)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    # Verify Finite Values
    assert torch.isfinite(output).all(), "Model output contains NaNs or Infs."

    print("Model forward pass verification passed.")
    return model


def verify_loss_calculation(model, train_loader):
    """
    Demonstrates loss calculation using MCRMSELoss.
    """
    print("\n=== Verifying Loss Calculation ===")

    device = Config.DEVICE
    criterion = MCRMSELoss()

    batch = next(iter(train_loader))
    features = batch["features"].to(device)
    pair_indices = batch["pair_indices"].to(device)
    pair_mask = batch["pair_mask"].to(device)
    targets = batch["targets"].to(device)

    # Forward
    with torch.no_grad():
        preds = model(features, pair_indices, pair_mask)

    # Calculate Loss
    loss = criterion(preds, targets)

    # Verify Loss
    assert loss.dim() == 0, "Loss should be a scalar."
    assert loss.item() >= 0, "Loss should be non-negative."

    print(f"Loss verification passed. Calculated Loss: {loss.item():.4f}")


def verify_full_pipeline():
    """
    Runs the full training pipeline provided in library.train.
    """
    print("\n=== Running Full Training Pipeline (Demo) ===")

    # run_training handles loading, training loop, validation, and submission generation
    run_training()

    # Verify artifacts exist
    assert os.path.exists(Config.MODEL_PATH), "Model file was not saved."
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not saved."

    print("Training pipeline completed successfully.")


def verify_submission_file():
    """
    Verifies the content and format of the generated submission file.
    """
    print("\n=== Verifying Submission File ===")

    df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check Columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(df.columns) == expected_cols
    ), f"Submission columns mismatch. Got: {list(df.columns)}"

    # Check Row Count
    # In Debug mode, we used Config.SUBSET_SIZE samples for test as well.
    # Total rows = SUBSET_SIZE * SEQ_LEN
    expected_rows = Config.SUBSET_SIZE * Config.SEQ_LEN
    assert (
        len(df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df)}"

    # Check ID format
    sample_id_seqpos = df.iloc[0]["id_seqpos"]
    assert "id_" in sample_id_seqpos, "id_seqpos format seems incorrect."

    print("Submission file verification passed.")


if __name__ == "__main__":
    # 1. Setup Demo Configuration
    setup_demo_config()

    # 2. Verify Data Loading
    train_loader, _ = verify_data_loading()

    # 3. Verify Model
    model = verify_model_forward(train_loader)

    # 4. Verify Loss
    verify_loss_calculation(model, train_loader)

    # 5. Run Full Pipeline (Train -> Val -> Predict)
    verify_full_pipeline()

    # 6. Verify Submission
    verify_submission_file()

    print("\nAll demonstrations and verifications completed successfully.")
