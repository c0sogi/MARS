import os
import shutil
import numpy as np
import pandas as pd
import torch
import warnings

# Import provided library modules
from library.config import Config
from library.utils import set_seed, MCRMSELoss
from library.dataset import get_dataloaders, RNADataset
from library.model import RNAResNet
from library.trainer import Trainer, run_training

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def verify_data_loading():
    """
    Demonstrates and verifies data loading, processing, and batching.
    """
    print("\n=== Verifying Data Loading ===")

    # Use a small subset for verification
    batch_size = 4
    max_samples = 20

    # Force reload to test processing logic (load_cached_data=False)
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=0,  # Use 0 workers for simple debugging/demo
        load_cached_data=False,
        max_samples=max_samples,
    )

    # Fetch one batch from training loader
    inputs, targets = next(iter(train_loader))

    # Verify Shapes
    # Inputs: (Batch, Seq_Len, Channels) -> (4, 107, 14)
    # Channels = 4 (Seq) + 3 (Struct) + 7 (Loop) = 14
    expected_input_shape = (batch_size, Config.SEQ_LEN, Config.INPUT_CHANNELS)
    assert (
        inputs.shape == expected_input_shape
    ), f"Input shape mismatch. Expected {expected_input_shape}, got {inputs.shape}"

    # Targets: (Batch, Seq_Len, Num_Targets) -> (4, 107, 5)
    expected_target_shape = (batch_size, Config.SEQ_LEN, Config.NUM_TARGETS)
    assert (
        targets.shape == expected_target_shape
    ), f"Target shape mismatch. Expected {expected_target_shape}, got {targets.shape}"

    print(
        f"Data Loading Verified. Input shape: {inputs.shape}, Target shape: {targets.shape}"
    )
    return train_loader, val_loader, test_loader


def verify_model_architecture():
    """
    Demonstrates model instantiation and verifies forward pass dimensions.
    """
    print("\n=== Verifying Model Architecture ===")

    model = RNAResNet(
        input_channels=Config.INPUT_CHANNELS,
        num_targets=Config.NUM_TARGETS,
        hidden_dim=32,  # Reduced for speed in demo
        kernel_size=3,
        dropout=0.0,
        dilations=[1, 2],  # Reduced complexity
    )

    # Create dummy input: (Batch=2, Seq_Len=107, Channels=14)
    dummy_input = torch.randn(2, Config.SEQ_LEN, Config.INPUT_CHANNELS)

    # Forward pass
    output = model(dummy_input)

    # Verify Output Shape: (Batch, Seq_Len, Num_Targets) -> (2, 107, 5)
    expected_shape = (2, Config.SEQ_LEN, Config.NUM_TARGETS)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print(f"Model Architecture Verified. Output shape: {output.shape}")
    return model


def verify_loss_function():
    """
    Verifies the MCRMSELoss logic with manual calculation.
    """
    print("\n=== Verifying MCRMSE Loss Function ===")

    criterion = MCRMSELoss()
    scored_len = Config.SCORED_LEN  # 68

    # Create synthetic data
    # Batch=1, Seq_Len=107, Targets=5
    # We will set errors only in the scored region (first 68)

    # Case 1: Perfect prediction
    preds_perfect = torch.ones(1, 107, 5)
    targets_perfect = torch.ones(1, 107, 5)
    loss_perfect = criterion(preds_perfect, targets_perfect)
    assert torch.isclose(
        loss_perfect, torch.tensor(0.0)
    ), "Loss should be 0 for perfect predictions"

    # Case 2: Constant error
    # Preds = 1.0, Targets = 0.0
    # Squared Error = 1.0
    # Mean Squared Error = 1.0
    # RMSE = 1.0
    # Mean of RMSEs = 1.0
    preds_err = torch.ones(1, 107, 5)
    targets_err = torch.zeros(1, 107, 5)
    loss_err = criterion(preds_err, targets_err)
    assert torch.isclose(
        loss_err, torch.tensor(1.0)
    ), f"Loss should be 1.0, got {loss_err.item()}"

    # Case 3: Error only in non-scored region (indices >= 68)
    # Should result in 0 loss because metric ignores > 68
    preds_ignore = torch.zeros(1, 107, 5)
    targets_ignore = torch.zeros(1, 107, 5)

    # Introduce error at index 70
    preds_ignore[:, 70, :] = 100.0

    loss_ignore = criterion(preds_ignore, targets_ignore)
    assert torch.isclose(
        loss_ignore, torch.tensor(0.0)
    ), "Loss should ignore positions > seq_scored"

    print("MCRMSE Loss Logic Verified.")


def verify_submission_file(submission_path):
    """
    Verifies the generated submission file format.
    """
    print("\n=== Verifying Submission File ===")

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df = pd.read_csv(submission_path)

    # Check columns
    expected_cols = ["id_seqpos"] + Config.TARGET_COLS
    assert (
        list(df.columns) == expected_cols
    ), f"Column mismatch. Expected {expected_cols}, got {list(df.columns)}"

    # Check if file is not empty
    assert len(df) > 0, "Submission file is empty."

    # Check format of id_seqpos
    sample_id = df.iloc[0]["id_seqpos"]
    assert "_0" in sample_id or "_" in sample_id, "id_seqpos format seems incorrect."

    print(f"Submission File Verified. Shape: {df.shape}")


def main():
    # 1. Setup
    set_seed(42)

    # Clean up working directory to ensure fresh run
    if os.path.exists(Config.CACHE_DIR):
        shutil.rmtree(Config.CACHE_DIR)

    # 2. Verify Components
    verify_data_loading()
    verify_model_architecture()
    verify_loss_function()

    # 3. Run Full Training Pipeline (Demo)
    print("\n=== Running Full Training Pipeline (Demo) ===")
    # We use a small subset (max_samples=100) and few epochs for speed
    run_training(
        max_samples=100,
        epochs=2,
        batch_size=16,
        load_cached_data=False,  # Force processing
    )

    # 4. Verify Output
    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    verify_submission_file(submission_path)

    print("\nSUCCESS: All demonstrations and verifications passed.")


if __name__ == "__main__":
    main()
