import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import Config
from library.data_loader import EEGDataset, get_dataloaders
from library.model import EEGNet1D
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_demo():
    print("=== Starting EEG Classification Task Demo ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Modify Config attributes directly to affect all modules
    Config.EPOCHS = 2  # Train for only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples
    Config.NUM_WORKERS = 2  # Reduce workers for simple script
    Config.PATIENCE = 1  # Early stopping quickly

    # Ensure output directories are clean/ready
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    set_seed(Config.SEED)
    print(
        f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.BATCH_SIZE}, Subset={Config.DEBUG_SUBSET_SIZE}"
    )

    # -------------------------------------------------------------------------
    # 2. Data Loader Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Data Loading components...")

    # Test Dataset instantiation
    try:
        dataset = EEGDataset(metadata_path=Config.TRAIN_CSV, mode="train", debug=True)
        print(f"Dataset initialized successfully. Size (Debug): {len(dataset)}")

        # Fetch one sample
        data, target = dataset[0]

        # Verify Shapes
        # Expected Data Shape: (Channels, Time) -> (20, 5000)
        # Expected Target Shape: (Num_Classes,) -> (6,)
        expected_shape = (Config.NUM_CHANNELS, Config.FIXED_LENGTH)
        assert (
            data.shape == expected_shape
        ), f"Data shape mismatch. Got {data.shape}, expected {expected_shape}"
        assert target.shape == (
            Config.NUM_CLASSES,
        ), f"Target shape mismatch. Got {target.shape}"

        print(f"Sample verification passed: Data {data.shape}, Target {target.shape}")

    except Exception as e:
        print(f"Data Loading failed: {e}")
        raise e

    # Test DataLoader
    train_loader, val_loader, test_loader = get_dataloaders(
        train_batch_size=Config.BATCH_SIZE, val_batch_size=Config.BATCH_SIZE, debug=True
    )

    batch_data, batch_target = next(iter(train_loader))
    assert batch_data.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CHANNELS,
        Config.FIXED_LENGTH,
    )
    print(f"DataLoader verification passed. Batch shape: {batch_data.shape}")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = EEGNet1D(config=Config).to(device)

    # Create dummy input
    dummy_input = torch.randn(2, Config.NUM_CHANNELS, Config.FIXED_LENGTH).to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    # Verify Output
    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Got {output.shape}"

    # Verify Probabilities sum to 1
    sums = output.sum(dim=1).cpu().numpy()
    assert np.allclose(sums, 1.0, atol=1e-5), f"Probabilities do not sum to 1: {sums}"

    print("Model forward pass verification passed.")

    # -------------------------------------------------------------------------
    # 4. Training & Inference Pipeline
    # -------------------------------------------------------------------------
    print("\n[4] Running Training Pipeline (Trainer.fit)...")

    trainer = Trainer(config=Config)

    # Run fit (includes training, validation, saving model, and predicting on test)
    # This uses the debug=True flag to use the subset of data defined in Config
    trainer.fit(debug=True)

    print("Training pipeline execution completed.")

    # -------------------------------------------------------------------------
    # 5. Output Validation
    # -------------------------------------------------------------------------
    print("\n[5] Validating Submission Output...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check shape
    # In debug mode, test loader also loads a subset (Config.DEBUG_SUBSET_SIZE)
    # However, depending on sampling, it might be slightly less if file loading fails,
    # but usually it matches exactly or is limited by available files.
    print(f"Submission shape: {submission_df.shape}")

    # Check columns
    expected_cols = ["eeg_id"] + Config.OUTPUT_COLS
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Column mismatch. Got {submission_df.columns}"

    # Check values are probabilities
    vote_cols = Config.OUTPUT_COLS
    probs = submission_df[vote_cols].values

    # Check range [0, 1]
    assert (probs >= 0).all() and (
        probs <= 1.00001
    ).all(), "Probabilities out of range [0, 1]"

    # Check sum to 1
    row_sums = probs.sum(axis=1)
    # Allow small float tolerance
    assert np.allclose(row_sums, 1.0, atol=1e-4), "Submission rows do not sum to 1.0"

    print("Submission file validation passed.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
