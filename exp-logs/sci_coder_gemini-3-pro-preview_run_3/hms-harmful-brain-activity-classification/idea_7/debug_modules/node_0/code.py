import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
import logging
import warnings

# Ensure the current directory is in the path to import library modules
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_logger
from library.data_loader import get_dataloaders
from library.model import DualScaleSpectrogramNet
from library.engine import train_one_epoch, validate, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


def run_demo():
    print("=" * 50)
    print("Starting End-to-End Pipeline Demonstration")
    print("=" * 50)

    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[Step 1] Configuring environment...")

    # Override Config for a fast demo run
    Config.DEBUG = True
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 8  # Small batch size for demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_FILE = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Create working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds
    seed_everything(Config.SEED)

    # Setup Logger
    logger = get_logger(os.path.join(Config.WORKING_DIR, "demo.log"))
    logger.info(f"Working Directory: {Config.WORKING_DIR}")
    logger.info(f"Device: {Config.DEVICE}")

    # ==========================================
    # 2. Data Loading (with Subset)
    # ==========================================
    print("\n[Step 2] Loading and preprocessing data...")

    # Load full metadata
    train_full = pd.read_csv(Config.TRAIN_CSV)
    val_full = pd.read_csv(Config.VAL_CSV)
    test_full = pd.read_csv(Config.TEST_CSV)

    # Slice for demo (use top N rows)
    # Ensure we have enough for a few batches
    N_TRAIN = 32
    N_VAL = 16
    N_TEST = 16

    train_subset = train_full.iloc[:N_TRAIN].copy()
    val_subset = val_full.iloc[:N_VAL].copy()
    test_subset = test_full.iloc[:N_TEST].copy()

    logger.info(f"Training samples: {len(train_subset)}")
    logger.info(f"Validation samples: {len(val_subset)}")
    logger.info(f"Test samples: {len(test_subset)}")

    # Get DataLoaders
    # This will process the specific files referenced in the subsets and cache them
    train_loader, val_loader, test_loader = get_dataloaders(
        train_subset, val_subset, test_subset
    )

    # Verification: Check DataLoader shapes
    try:
        inputs, targets = next(iter(train_loader))
        x_eeg, x_spec = inputs

        # Verify EEG shape: (Batch, 19, 128, 512)
        assert x_eeg.shape == (
            Config.BATCH_SIZE,
            19,
            128,
            512,
        ), f"Incorrect EEG shape: {x_eeg.shape}"

        # Verify Spec shape: (Batch, 4, 256, 256)
        assert x_spec.shape == (
            Config.BATCH_SIZE,
            4,
            256,
            256,
        ), f"Incorrect Spec shape: {x_spec.shape}"

        # Verify Targets shape: (Batch, 6)
        assert targets.shape == (
            Config.BATCH_SIZE,
            6,
        ), f"Incorrect Target shape: {targets.shape}"

        logger.info("DataLoader shapes verified successfully.")

    except StopIteration:
        raise RuntimeError("DataLoader is empty!")

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    print("\n[Step 3] Initializing Model...")

    model = DualScaleSpectrogramNet(Config)
    model.to(Config.DEVICE)

    # Verification: Forward pass with dummy batch
    with torch.no_grad():
        model.eval()
        dummy_out = model((x_eeg.to(Config.DEVICE), x_spec.to(Config.DEVICE)))

        # Check output shape
        assert dummy_out.shape == (
            Config.BATCH_SIZE,
            6,
        ), f"Model output shape mismatch: {dummy_out.shape}"

        # Check Softmax constraint (sum to 1)
        sums = dummy_out.sum(dim=1).cpu().numpy()
        assert np.allclose(
            sums, 1.0, atol=1e-5
        ), f"Model outputs do not sum to 1: {sums}"

    logger.info("Model initialized and verified.")

    # ==========================================
    # 4. Training Loop
    # ==========================================
    print("\n[Step 4] Running Training Loop...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    criterion = torch.nn.KLDivLoss(reduction="batchmean")

    best_kl = float("inf")

    for epoch in range(Config.EPOCHS):
        logger.info(f"--- Epoch {epoch + 1}/{Config.EPOCHS} ---")

        # Train
        train_loss, train_kl = train_one_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE, Config
        )

        # Validate
        val_loss, val_kl = validate(model, val_loader, criterion, Config.DEVICE)

        logger.info(f"Train Loss: {train_loss:.4f} | Train KL: {train_kl:.4f}")
        logger.info(f"Val Loss:   {val_loss:.4f} | Val KL:   {val_kl:.4f}")

        # Save best
        if val_kl < best_kl:
            best_kl = val_kl
            torch.save(model.state_dict(), Config.MODEL_PATH)
            logger.info("Saved best model.")

    # Verify model file exists
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError("Model checkpoint was not created.")

    # ==========================================
    # 5. Inference & Submission
    # ==========================================
    print("\n[Step 5] Generating Submission...")

    # Generate submission using the engine function
    # Note: We pass the subset test_df so the IDs match the loader
    generate_submission(model, test_loader, test_subset, Config, logger)

    # Verification: Check submission file
    if not os.path.exists(Config.SUBMISSION_FILE):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_FILE}"
        )

    sub_df = pd.read_csv(Config.SUBMISSION_FILE)

    # Check rows
    assert len(sub_df) == len(
        test_subset
    ), f"Submission row count mismatch. Expected {len(test_subset)}, got {len(sub_df)}"

    # Check columns
    expected_cols = [
        "eeg_id",
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Submission columns mismatch. Got {list(sub_df.columns)}"

    # Check probability sum
    vote_cols = expected_cols[1:]
    row_sums = sub_df[vote_cols].sum(axis=1)
    # Allow small float error
    assert np.allclose(
        row_sums, 1.0, atol=1e-4
    ), "Submission probabilities do not sum to 1."

    logger.info("Submission file verified successfully.")

    # Show first few rows
    print("\nSubmission Head:")
    print(sub_df.head())

    print("\n" + "=" * 50)
    print("Demo Completed Successfully")
    print("=" * 50)


if __name__ == "__main__":
    run_demo()
