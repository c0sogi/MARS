import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config
from library.data import get_loaders
from library.model import RSNAModel
from library.utils import weighted_loss
from library.engine import RSNAEngine, run


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration Override for Fast Demonstration
    # -------------------------------------------------------------------------
    print(">>> 1. Configuring environment for demo...")

    # Enable Debug mode to use a small subset of data
    Config.DEBUG = True
    Config.DEBUG_DATA_SIZE = 10  # Use only 10 studies for train/val/test

    # Reduce computational load
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.GRADIENT_ACCUMULATION_STEPS = 1
    Config.NUM_WORKERS = 2

    # Reduce dimensions for speed (Model handles arbitrary input size due to GAP)
    Config.IMAGE_SIZE = (128, 128)
    Config.SEQ_LEN = 16  # Reduce sequence length from 96 to 16

    # Ensure reproducibility
    Config.setup()

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Image Size: {Config.IMAGE_SIZE}")
    print(f"Sequence Length: {Config.SEQ_LEN}")

    # -------------------------------------------------------------------------
    # 2. Verify Data Loading
    # -------------------------------------------------------------------------
    print("\n>>> 2. Verifying Data Loading...")

    # Initialize loaders (load_cached_data=False forces re-scan of directories for the subset)
    train_loader, val_loader, test_loader = get_loaders(
        load_cached_data=False, debug=True
    )

    # Fetch one batch
    images, targets = next(iter(train_loader))

    print(f"Batch Images Shape: {images.shape}")
    print(f"Batch Targets Shape: {targets.shape}")

    # Assertions
    # Expected: (Batch, Seq_Len, Channels, H, W)
    expected_shape = (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        Config.IN_CHANNELS,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    )
    assert (
        images.shape == expected_shape
    ), f"Image tensor shape mismatch. Expected {expected_shape}, got {images.shape}"

    # Expected: (Batch, 8) -> [C1..C7, Overall]
    assert targets.shape == (
        Config.BATCH_SIZE,
        8,
    ), f"Target tensor shape mismatch. Expected {(Config.BATCH_SIZE, 8)}, got {targets.shape}"

    # -------------------------------------------------------------------------
    # 3. Verify Model Architecture & Forward Pass
    # -------------------------------------------------------------------------
    print("\n>>> 3. Verifying Model Architecture...")

    device = Config.DEVICE
    # Initialize model (pretrained=False for speed in this check, Engine uses True)
    model = RSNAModel(pretrained=False).to(device)
    model.eval()

    # Move batch to device
    images = images.to(device)
    targets = targets.to(device)

    with torch.no_grad():
        logits = model(images)

    print(f"Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        Config.BATCH_SIZE,
        8,
    ), f"Logits shape mismatch. Expected {(Config.BATCH_SIZE, 8)}, got {logits.shape}"

    # Check for NaNs
    assert not torch.isnan(logits).any(), "Model produced NaN logits."

    # -------------------------------------------------------------------------
    # 4. Verify Loss Function
    # -------------------------------------------------------------------------
    print("\n>>> 4. Verifying Loss Function...")

    loss = weighted_loss(logits, targets)
    print(f"Calculated Loss: {loss.item()}")

    assert loss.dim() == 0, "Loss should be a scalar tensor."
    assert loss.item() >= 0, "Loss should be non-negative."

    # -------------------------------------------------------------------------
    # 5. Run Full Engine (Training & Inference)
    # -------------------------------------------------------------------------
    print("\n>>> 5. Running Full Engine Pipeline...")

    # Initialize Engine
    engine = RSNAEngine(device=device)

    # Run Training
    # This calls fit(), which runs train_one_epoch and validate for Config.EPOCHS
    engine.fit(train_loader, val_loader)

    # Run Inference
    # This generates submission.csv
    submission_df = engine.predict_and_submit(test_loader)

    print("Engine execution completed.")

    # -------------------------------------------------------------------------
    # 6. Validate Submission Output
    # -------------------------------------------------------------------------
    print("\n>>> 6. Validating Submission...")

    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(submission_path), "Submission file was not created."

    df = pd.read_csv(submission_path)
    print(f"Submission Rows: {len(df)}")
    print(df.head())

    # Check columns
    assert (
        "row_id" in df.columns and "fractured" in df.columns
    ), "Submission missing required columns."

    # Check values are probabilities
    assert (
        df["fractured"].min() >= 0.0 and df["fractured"].max() <= 1.0
    ), "Probabilities out of range [0, 1]."

    # Check row count
    # We used Config.DEBUG_DATA_SIZE studies for test.
    # Each study has 8 targets. Total rows = DEBUG_DATA_SIZE * 8
    expected_rows = Config.DEBUG_DATA_SIZE * 8
    assert (
        len(df) == expected_rows
    ), f"Incorrect number of rows in submission. Expected {expected_rows}, got {len(df)}."

    print("\n>>> Demo completed successfully!")


if __name__ == "__main__":
    # Suppress specific warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
