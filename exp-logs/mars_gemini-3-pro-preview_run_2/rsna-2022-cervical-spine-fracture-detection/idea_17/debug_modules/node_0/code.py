import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings

# Filter warnings for clean output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.utils import seed_everything, setup_logger
from library.data import create_dataloaders
from library.model import Calibrated25DModel
from library.engine import train_one_epoch, validate, predict_and_submit


def main():
    # 1. Setup and Configuration Overrides
    print(">>> Setting up configuration for demo...")
    seed_everything(Config.SEED)

    # Override Config for speed and resource efficiency in this demo
    Config.IMAGE_SIZE = (256, 256)  # Reduce resolution
    Config.SEQ_LEN = 16  # Reduce sequence length (fewer slices per study)
    Config.BATCH_SIZE = 2  # Small batch size
    Config.NUM_WORKERS = 2  # Reduce workers to minimize overhead
    Config.EPOCHS = 1  # Single epoch
    Config.ACCUMULATION_STEPS = 1  # No accumulation needed for small batch demo
    Config.DEBUG = True  # Enable debug mode logic if present

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Load and Subset Metadata
    print(">>> Loading and subsetting metadata...")
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Select a small subset of studies for the demo
    # We select studies that actually exist in the directories to avoid errors
    train_subset = train_df.iloc[:10].reset_index(drop=True)
    val_subset = val_df.iloc[:4].reset_index(drop=True)
    test_subset = test_df.iloc[:4].reset_index(drop=True)

    print(f"Train subset size: {len(train_subset)}")
    print(f"Val subset size: {len(val_subset)}")
    print(f"Test subset size: {len(test_subset)}")

    # 3. Create DataLoaders
    print(">>> Creating DataLoaders...")
    # We pass the subsets. The create_dataloaders function handles caching paths.
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df=train_subset,
        val_df=val_subset,
        test_df=test_subset,
        load_cached_data=False,  # Force re-scan for the subset to avoid loading full cache
        debug=False,  # We already manually subsetted, so we set debug=False here
    )

    # Validate DataLoader output
    sample_batch, sample_labels = next(iter(train_loader))
    print(f"Batch Image Shape: {sample_batch.shape}")  # Should be (B, Seq, C, H, W)
    print(f"Batch Label Shape: {sample_labels.shape}")  # Should be (B, 8)

    assert sample_batch.shape == (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        3,
        *Config.IMAGE_SIZE,
    ), f"Incorrect batch shape: {sample_batch.shape}"
    assert sample_labels.shape == (
        Config.BATCH_SIZE,
        8,
    ), f"Incorrect label shape: {sample_labels.shape}"

    # 4. Initialize Model
    print(">>> Initializing Calibrated25DModel...")
    model = Calibrated25DModel()
    model.to(device)

    # Verify Model Forward Pass
    print(">>> Verifying model forward pass...")
    with torch.no_grad():
        dummy_input = torch.randn(2, Config.SEQ_LEN, 3, *Config.IMAGE_SIZE).to(device)
        dummy_output = model(dummy_input)

    print(f"Model Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (
        2,
        8,
    ), f"Model output shape mismatch. Expected (2, 8), got {dummy_output.shape}"
    assert not torch.isnan(dummy_output).any(), "Model produced NaN outputs."

    # 5. Training Loop Demo
    print(">>> Starting Training Demo (1 Epoch)...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    train_loss = train_one_epoch(
        model, train_loader, optimizer, scaler, device, epoch=0
    )
    print(f"Train Loss: {train_loss:.4f}")
    assert np.isfinite(train_loss), "Training loss is not finite."

    # 6. Validation Loop Demo
    print(">>> Starting Validation Demo...")
    val_loss = validate(model, val_loader, device)
    print(f"Validation Loss: {val_loss:.4f}")
    assert np.isfinite(val_loss), "Validation loss is not finite."

    # Save a dummy checkpoint for inference to use
    torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
    print(f"Saved checkpoint to {Config.MODEL_SAVE_PATH}")

    # 7. Inference & Submission Demo
    print(">>> Starting Inference & Submission Demo...")
    # Ensure sample submission exists (it's in input/)
    if not os.path.exists(Config.SAMPLE_SUBMISSION_PATH):
        raise FileNotFoundError("Sample submission file missing.")

    predict_and_submit(model, test_loader, test_subset, device)

    # Verify Submission File
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not created at {Config.SUBMISSION_PATH}"
        )

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission file head:")
    print(submission_df.head())

    # Check if we have probabilities
    assert "fractured" in submission_df.columns, "Submission missing 'fractured' column"
    assert "row_id" in submission_df.columns, "Submission missing 'row_id' column"

    # Check that predictions are within [0, 1] (though sigmoid output guarantees this, we verify file content)
    # Note: predict_and_submit fills 0.5 for rows not in test_subset, so we check general validity
    preds = submission_df["fractured"].values
    assert (preds >= 0).all() and (preds <= 1).all(), "Predictions out of range [0, 1]"

    print("\n>>> Demo completed successfully!")


if __name__ == "__main__":
    main()
