import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
import library.config as config
from library.utils import seed_everything, EarlyStopping
from library.data_loader import get_dataloaders
from library.model import WideAsymmetricDCNResNet
from library.train import Trainer

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration of Provided Library Modules ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Fast Execution
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for rapid demonstration...")

    # Override config constants to use a tiny subset of data and few epochs
    config.MAX_TRAIN_SAMPLES = 2000
    config.MAX_VAL_SAMPLES = 500
    config.EPOCHS = 2
    config.BATCH_SIZE = 256  # Smaller batch size for the small subset

    # Ensure reproducibility
    seed_everything(config.SEED)
    print(
        f"Configured: Max Train Samples={config.MAX_TRAIN_SAMPLES}, Epochs={config.EPOCHS}"
    )

    # -------------------------------------------------------------------------
    # 2. Data Loading & Processing
    # -------------------------------------------------------------------------
    print("\n[Step 2] Loading and processing data...")

    # Force load_cached_data=False to ensure we process the subsampled data
    # defined above, rather than loading full cached arrays if they exist.
    train_loader, val_loader, test_loader, test_ids, input_dim = get_dataloaders(
        load_cached_data=False
    )

    # Validation: Check DataLoaders
    print(f"Input Dimension: {input_dim}")
    print(f"Train Batches: {len(train_loader)}")
    print(f"Val Batches: {len(val_loader)}")

    # Assertions to ensure data is loaded correctly
    assert input_dim > 0, "Input dimension should be positive."
    assert len(train_loader) > 0, "Train loader should not be empty."

    # Check a single batch structure
    sample_X, sample_y = next(iter(train_loader))
    print(f"Sample Batch Shape - X: {sample_X.shape}, y: {sample_y.shape}")
    assert sample_X.shape[1] == input_dim, "Feature dimension mismatch in loader."
    assert sample_y.max() < config.NUM_CLASSES, "Target labels exceed num_classes."

    # -------------------------------------------------------------------------
    # 3. Model Initialization & Verification
    # -------------------------------------------------------------------------
    print("\n[Step 3] Initializing and verifying model architecture...")

    model = WideAsymmetricDCNResNet(input_dim=input_dim, num_classes=config.NUM_CLASSES)
    model.to(config.DEVICE)

    # Dummy Forward Pass
    dummy_input = torch.randn(16, input_dim).to(config.DEVICE)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    print(f"Model Output Shape: {dummy_output.shape}")

    # Assertions for model logic
    assert dummy_output.shape == (
        16,
        config.NUM_CLASSES,
    ), f"Expected output shape (16, {config.NUM_CLASSES}), got {dummy_output.shape}"
    assert not torch.isnan(dummy_output).any(), "Model output contains NaNs."

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[Step 4] Running training loop...")

    trainer = Trainer(model, train_loader, val_loader, config.DEVICE)

    # Capture initial weights of the head to verify updates later
    initial_head_weights = model.head.weight.clone()

    # Run fit (uses the overridden config.EPOCHS = 2)
    trainer.fit(epochs=config.EPOCHS)

    # Check if weights updated
    final_head_weights = model.head.weight
    assert not torch.equal(
        initial_head_weights, final_head_weights
    ), "Model weights did not change after training."

    print("Training loop completed successfully.")

    # -------------------------------------------------------------------------
    # 5. Inference & Submission Generation
    # -------------------------------------------------------------------------
    print("\n[Step 5] Generating predictions and submission...")

    # Generate predictions
    preds = trainer.predict(test_loader)

    print(f"Predictions generated: {len(preds)}")
    print(f"First 10 predictions (0-6 scale): {preds[:10]}")

    # Assertions for inference
    assert len(preds) == len(
        test_ids
    ), f"Mismatch between predictions ({len(preds)}) and test IDs ({len(test_ids)})."

    # Post-processing: Convert 0-6 back to 1-7 class labels
    final_preds = preds + 1

    # Create submission DataFrame
    submission_df = pd.DataFrame(
        {config.ID_COL: test_ids, config.TARGET_COL: final_preds}
    )

    # Ensure ID is integer
    submission_df[config.ID_COL] = submission_df[config.ID_COL].astype(int)

    print("Submission DataFrame Head:")
    print(submission_df.head())

    # Verify format
    assert submission_df.shape[1] == 2, "Submission should have 2 columns."
    assert config.ID_COL in submission_df.columns, f"Missing {config.ID_COL} column."
    assert (
        config.TARGET_COL in submission_df.columns
    ), f"Missing {config.TARGET_COL} column."

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
