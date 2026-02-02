import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    DEVICE,
    SUBMISSION_DIR,
    IN_CHANNELS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)
from library.dataset import get_dataloaders
from library.model import get_multichannel_resnet
from library.engine import train_one_epoch, validate, generate_submission
from library.utils import seed_everything


def main():
    print("Starting Technosignature Pipeline Demo...")

    # 1. Setup
    # Set seeds for reproducibility
    seed_everything(42)

    # Configuration for a fast demo run
    BATCH_SIZE = 8
    DEBUG_SIZE = 50  # Use only 50 samples to ensure speed
    LEARNING_RATE = 1e-3

    # 2. Data Loading
    print("\n--- Initializing DataLoaders (Debug Mode) ---")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_meta_path=TRAIN_METADATA_PATH,
        val_meta_path=VAL_METADATA_PATH,
        test_meta_path=TEST_METADATA_PATH,
        batch_size=BATCH_SIZE,
        debug=True,
        debug_size=DEBUG_SIZE,
    )

    # Verify Data Loading
    try:
        inputs, targets = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    print(f"Batch shapes -> Inputs: {inputs.shape}, Targets: {targets.shape}")

    # Assertions to ensure data pipeline is correct
    assert inputs.shape == (
        BATCH_SIZE,
        IN_CHANNELS,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    ), f"Input shape mismatch. Expected {(BATCH_SIZE, IN_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)}, got {inputs.shape}"
    assert targets.shape == (
        BATCH_SIZE,
    ), f"Target shape mismatch. Expected {(BATCH_SIZE,)}, got {targets.shape}"
    assert inputs.dtype == torch.float32, "Input tensor should be float32"

    # 3. Model Initialization
    print("\n--- Initializing Model ---")
    # Using pretrained=False to avoid internet dependency during this demo
    model = get_multichannel_resnet(pretrained=False)
    model = model.to(DEVICE)

    # Verify Model Forward Pass
    with torch.no_grad():
        # Move sample batch to device
        sample_input = inputs.to(DEVICE)
        output = model(sample_input)

    print(f"Model output shape: {output.shape}")
    assert output.shape == (
        BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected {(BATCH_SIZE, 1)}, got {output.shape}"

    # 4. Training Loop Demo
    print("\n--- Running Training Step (1 Epoch) ---")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Run one epoch
    train_loss = train_one_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=DEVICE,
    )

    print(f"Training Loss: {train_loss:.6f}")
    assert not np.isnan(train_loss), "Training loss is NaN"
    assert train_loss > 0, "Training loss should be positive"

    # 5. Validation Demo
    print("\n--- Running Validation Step ---")
    val_loss, val_auc = validate(
        model=model, dataloader=val_loader, criterion=criterion, device=DEVICE
    )

    print(f"Validation Loss: {val_loss:.6f}")
    print(f"Validation AUC: {val_auc:.6f}")

    # Basic sanity checks for metrics
    assert isinstance(val_loss, float), "Validation loss should be a float"
    assert 0 <= val_auc <= 1, "AUC score must be between 0 and 1"

    # 6. Submission Generation
    print("\n--- Generating Submission ---")
    # Define output path
    submission_file = os.path.join(SUBMISSION_DIR, "demo_submission.csv")

    generate_submission(
        model=model, dataloader=test_loader, device=DEVICE, save_path=submission_file
    )

    # Verify Submission File
    assert os.path.exists(
        submission_file
    ), f"Submission file not found at {submission_file}"

    df_sub = pd.read_csv(submission_file)
    print(f"Submission file loaded. Shape: {df_sub.shape}")
    print(f"Columns: {list(df_sub.columns)}")

    # Check dimensions
    # In debug mode, test set size is min(total_test, DEBUG_SIZE)
    # Total test size is 6000, DEBUG_SIZE is 50 -> Expect 50 rows
    expected_rows = min(6000, DEBUG_SIZE)
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    # Check columns
    assert list(df_sub.columns) == [
        "id",
        "target",
    ], "Submission columns mismatch. Expected ['id', 'target']"

    # Check value ranges
    assert (
        df_sub["target"].min() >= 0 and df_sub["target"].max() <= 1
    ), "Predictions should be probabilities between 0 and 1"

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()
