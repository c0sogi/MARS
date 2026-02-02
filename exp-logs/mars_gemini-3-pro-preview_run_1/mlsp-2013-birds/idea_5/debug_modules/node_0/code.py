import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, create_submission
from library.data import get_dataloaders
from library.model import build_model
from library.engine import train_one_epoch, evaluate, predict


def main():
    print("Initializing demonstration...")

    # 1. Configuration Override for Speed
    # We modify the Config class directly to ensure the demo runs quickly.
    print("Overriding configuration for fast demonstration...")
    Config.DEBUG_SAMPLE_SIZE = 32  # Use only 32 samples for train/val/test
    Config.BATCH_SIZE = 8  # Small batch size
    Config.TEACHER_EPOCHS = 1  # Train for only 1 epoch
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # 2. Data Loading
    print("Loading data...")
    # We force re-processing (load_cached_data=False) to demonstrate the raw data pipeline
    # and to ensure our DEBUG_SAMPLE_SIZE takes effect on the raw metadata read.
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        load_cached_data=False
    )

    # Verification: Check DataLoaders
    print("Verifying data loaders...")
    try:
        images, labels = next(iter(train_loader))
        print(f"  Train Batch Shape: Images {images.shape}, Labels {labels.shape}")

        # Assertions
        assert images.shape == (
            Config.BATCH_SIZE,
            3,
            Config.IMG_HEIGHT,
            Config.IMG_WIDTH,
        ), f"Incorrect image batch shape: {images.shape}"
        assert labels.shape == (
            Config.BATCH_SIZE,
            Config.NUM_CLASSES,
        ), f"Incorrect label batch shape: {labels.shape}"
        assert images.dtype == torch.float32, "Images should be float32"
        assert labels.dtype == torch.float32, "Labels should be float32"

    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # 3. Model Initialization
    print("Building model...")
    model = build_model(Config.DEVICE)

    # Verification: Model Forward Pass
    print("Verifying model forward pass...")
    images = images.to(Config.DEVICE)
    with torch.no_grad():
        outputs = model(images)

    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {outputs.shape}"
    print("  Model forward pass successful.")

    # 4. Training Loop Demonstration
    print("Starting training loop demonstration...")
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # Train for one epoch
    avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
    print(f"  Epoch 1 Loss: {avg_loss:.4f}")

    assert not np.isnan(avg_loss), "Training loss returned NaN"
    assert avg_loss > 0, "Training loss should be positive"

    # 5. Evaluation Demonstration
    print("Starting evaluation demonstration...")
    val_loss, val_auc = evaluate(model, val_loader, criterion, Config.DEVICE)
    print(f"  Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0 <= val_auc <= 1, "AUC score out of range [0, 1]"

    # 6. Prediction and Submission
    print("Generating predictions on test set...")
    test_probs = predict(model, test_loader, Config.DEVICE)

    print(f"  Prediction shape: {test_probs.shape}")
    expected_test_samples = (
        min(len(test_ids), Config.DEBUG_SAMPLE_SIZE)
        if Config.DEBUG_SAMPLE_SIZE
        else len(test_ids)
    )

    # Note: In the library code, load_and_process_data applies head(DEBUG_SAMPLE_SIZE) to the dataframe.
    # So the number of predictions matches the reduced dataset size.
    assert test_probs.shape == (
        len(test_ids),
        Config.NUM_CLASSES,
    ), f"Prediction shape mismatch. Expected {(len(test_ids), Config.NUM_CLASSES)}, got {test_probs.shape}"

    # Create Submission
    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    print(f"Creating submission file at {submission_path}...")
    create_submission(test_probs, test_ids, submission_path)

    # Verification: Check submission file
    assert os.path.exists(submission_path), "Submission file was not created"

    df_sub = pd.read_csv(submission_path)
    print(f"  Submission file loaded. Rows: {len(df_sub)}")

    # Expected rows = num_test_samples * num_classes
    expected_rows = len(test_ids) * Config.NUM_CLASSES
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    assert (
        "Id" in df_sub.columns and "Probability" in df_sub.columns
    ), "Submission columns missing"

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()
