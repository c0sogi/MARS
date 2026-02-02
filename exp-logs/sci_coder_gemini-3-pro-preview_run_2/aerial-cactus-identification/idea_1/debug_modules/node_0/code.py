import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.dataset import get_datasets
from library.model import ShallowCNN
from library.engine import train_model, predict_and_submit


def main():
    print("Initializing Cactus Classification Demo...")

    # 1. Setup and Configuration
    # Set seed for reproducibility
    set_seed(Config.SEED)

    # Override Config for a fast demonstration
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 32
    Config.DEBUG_SAMPLE_SIZE = 100  # Use a very small subset for speed

    print(
        f"Configuration: Epochs={Config.NUM_EPOCHS}, Batch Size={Config.BATCH_SIZE}, Debug Size={Config.DEBUG_SAMPLE_SIZE}"
    )

    # 2. Data Loading
    print("Loading datasets...")
    # Use debug=True to load truncated datasets
    train_ds, val_ds, test_ds = get_datasets(debug=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Use 0 workers for simple script execution
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    # --- Validation: Dataset Sizes ---
    print("Verifying dataset sizes...")
    assert (
        len(train_ds) <= Config.DEBUG_SAMPLE_SIZE
    ), "Train dataset size exceeds debug limit"
    assert (
        len(val_ds) <= Config.DEBUG_SAMPLE_SIZE
    ), "Val dataset size exceeds debug limit"
    assert (
        len(test_ds) <= Config.DEBUG_SAMPLE_SIZE
    ), "Test dataset size exceeds debug limit"

    # --- Validation: Data Shapes ---
    print("Verifying data shapes...")
    sample_imgs, sample_labels = next(iter(train_loader))

    # Expected Image Shape: (Batch, 3, 32, 32)
    assert sample_imgs.dim() == 4, "Image batch should be 4-dimensional"
    assert sample_imgs.shape[1] == 3, "Images should have 3 channels"
    assert (
        sample_imgs.shape[2] == 32 and sample_imgs.shape[3] == 32
    ), "Images should be 32x32"

    # Expected Label Shape: (Batch, 1)
    assert sample_labels.dim() == 2, "Labels batch should be 2-dimensional"
    assert sample_labels.shape[1] == 1, "Labels should have shape (B, 1)"

    print("Data validation passed.")

    # 3. Model Initialization
    print("Initializing model...")
    model = ShallowCNN()

    # --- Validation: Model Forward Pass ---
    print("Verifying model forward pass...")
    # Move sample to device if available, though here we test on CPU for simplicity or Config.DEVICE
    device = torch.device(Config.DEVICE)
    model.to(device)
    sample_imgs = sample_imgs.to(device)

    with torch.no_grad():
        output = model(sample_imgs)

    assert output.shape == (
        sample_imgs.shape[0],
        1,
    ), f"Model output shape mismatch. Expected {(sample_imgs.shape[0], 1)}, got {output.shape}"
    print("Model validation passed.")

    # 4. Training
    print("Starting training...")
    trained_model = train_model(model, train_loader, val_loader, config=Config)

    # --- Validation: Model Artifacts ---
    print("Verifying training artifacts...")
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model checkpoint not found at {Config.MODEL_SAVE_PATH}"
    print("Training validation passed.")

    # 5. Inference and Submission
    print("Generating predictions...")
    predict_and_submit(trained_model, test_loader, Config.SUBMISSION_PATH, device)

    # --- Validation: Submission File ---
    print("Verifying submission file...")
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    expected_cols = ["id", "has_cactus"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Check length
    assert len(df_sub) == len(
        test_ds
    ), f"Submission length mismatch. Expected {len(test_ds)}, got {len(df_sub)}"

    # Check value range (probabilities should be between 0 and 1)
    assert df_sub["has_cactus"].min() >= 0.0, "Probabilities contain values < 0"
    assert df_sub["has_cactus"].max() <= 1.0, "Probabilities contain values > 1"

    print("Submission validation passed.")
    print("Demo completed successfully.")


if __name__ == "__main__":
    main()
