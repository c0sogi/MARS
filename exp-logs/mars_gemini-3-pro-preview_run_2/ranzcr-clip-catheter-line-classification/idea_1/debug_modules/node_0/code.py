import os
import sys
import torch
import pandas as pd
import numpy as np

# Import library components
from library.config import Config
from library.dataset import get_dataloaders
from library.model import ResNet18Model
from library.trainer import fit
from library.inference import create_submission


def main():
    print("=== Catheter Detection Pipeline Demonstration ===")

    # 1. Configuration Override
    # We modify the Config class attributes directly to optimize for a fast demonstration.
    print("\n>>> 1. Configuring environment for rapid execution...")
    Config.DEBUG = True  # Use a tiny subset of data
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.PRETRAINED = False  # Skip downloading weights to save time/bandwidth

    # Ensure working directory exists (as defined in Config)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    print(f"Working Directory: {Config.WORKING_DIR}")

    # 2. Data Loading Demonstration
    print("\n>>> 2. Verifying Data Loading...")
    # Initialize DataLoaders
    loaders = get_dataloaders(batch_size=Config.BATCH_SIZE, debug=Config.DEBUG)
    train_loader = loaders["train"]

    # Check if loader is populated
    assert len(train_loader) > 0, "Train loader should not be empty."

    # Fetch a single batch to verify shapes
    images, labels = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")  # Expected: (4, 3, 512, 512)
    print(f"Batch Label Shape: {labels.shape}")  # Expected: (4, 11)

    # Assertions for data integrity
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect image tensor shape"
    assert labels.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Incorrect label tensor shape"
    assert labels.dtype == torch.float32, "Labels must be float32"

    # 3. Model Architecture Demonstration
    print("\n>>> 3. Verifying Model Architecture...")
    # Instantiate model
    model = ResNet18Model(num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED)
    model.to(Config.DEVICE)
    model.eval()

    # Perform a dummy forward pass
    with torch.no_grad():
        input_tensor = images.to(Config.DEVICE)
        outputs = model(input_tensor)

    print(f"Model Output Shape: {outputs.shape}")

    # Assertions for model output
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"
    assert not torch.isnan(outputs).any(), "Model output contains NaNs"

    # 4. Training Loop Demonstration
    print("\n>>> 4. Running Training Loop (Short)...")
    # Run the training process using the trainer library
    # This will save the best model to Config.WORKING_DIR/best_model.pth
    best_auc = fit(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        debug=Config.DEBUG,
        patience=1,
    )

    print(f"Training finished. Best AUC (Validation): {best_auc}")

    # Verify that the model artifact was saved
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(model_path), f"Model file was not saved to {model_path}"
    print("Model artifact successfully created.")

    # 5. Inference and Submission Demonstration
    print("\n>>> 5. Running Inference and Generating Submission...")
    # Generate submission using the trained model
    create_submission(
        model_path=model_path, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
    )

    # Verify submission file existence
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Load and inspect submission
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File Shape: {df_sub.shape}")
    print("Submission Columns:", df_sub.columns.tolist())

    # Verify columns match requirements
    expected_cols = ["StudyInstanceUID"] + Config.TARGET_COLS
    assert (
        list(df_sub.columns) == expected_cols
    ), "Submission columns do not match the required format"

    # Verify row count
    # In debug mode, get_dataloaders subsets the test set to batch_size length
    assert (
        len(df_sub) == Config.BATCH_SIZE
    ), f"Expected {Config.BATCH_SIZE} rows in debug submission, found {len(df_sub)}"

    print("\n>>> Demonstration completed successfully. All components verified.")


if __name__ == "__main__":
    main()
