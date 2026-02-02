import os
import torch
import pandas as pd
import numpy as np
import random
import sys

# Import from the provided library
from library.config import Config
from library.dataset import get_dataloaders
from library.model import build_model
from library.engine import run_training


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_dataset_and_loader():
    print("\n=== Verifying Dataset and DataLoader ===")

    # Use debug=True to load a small subset (defined in dataset.py as batch_size * 2 for train)
    dataloaders = get_dataloaders(debug=True)
    train_loader = dataloaders["train"]

    print(f"Train loader length (batches): {len(train_loader)}")

    # Fetch one batch
    images, labels = next(iter(train_loader))

    # Check Image Shapes: (Batch_Size, 3, Height, Width)
    # Config.BATCH_SIZE is 128, Config.IMAGE_SIZE is (224, 224)
    expected_shape = (Config.BATCH_SIZE, 3, Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1])
    print(f"Image tensor shape: {images.shape}")
    assert (
        images.shape == expected_shape
    ), f"Expected image shape {expected_shape}, got {images.shape}"

    # Check Label Shapes: (Batch_Size,)
    print(f"Label tensor shape: {labels.shape}")
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Expected label shape {(Config.BATCH_SIZE,)}, got {labels.shape}"

    # Check Data Types
    assert images.dtype == torch.float32, "Images should be float32"
    assert labels.dtype == torch.float32, "Labels should be float32"

    print("Dataset and DataLoader verification passed.")
    return images, labels


def verify_model_forward_pass(images):
    print("\n=== Verifying Model Architecture and Forward Pass ===")

    device = Config.DEVICE
    print(f"Using device: {device}")

    model = build_model()
    model.eval()  # Set to eval mode for deterministic check

    # Move images to device
    images = images.to(device)

    with torch.no_grad():
        outputs = model(images)

    # Check Output Shape: (Batch_Size, 1) because NUM_CLASSES = 1
    expected_output_shape = (Config.BATCH_SIZE, 1)
    print(f"Model output shape: {outputs.shape}")
    assert (
        outputs.shape == expected_output_shape
    ), f"Expected output shape {expected_output_shape}, got {outputs.shape}"

    # Check for NaN or Inf
    assert not torch.isnan(outputs).any(), "Model output contains NaNs"
    assert not torch.isinf(outputs).any(), "Model output contains Infs"

    print("Model forward pass verification passed.")


def verify_training_pipeline():
    print("\n=== Verifying Training Pipeline (Engine) ===")

    # Run a quick training session: 1 epoch, debug mode (small dataset)
    # This tests the integration of loader, model, loss, optimizer, and saving.
    try:
        run_training(epochs=1, debug=True)
    except Exception as e:
        print(f"Training pipeline failed with error: {e}")
        raise e

    print("Training pipeline execution completed.")


def verify_submission_file():
    print("\n=== Verifying Submission File ===")

    submission_path = Config.SUBMISSION_FILE
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df = pd.read_csv(submission_path)
    print(f"Submission file loaded. Rows: {len(df)}")
    print(df.head())

    # Check Columns
    assert "id" in df.columns, "Column 'id' missing in submission"
    assert "label" in df.columns, "Column 'label' missing in submission"

    # Check ID types
    assert pd.api.types.is_integer_dtype(df["id"]), "ID column should be integers"

    # Check Probability Range
    probs = df["label"]
    assert probs.min() >= 0.0, "Probabilities contain values < 0"
    assert probs.max() <= 1.0, "Probabilities contain values > 1"

    # In debug mode, get_dataloaders subsets test set to BATCH_SIZE (128)
    # So we expect 128 rows in submission
    expected_rows = Config.BATCH_SIZE
    assert (
        len(df) == expected_rows
    ), f"Expected {expected_rows} rows in debug submission, got {len(df)}"

    print("Submission file verification passed.")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(Config.SEED)

    # 1. Verify Data Loading
    batch_images, batch_labels = verify_dataset_and_loader()

    # 2. Verify Model
    verify_model_forward_pass(batch_images)

    # 3. Verify Training Engine
    verify_training_pipeline()

    # 4. Verify Output
    verify_submission_file()

    print("\nAll verifications passed successfully!")
