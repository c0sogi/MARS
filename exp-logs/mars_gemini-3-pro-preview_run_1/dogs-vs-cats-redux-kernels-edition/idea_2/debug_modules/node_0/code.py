import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.dataset import DogCatDataset, get_transforms
from library.model import DogCatClassifier
from library.trainer import run_training_pipeline


def main():
    print("--- Starting Demonstration Script ---")

    # -------------------------------------------------------------------------
    # 1. Setup & Configuration Overrides
    # -------------------------------------------------------------------------
    # We modify Config attributes at runtime to optimize for a quick demo run.
    print("\n[1] Configuring runtime environment...")

    # Use a specific demo directory to avoid interfering with other runs
    demo_working_dir = "./working/demo_execution"
    Config.working_dir = demo_working_dir
    Config.model_dir = os.path.join(demo_working_dir, "models")
    Config.submission_dir = demo_working_dir

    # Ensure directories exist
    os.makedirs(Config.model_dir, exist_ok=True)
    os.makedirs(Config.submission_dir, exist_ok=True)

    # Speed optimizations
    Config.image_size = 128  # Reduce image size for faster processing
    Config.pretrained = False  # Disable downloading weights for speed
    Config.num_workers = 2  # Reduce worker overhead

    seed_everything(Config.seed)
    print(f"Working directory set to: {Config.working_dir}")
    print(f"Image size set to: {Config.image_size}x{Config.image_size}")

    # -------------------------------------------------------------------------
    # 2. Dataset & DataLoader Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Dataset and Transforms...")

    # Load metadata
    df_train = pd.read_csv(Config.train_metadata_path)

    # Sample a tiny subset for verification
    df_sample = df_train.head(10).reset_index(drop=True)

    # Instantiate dataset
    dataset = DogCatDataset(
        df=df_sample, transforms=get_transforms("train"), mode="train"
    )

    # Retrieve one sample
    image, label = dataset[0]

    # Assertions
    print(f"Sample Image Shape: {image.shape}")
    print(f"Sample Label: {label}")

    # Check shape: (Channels, Height, Width)
    assert image.shape == (
        3,
        Config.image_size,
        Config.image_size,
    ), f"Expected image shape (3, {Config.image_size}, {Config.image_size}), got {image.shape}"

    # Check label type (should be a float tensor for BCEWithLogitsLoss)
    assert isinstance(label, torch.Tensor), "Label should be a torch.Tensor"
    assert label.dtype == torch.float32, "Label dtype should be float32"

    print("Dataset verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DogCatClassifier(pretrained=False).to(device)
    model.eval()

    # Create dummy batch (Batch_Size, Channels, Height, Width)
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, Config.image_size, Config.image_size).to(
        device
    )

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    # Output should be (Batch_Size,) because the model squeezes the last dimension
    assert output.shape == (
        batch_size,
    ), f"Expected output shape ({batch_size},), got {output.shape}"

    print("Model verification passed.")

    # -------------------------------------------------------------------------
    # 4. End-to-End Training Pipeline Execution
    # -------------------------------------------------------------------------
    print("\n[4] Executing Training Pipeline (Debug Mode)...")

    # We run the pipeline with:
    # - debug=True (subsamples data to 200 train, 50 test)
    # - epochs=1 (minimal training)
    # - n_folds=2 (minimal cross-validation)
    # - batch_size=8 (small batch for safety)

    try:
        run_training_pipeline(
            debug=True,
            epochs=1,
            n_folds=2,
            batch_size=8,
            learning_rate=1e-4,
            weight_decay=1e-2,
        )
        print("Training pipeline executed successfully.")
    except Exception as e:
        print(f"Training pipeline failed with error: {e}")
        raise e

    # -------------------------------------------------------------------------
    # 5. Submission File Verification
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Submission File...")

    submission_path = os.path.join(Config.submission_dir, "submission.csv")

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)
    print(f"Submission file loaded. Rows: {len(df_sub)}")
    print(df_sub.head())

    # Assertions
    required_columns = {"id", "label"}
    assert required_columns.issubset(
        df_sub.columns
    ), f"Submission missing required columns. Found: {df_sub.columns}"

    # Check ID format (should be numeric/int)
    assert pd.api.types.is_numeric_dtype(df_sub["id"]), "ID column should be numeric"

    # Check Probabilities (should be between 0 and 1)
    probs = df_sub["label"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), f"Probabilities out of range [0, 1]. Min: {probs.min()}, Max: {probs.max()}"

    # In debug mode, we expect exactly 50 test samples
    expected_samples = 50
    assert (
        len(df_sub) == expected_samples
    ), f"Expected {expected_samples} rows in debug mode, got {len(df_sub)}"

    print("Submission verification passed.")
    print("\n--- Demonstration Complete ---")


if __name__ == "__main__":
    main()
