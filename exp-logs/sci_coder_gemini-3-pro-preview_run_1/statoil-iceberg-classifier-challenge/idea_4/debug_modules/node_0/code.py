import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, get_device
from library.data_loader import load_and_process_data
from library.model import IcebergResNet34
from library.trainer import train_all_folds
from library.inference import generate_ensemble_predictions

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Iceberg Classification Pipeline Demonstration ===")

    # ---------------------------------------------------------
    # 1. Configuration Override for Demonstration Speed
    # ---------------------------------------------------------
    print("\n[Step 1] Configuring environment...")

    # Override Config to run a minimal version of the pipeline
    Config.NUM_EPOCHS = 1  # Run only 1 epoch per fold for speed
    Config.NUM_FOLDS = 2  # Run only 2 folds instead of 5
    Config.BATCH_SIZE = 16  # Moderate batch size
    Config.WORKING_DIR = "./working/demo_run"  # Separate working dir for this demo
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)
    print(
        f"Configuration updated: Epochs={Config.NUM_EPOCHS}, Folds={Config.NUM_FOLDS}"
    )
    print(f"Working Directory: {Config.WORKING_DIR}")

    # ---------------------------------------------------------
    # 2. Data Processing and Loading
    # ---------------------------------------------------------
    print("\n[Step 2] Processing and Validating Data...")

    # Trigger data processing.
    # load_cached_data=False ensures we test the raw data processing logic from scratch.
    data = load_and_process_data(load_cached_data=False)

    # Verify Data Integrity
    train_images = data["train_images"]
    train_angles = data["train_angles"]
    train_labels = data["train_labels"]

    # Check shapes
    # Images should be (N, 75, 75, 3) after processing (Band1, Band2, Mean)
    assert train_images.ndim == 4, "Train images should be 4D (N, H, W, C)"
    assert train_images.shape[1:] == (
        75,
        75,
        3,
    ), f"Expected (75, 75, 3), got {train_images.shape[1:]}"
    assert len(train_images) == len(
        train_angles
    ), "Mismatch between images and angles count"
    assert len(train_images) == len(
        train_labels
    ), "Mismatch between images and labels count"

    print(f"Data Loaded Successfully:")
    print(f" - Train Images: {train_images.shape}")
    print(f" - Train Angles: {train_angles.shape}")
    print(f" - Test Images: {data['test_images'].shape}")

    # ---------------------------------------------------------
    # 3. Model Initialization and Forward Pass Check
    # ---------------------------------------------------------
    print("\n[Step 3] Verifying Model Architecture...")

    device = get_device()
    model = IcebergResNet34().to(device)

    # Create a dummy batch to simulate the input after Albumentations resizing (224x224)
    # Batch size 4, 3 channels, 224x224
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    dummy_angle = torch.randn(4).to(device)  # Normalized angles

    model.eval()
    with torch.no_grad():
        output = model(dummy_input, dummy_angle)

    # Expect output shape (Batch, 1) for binary classification logits
    assert output.shape == (
        4,
        1,
    ), f"Model output shape mismatch. Expected (4, 1), got {output.shape}"
    print("Model forward pass successful. Output shape verified.")

    # ---------------------------------------------------------
    # 4. Training Loop Execution
    # ---------------------------------------------------------
    print("\n[Step 4] Executing Training Pipeline...")

    # train_all_folds runs the training loop for Config.NUM_FOLDS
    # It saves the best model for each fold and returns the paths
    model_paths = train_all_folds()

    # Verify that we got paths for the expected number of folds
    assert (
        len(model_paths) == Config.NUM_FOLDS
    ), f"Expected {Config.NUM_FOLDS} model paths, got {len(model_paths)}"

    # Verify files actually exist
    for path in model_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Trained model file missing: {path}")

    print("Training complete. Models saved.")

    # ---------------------------------------------------------
    # 5. Inference and Submission
    # ---------------------------------------------------------
    print("\n[Step 5] Generating Submission...")

    # Generate predictions using the ensemble of trained models
    generate_ensemble_predictions(model_paths)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Validate Submission Format
    required_columns = ["id", "is_iceberg"]
    if not all(col in df_sub.columns for col in required_columns):
        raise ValueError(
            f"Submission missing required columns. Found: {df_sub.columns}"
        )

    # Validate Probabilities
    if df_sub["is_iceberg"].min() < 0 or df_sub["is_iceberg"].max() > 1:
        raise ValueError("Predictions contain values outside [0, 1] range.")

    # Validate Row Count against Metadata
    test_meta = pd.read_csv(Config.TEST_META)
    expected_rows = len(test_meta)
    if len(df_sub) != expected_rows:
        raise AssertionError(
            f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"
        )

    print(f"Submission generated successfully at {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {df_sub.shape}")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
