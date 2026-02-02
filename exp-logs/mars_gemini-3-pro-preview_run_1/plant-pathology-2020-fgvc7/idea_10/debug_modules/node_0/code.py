import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from provided library
from library.config import Config
from library.utils import seed_everything
from library.dataset import get_loaders, get_test_loader
from library.models import get_model
from library.calibration import run_calibration
from library.production import train_final_model
from library.inference import predict_and_submit


def main():
    print("==== Starting Demonstration Script ====")

    # 1. Setup and Configuration Overrides for Speed
    print("\n[1] Configuring Environment...")
    seed_everything(42)

    # Override Config to use a separate working directory for this demo
    # and reduce computational load
    Config.WORKING_DIR = "./working/demo_execution"
    Config.OUTPUT_DIR = os.path.join(Config.WORKING_DIR, "output")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Clean up demo directory if it exists to ensure fresh cache generation
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    # Reduce folds and epochs for rapid execution
    Config.N_FOLDS = 2
    Config.EPOCHS_CALIBRATION = 1

    # Initialize directories
    Config.setup()
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Device: {Config.DEVICE}")

    # 2. Verify Data Loading
    print("\n[2] Verifying Data Loading...")
    # Get loaders for Fold 0
    train_loader, val_loader = get_loaders(fold=0, mode="calibration")

    # Fetch one batch
    images, targets = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Target Shape: {targets.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected image shape {(Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)}, got {images.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Expected target shape {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {targets.shape}"

    # Check normalization (approximate range check)
    # Standard normalization can push values negative, but usually within [-3, 3]
    assert (
        images.max() <= 5.0 and images.min() >= -5.0
    ), "Image normalization seems incorrect."
    print("Data Loading verified successfully.")

    # 3. Verify Model Architecture
    print("\n[3] Verifying Model Architecture...")
    model = get_model(
        pretrained=False
    )  # No need to download weights for architecture check
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(Config.DEVICE)

    with torch.no_grad():
        outputs = model(dummy_input)

    print(f"Model Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Expected output shape (2, {Config.NUM_CLASSES}), got {outputs.shape}"
    print("Model Architecture verified successfully.")

    # 4. Run Calibration Phase (Simulated)
    print("\n[4] Running Calibration Phase (1 Epoch, 2 Folds)...")
    # This runs the full cross-validation logic defined in library.calibration
    optimal_epochs = run_calibration(epochs=1)

    print(f"Calibration complete. Optimal Epochs: {optimal_epochs}")
    assert (
        isinstance(optimal_epochs, int) and optimal_epochs > 0
    ), "Optimal epochs should be a positive integer."

    # 5. Run Production Phase
    print("\n[5] Running Production Phase (Training Final Model)...")
    # This trains on the full dataset
    final_model = train_final_model(optimal_epochs=1)  # Force 1 epoch for speed

    # Verify model file creation
    expected_model_path = os.path.join(Config.OUTPUT_DIR, "final_model.pth")
    assert os.path.exists(
        expected_model_path
    ), f"Final model not found at {expected_model_path}"
    print(f"Final model verified at {expected_model_path}")

    # 6. Run Inference Phase
    print("\n[6] Running Inference Phase...")
    # This generates predictions on the test set
    predict_and_submit(model=final_model)

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    # Load submission and check format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {df_sub.shape}")

    # Check row count (Test set has 183 images)
    expected_rows = 183
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} rows in submission, got {len(df_sub)}"

    # Check columns
    expected_cols = ["image_id"] + Config.TARGET_COLS
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Check probability range
    prob_cols = Config.TARGET_COLS
    probs = df_sub[prob_cols].values
    assert (probs >= 0).all() and (
        probs <= 1
    ).all(), "Probabilities must be between 0 and 1."

    # Check if probabilities sum to 1 (Softmax was used)
    # Allow small floating point tolerance
    sums = probs.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-5), "Probabilities do not sum to 1.0"

    print("Inference and Submission verified successfully.")
    print("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    main()
