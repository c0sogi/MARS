import os
import torch
import pandas as pd
import numpy as np
import logging

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, probabilistic_f1
from library.data import get_dataloaders
from library.model import MCSINModel
from library.engine import run


def run_demo():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    print(">>> Setting up configuration for fast demonstration...")

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Small subset for rapid execution
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 2  # Reduce worker overhead
    Config.WORKING_DIR = "./working/demo_execution"
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_DIR = Config.WORKING_DIR
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # -------------------------------------------------------------------------
    # 2. Metric Verification
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Probabilistic F1 Metric...")

    # Case: Perfect prediction
    y_true = torch.tensor([1.0, 0.0, 1.0])
    y_pred_perfect = torch.tensor([1.0, 0.0, 1.0])
    score_perfect = probabilistic_f1(y_true, y_pred_perfect)

    # pTP = 1*1 + 0*0 + 1*1 = 2
    # pFP = (1-1)*1 + (1-0)*0 + (1-1)*1 = 0
    # Total Positives = 2
    # pPrecision = 2 / (2 + 0) = 1.0
    # pRecall = 2 / 2 = 1.0
    # pF1 = 2 * (1*1) / (1+1) = 1.0
    assert abs(score_perfect - 1.0) < 1e-5, f"Expected 1.0, got {score_perfect}"

    # Case: Partial prediction
    y_pred_partial = torch.tensor([0.8, 0.2, 0.6])
    score_partial = probabilistic_f1(y_true, y_pred_partial)

    # pTP = 1*0.8 + 0*0.2 + 1*0.6 = 1.4
    # pFP = 0*0.8 + 1*0.2 + 0*0.6 = 0.2
    # Total Positives = 2
    # pPrec = 1.4 / (1.4 + 0.2) = 1.4 / 1.6 = 0.875
    # pRec = 1.4 / 2 = 0.7
    # pF1 = 2 * (0.875 * 0.7) / (0.875 + 0.7) = 1.225 / 1.575 ≈ 0.7777...
    expected_score = 2 * (0.875 * 0.7) / (0.875 + 0.7)
    assert (
        abs(score_partial - expected_score) < 1e-5
    ), f"Expected {expected_score}, got {score_partial}"

    print("Metric verification passed.")

    # -------------------------------------------------------------------------
    # 3. Data Pipeline Verification
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Data Loading Pipeline...")

    train_loader, val_loader, test_loader = get_dataloaders()

    # Fetch one batch
    images, targets, pred_ids = next(iter(train_loader))

    print(f"Batch shapes - Images: {images.shape}, Targets: {targets.shape}")

    # Verify Image Tensor Shape: (Batch, 3, 640, 640)
    assert images.shape == (
        Config.BATCH_SIZE,
        Config.IN_CHANNELS,
        Config.IMG_SIZE[0],
        Config.IMG_SIZE[1],
    ), f"Incorrect image shape: {images.shape}"

    # Verify Target Shape
    assert targets.shape[0] == Config.BATCH_SIZE, "Target batch size mismatch"

    # Verify Data Types
    assert images.dtype == torch.float32, "Images should be float32"
    assert targets.dtype == torch.float32, "Targets should be float32"

    print("Data pipeline verification passed.")

    # -------------------------------------------------------------------------
    # 4. Model Verification
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Model Architecture...")

    model = MCSINModel(pretrained=False)  # No need to download weights for shape check
    model.to(device)
    model.eval()

    images = images.to(device)

    with torch.no_grad():
        logits = model(images)

    print(f"Logits shape: {logits.shape}")

    # Verify Output Shape: (Batch, 1)
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Incorrect output shape: {logits.shape}"

    print("Model verification passed.")

    # -------------------------------------------------------------------------
    # 5. Full Engine Execution
    # -------------------------------------------------------------------------
    print("\n>>> Running Full Engine (Train/Val/Test) on Debug Subset...")

    # Run the engine (this handles training loop, validation, and submission generation)
    # We use the provided library function directly
    run(epochs=Config.EPOCHS, debug=Config.DEBUG)

    print("Engine execution completed.")

    # -------------------------------------------------------------------------
    # 6. Submission Verification
    # -------------------------------------------------------------------------
    print("\n>>> Verifying Submission File...")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    print(f"Submission rows: {len(submission_df)}")
    print(f"Columns: {submission_df.columns.tolist()}")

    # Check columns
    assert "prediction_id" in submission_df.columns, "Missing 'prediction_id' column"
    assert "cancer" in submission_df.columns, "Missing 'cancer' column"

    # Check that we have rows
    assert len(submission_df) > 0, "Submission file is empty"

    # Check probability range
    probs = submission_df["cancer"].values
    assert np.all((probs >= 0) & (probs <= 1)), "Probabilities out of range [0, 1]"

    print("Submission verification passed.")
    print("\n>>> All demonstrations completed successfully.")


if __name__ == "__main__":
    run_demo()
