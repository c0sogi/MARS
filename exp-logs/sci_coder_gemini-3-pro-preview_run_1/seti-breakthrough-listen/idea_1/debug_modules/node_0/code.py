import sys
import os
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

# Add current directory to path to ensure library imports work correctly
sys.path.append(".")

# Import library components
from library.config import Config
from library.utils import set_seed, calculate_roc_auc
from library.data import get_dataloaders
from library.model import SpatialDifferenceCNN
from library.train import run_training
from library.inference import predict_test_set

if __name__ == "__main__":
    print(">>> Starting SETI Technosignature Detection Library Demo...")

    # 1. Configuration Setup for Demonstration
    # We override default config values to ensure the script runs quickly and uses minimal resources.
    Config.DEBUG = True  # Use a small subset of data (100 samples)
    Config.BATCH_SIZE = 4  # Small batch size for verification
    Config.MAX_EPOCHS = 1  # Train for only 1 epoch
    Config.PATIENCE = 1  # Minimal patience
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for this quick test

    # Initialize directories
    Config.setup()

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print("Configuration configured for fast debug execution.")

    # 2. Data Pipeline Verification
    print("\n>>> Verifying Data Loading...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS, debug=Config.DEBUG
    )

    # Fetch a single batch from the training loader
    try:
        images, targets = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("DataLoader is empty. Check metadata or debug settings.")

    # Verify Data Shapes
    # Expected Image Shape: (Batch, Channels, Freq, Time) -> (4, 1, 273, 256)
    # Expected Target Shape: (Batch,) -> (4,)
    print(f"Fetched Batch - Images: {images.shape}, Targets: {targets.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        1,
        273,
        256,
    ), f"Image batch shape mismatch. Expected {(Config.BATCH_SIZE, 1, 273, 256)}, got {images.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
    ), f"Target batch shape mismatch. Expected {(Config.BATCH_SIZE,)}, got {targets.shape}"
    assert images.dtype == torch.float32, "Image tensor dtype should be float32"

    # 3. Model Logic Verification
    print("\n>>> Verifying Model Architecture...")
    device = torch.device(Config.DEVICE)
    model = SpatialDifferenceCNN().to(device)

    # Perform a forward pass
    images = images.to(device)
    logits = model(images)

    # Verify Output Shape
    # Expected Logits Shape: (Batch, 1)
    print(f"Model Output (Logits) Shape: {logits.shape}")
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {logits.shape}"

    # 4. Training Loop Execution
    print("\n>>> Running Training Simulation (1 Epoch)...")
    # This function handles the training loop, validation, and model saving
    run_training(debug=True, epochs=1, patience=1)

    # Verify that the model artifact was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model file was not saved at {Config.MODEL_SAVE_PATH}"
    print("Training simulation complete. Model artifact verified.")

    # 5. Inference & Submission Verification
    print("\n>>> Running Inference on Test Set...")
    # This function loads the saved model and generates a submission file
    submission_df = predict_test_set(debug=True)

    # Verify Submission File Existence
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file was not saved at {Config.SUBMISSION_PATH}"

    # Verify Submission Content
    print(f"Submission Shape: {submission_df.shape}")
    assert "id" in submission_df.columns, "Submission missing 'id' column"
    assert "target" in submission_df.columns, "Submission missing 'target' column"
    assert len(submission_df) > 0, "Submission DataFrame is empty"

    # Verify values are probabilities (0 to 1)
    preds = submission_df["target"].values
    assert np.all(
        (preds >= 0) & (preds <= 1)
    ), "Predictions contain values outside [0, 1]"

    # 6. Metric Utility Verification
    print("\n>>> Verifying Metric Calculation (ROC AUC)...")
    # Create synthetic ground truth and predictions
    # We include both classes (0 and 1) to ensure AUC can be calculated
    y_true_dummy = torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32)
    y_pred_dummy = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.4], dtype=torch.float32)

    auc_score = calculate_roc_auc(y_true_dummy, y_pred_dummy)
    print(f"Calculated Dummy AUC: {auc_score:.4f}")

    assert isinstance(auc_score, float), "AUC score should be a float"
    assert 0.0 <= auc_score <= 1.0, "AUC score must be between 0 and 1"
    # Given the dummy data, AUC should be 1.0 (perfect separation)
    assert (
        auc_score == 1.0
    ), "AUC calculation logic seems incorrect for perfect predictions"

    print("\n>>> All demonstrations and verifications passed successfully!")
