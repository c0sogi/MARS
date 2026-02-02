import os
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Import library components
from library.config import Config
from library.data_processing import prepare_data
from library.dataset import get_dataloaders
from library.model import CKResNet
from library.utils import FocalLoss
from library.train import run_training
from library.inference import predict


def run_demo():
    print("Starting NFL Contact Detection Library Demo...")

    # ==========================================
    # 1. Configuration Setup for Demo
    # ==========================================
    print("\n[1] Setting up configuration for fast execution...")

    # Override Config paths to use a dedicated demo directory
    # Note: Since Config paths are static, we must update dependent paths manually
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    Config.TRAIN_FEATURES_CACHE = os.path.join(
        Config.WORKING_DIR, "train_features.parquet"
    )
    Config.VAL_FEATURES_CACHE = os.path.join(Config.WORKING_DIR, "val_features.parquet")
    Config.TEST_FEATURES_CACHE = os.path.join(
        Config.WORKING_DIR, "test_features.parquet"
    )

    Config.SCALER_PATH = os.path.join(Config.WORKING_DIR, "scaler.joblib")
    Config.MODEL_CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Override hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 32
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.THRESHOLD_STEPS = 10  # Reduce grid search steps

    # Clean up previous demo run if exists
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.setup_directories()

    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Epochs: {Config.EPOCHS}, Batch Size: {Config.BATCH_SIZE}")

    # ==========================================
    # 2. Data Processing Verification
    # ==========================================
    print("\n[2] Verifying Data Processing (prepare_data)...")

    # Run data preparation in debug mode (subsamples data)
    # load_cached_data=False forces processing from raw CSVs
    train_data, val_data, test_data = prepare_data(load_cached_data=False, debug=True)

    # Verify Cache Files Created
    assert os.path.exists(
        Config.TRAIN_FEATURES_CACHE
    ), "Train features parquet not created."
    assert os.path.exists(Config.SCALER_PATH), "Scaler joblib not created."

    # Verify Tensor Structure: ((X_wide, X_center, condition), targets)
    # Check Train Data
    inputs, targets = train_data
    x_wide, x_center, condition = inputs

    print(
        f"Train Tensors - Wide: {x_wide.shape}, Center: {x_center.shape}, Cond: {condition.shape}, Targets: {targets.shape}"
    )

    assert isinstance(x_wide, torch.Tensor)
    assert x_wide.ndim == 2
    assert condition.shape[1] == Config.FILM_DIM
    assert targets.ndim == 1

    print("Data processing verification passed.")

    # ==========================================
    # 3. Model & Loss Logic Verification
    # ==========================================
    print("\n[3] Verifying Model and Loss Logic...")

    # Get DataLoaders
    train_loader, _, _ = get_dataloaders(
        load_cached_data=True, debug=True, batch_size=Config.BATCH_SIZE, num_workers=0
    )

    # Fetch one batch
    batch_inputs, batch_targets = next(iter(train_loader))
    b_wide, b_center, b_cond = batch_inputs

    input_dim = b_wide.shape[1]
    center_dim = b_center.shape[1]

    # Instantiate Model
    model = CKResNet(input_dim=input_dim, center_dim=center_dim)
    model.eval()

    # Forward Pass
    with torch.no_grad():
        logits = model(b_wide, b_center, b_cond)

    print(f"Logits Shape: {logits.shape}")
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected output shape ({Config.BATCH_SIZE}, 1), got {logits.shape}"

    # Loss Calculation
    criterion = FocalLoss()
    # Targets need to be shaped (Batch, 1) for BCE
    loss = criterion(logits, batch_targets.unsqueeze(1))

    print(f"Calculated Loss: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"

    print("Model and Loss verification passed.")

    # ==========================================
    # 4. Training Pipeline Verification
    # ==========================================
    print("\n[4] Running Training Loop (run_training)...")

    # Run training (debug=True uses the subset processed earlier)
    trained_model, best_threshold = run_training(debug=True, load_cached_data=True)

    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), "Model checkpoint file not found after training."
    assert 0.0 <= best_threshold <= 1.0, f"Invalid threshold: {best_threshold}"

    print(f"Training complete. Best Threshold: {best_threshold}")

    # ==========================================
    # 5. Inference Pipeline Verification
    # ==========================================
    print("\n[5] Running Inference (predict)...")

    # Run prediction
    df_submission = predict(threshold=best_threshold, debug=True, load_cached_data=True)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    # Verify DataFrame Structure
    assert "contact_id" in df_submission.columns
    assert "contact" in df_submission.columns
    assert (
        df_submission["contact"].isin([0, 1]).all()
    ), "Predictions must be binary (0 or 1)."

    print(f"Inference complete. Submission shape: {df_submission.shape}")
    print(df_submission.head())

    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set fixed seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    run_demo()
