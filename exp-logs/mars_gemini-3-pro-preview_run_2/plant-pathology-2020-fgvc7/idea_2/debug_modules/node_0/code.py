import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# Import the library modules
# We import Config first to modify it before other modules use it
from library.config import Config

# -------------------------------------------------------------------------
# 1. Configuration Override for Speed and Demonstration
# -------------------------------------------------------------------------
print("=== 1. Configuring Environment for Demo ===")
# Enable debug mode to use a tiny subset of data
Config.DEBUG = True
Config.DEBUG_SAMPLE_SIZE = 32  # Small sample for quick execution

# Reduce training parameters for speed
Config.EPOCHS = 1
Config.BATCH_SIZE = 8
Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script
Config.PRETRAINED = False  # Skip downloading heavy weights for this demo
Config.IMAGE_SIZE = 224  # Smaller image size for faster processing

# Ensure clean state
if os.path.exists(Config.WORKING_DIR):
    shutil.rmtree(Config.WORKING_DIR)
os.makedirs(Config.WORKING_DIR, exist_ok=True)

# Now import the rest of the library which will use the modified Config
from library.data import get_loaders, get_test_loader
from library.model import AppleEfficientNet
from library.train import run_training
from library.inference import run_inference
from library.utils import calculate_metric, seed_everything

# Set seed for reproducibility
seed_everything(Config.SEED)

if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # 2. Data Loading Verification
    # -------------------------------------------------------------------------
    print("\n=== 2. Verifying Data Loading ===")

    # Initialize loaders (load_cached_data=False forces fresh processing)
    train_loader, val_loader = get_loaders(load_cached_data=False)

    print(f"Train Loader Length: {len(train_loader)}")
    print(f"Val Loader Length: {len(val_loader)}")

    # Fetch one batch
    images, labels = next(iter(train_loader))

    print(f"Batch Images Shape: {images.shape}")
    print(f"Batch Labels Shape: {labels.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Expected image shape {(Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)}, got {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Expected label shape {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {labels.shape}"
    assert images.dtype == torch.float32, "Images should be float32"
    assert labels.dtype == torch.float32, "Labels should be float32"

    print("Data loading verification passed.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n=== 3. Verifying Model Architecture ===")

    device = torch.device(Config.DEVICE)
    model = AppleEfficientNet(
        model_name=Config.MODEL_NAME,
        pretrained=False,  # Speed up initialization
        num_classes=Config.NUM_CLASSES,
    )
    model.to(device)

    # Forward pass check
    dummy_input = images.to(device)
    with torch.no_grad():
        logits = model(dummy_input)

    print(f"Logits Shape: {logits.shape}")

    # Assertions
    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch."
    assert not torch.isnan(logits).any(), "Model output contains NaNs."

    print("Model architecture verification passed.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Simulation
    # -------------------------------------------------------------------------
    print("\n=== 4. Running Training Loop (Demo) ===")

    # This will run for 1 epoch on the debug subset
    # It should save 'best_model.pth' to ./working
    try:
        run_training(load_cached_data=False)
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise e

    # Verify artifact creation
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), f"Best model not found at {Config.BEST_MODEL_PATH}"

    print("Training loop completed successfully. Model saved.")

    # -------------------------------------------------------------------------
    # 5. Inference Simulation
    # -------------------------------------------------------------------------
    print("\n=== 5. Running Inference (Demo) ===")

    # This will load the model saved above and predict on test set
    # Note: get_test_loader loads the full test set (183 images), which is fast enough.
    try:
        run_inference(load_cached_data=False)
    except Exception as e:
        print(f"Inference failed with error: {e}")
        raise e

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {submission_df.shape}")
    print("Submission Head:")
    print(submission_df.head(3))

    # Verify Submission Structure
    expected_cols = ["image_id"] + Config.CLASS_LABELS
    assert (
        list(submission_df.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(submission_df.columns)}"

    # Verify values are probabilities (roughly)
    # Since we used an untrained model (pretrained=False) for only 1 epoch,
    # predictions might be garbage, but they should be valid floats.
    numeric_cols = Config.CLASS_LABELS
    assert (
        submission_df[numeric_cols].values >= 0
    ).all(), "Probabilities should be non-negative"
    assert (
        submission_df[numeric_cols].values <= 1.0 + 1e-5
    ).all(), "Probabilities should be <= 1"

    print("Inference verification passed.")

    # -------------------------------------------------------------------------
    # 6. Metric Calculation Verification
    # -------------------------------------------------------------------------
    print("\n=== 6. Verifying Metric Calculation ===")

    # Create synthetic ground truth (one-hot)
    y_true = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # Case 1: Perfect predictions
    y_pred_perfect = np.array(
        [
            [0.9, 0.05, 0.02, 0.03],
            [0.05, 0.9, 0.02, 0.03],
            [0.05, 0.02, 0.9, 0.03],
            [0.05, 0.02, 0.03, 0.9],
        ]
    )

    score_perfect = calculate_metric(y_true, y_pred_perfect)
    print(f"Perfect Score: {score_perfect}")
    assert score_perfect == 1.0, "Perfect predictions should yield AUC 1.0"

    # Case 2: Random/Bad predictions
    y_pred_bad = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
            [0.25, 0.25, 0.25, 0.25],
        ]
    )

    score_bad = calculate_metric(y_true, y_pred_bad)
    print(f"Random Score: {score_bad}")
    # AUC for random guessing is 0.5
    assert np.isclose(
        score_bad, 0.5, atol=0.05
    ), "Random predictions should yield AUC ~0.5"

    print("Metric verification passed.")

    print("\n=== All Demonstrations Completed Successfully ===")
