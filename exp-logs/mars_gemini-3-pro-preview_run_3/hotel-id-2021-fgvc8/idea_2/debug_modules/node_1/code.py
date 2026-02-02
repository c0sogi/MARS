import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, apk, mapk
from library.dataset import get_dataloaders
from library.model import HotelEfficientNet
from library.engine import train_loop, predict_and_submit


def run_demonstration():
    print("--- Starting Hotel ID Task Demonstration ---")

    # ---------------------------------------------------------
    # 1. Configuration Override for Speed and Demo
    # ---------------------------------------------------------
    print("\n[1] Overriding Configuration for Fast Demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 100  # Small subset for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.SAMPLES_PER_CLASS = 2  # 8 / 2 = 4 classes per batch
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.PATIENCE = 1

    # Ensure working directory is clean/ready
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    print("Configuration updated: DEBUG=True, EPOCHS=1, BATCH_SIZE=8")

    # ---------------------------------------------------------
    # 2. Verify Utility Functions
    # ---------------------------------------------------------
    print("\n[2] Verifying Metric Utilities...")
    # Test APK
    # Ground truth: 1, Predicted: [1, 2, 3, 4, 5] -> Score should be 1.0
    score_1 = apk([1], [1, 2, 3, 4, 5], k=5)
    assert score_1 == 1.0, f"APK calculation failed. Expected 1.0, got {score_1}"

    # Ground truth: 1, Predicted: [2, 3, 1, 4, 5] -> Score should be 1/3 = 0.333...
    score_2 = apk([1], [2, 3, 1, 4, 5], k=5)
    assert (
        abs(score_2 - 1 / 3) < 1e-6
    ), f"APK calculation failed. Expected ~0.333, got {score_2}"

    # Test MAPK
    map_score = mapk([[1], [1]], [[1, 2], [2, 3, 1]], k=5)
    expected_map = (1.0 + 1 / 3) / 2
    assert (
        abs(map_score - expected_map) < 1e-6
    ), f"MAPK calculation failed. Expected {expected_map}, got {map_score}"
    print("Metric utility verification passed.")

    # ---------------------------------------------------------
    # 3. Verify Data Loading Pipeline
    # ---------------------------------------------------------
    print("\n[3] Initializing DataLoaders (Debug Mode)...")
    train_loader, val_loader, test_loader, num_classes, unique_ids = get_dataloaders(
        debug=True
    )

    print(f"Number of classes in debug set: {num_classes}")
    print(f"Train batches: {len(train_loader)}")

    # Fetch one batch to verify shapes
    images, labels = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), "Incorrect image tensor shape"
    assert labels.shape == (Config.BATCH_SIZE,), "Incorrect label tensor shape"
    assert images.dtype == torch.float32, "Images should be float32"
    assert labels.dtype == torch.long, "Labels should be long (int64)"
    print("Data pipeline verification passed.")

    # ---------------------------------------------------------
    # 4. Verify Model Architecture
    # ---------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")
    # Instantiate model (pretrained=False for speed in this check)
    model = HotelEfficientNet(num_classes=num_classes, pretrained=False)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)

    # Forward pass (Inference mode: labels=None)
    with torch.no_grad():
        output = model(dummy_input, labels=None)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        Config.BATCH_SIZE,
        num_classes,
    ), "Model output shape mismatch"
    print("Model architecture verification passed.")

    # ---------------------------------------------------------
    # 5. Execute Training Loop
    # ---------------------------------------------------------
    print("\n[5] Executing Training Loop (1 Epoch)...")
    # Note: train_loop instantiates its own model internally using Config parameters.
    # It will use the num_classes we derived from the data loader.
    best_score = train_loop(train_loader, val_loader, num_classes)

    print(f"Training complete. Best Validation MAP@5: {best_score:.4f}")

    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "best_model.pth was not saved!"
    print(f"Model saved successfully at {best_model_path}")

    # ---------------------------------------------------------
    # 6. Execute Inference and Submission
    # ---------------------------------------------------------
    print("\n[6] Generating Submission...")
    predict_and_submit(test_loader, num_classes, unique_ids)

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "submission.csv was not generated!"

    # Verify submission content
    sub_df = pd.read_csv(submission_path)
    print(f"Submission shape: {sub_df.shape}")
    print("First 3 rows of submission:")
    print(sub_df.head(3))

    assert (
        "image" in sub_df.columns and "hotel_id" in sub_df.columns
    ), "Submission missing required columns"
    assert len(sub_df) > 0, "Submission file is empty"

    # Verify format of hotel_id prediction (space-delimited string)
    sample_pred = sub_df.iloc[0]["hotel_id"]
    assert isinstance(sample_pred, str), "Prediction should be a string"
    assert (
        len(sample_pred.split()) <= Config.TOP_K
    ), f"Prediction contains more than {Config.TOP_K} items"

    print("Submission verification passed.")
    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    run_demonstration()
