import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config, seed_everything
from library.dataset import get_dataloaders
from library.model import CactusResNet
from library.utils import calculate_roc_auc
from library.train import run_training
from library.predict import run_prediction


def main():
    print("==== Starting Cactus Identification Library Demo ====")

    # 1. Setup & Configuration Override for Speed
    print("\n[1] Configuring environment for fast demonstration...")
    seed_everything(Config.SEED)

    # Override Config for a quick run
    Config.DEBUG = True
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 8  # Small batch size for debug
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small debug run

    # Set up a specific working directory for this demo
    demo_dir = "./working/demo_execution"
    Config.WORKING_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")
    checkpoint_path = os.path.join(demo_dir, "best_model.pth")

    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    print(f"Working directory set to: {Config.WORKING_DIR}")

    # 2. Validate Data Loading
    print("\n[2] Validating Data Loading...")
    dataloaders = get_dataloaders(debug=True)
    train_loader = dataloaders["train"]

    # Fetch one batch
    images, labels = next(iter(train_loader))

    # Check shapes
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        32,
        32,
    ), f"Expected image shape {(Config.BATCH_SIZE, 3, 32, 32)}, got {images.shape}"
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Expected label shape {(Config.BATCH_SIZE,)}, got {labels.shape}"
    assert images.dtype == torch.float32, "Images should be float32"
    assert labels.dtype == torch.float32, "Labels should be float32"

    print("Data loading validation passed.")

    # 3. Validate Model Architecture
    print("\n[3] Validating Model Architecture...")
    device = torch.device("cpu")  # Use CPU for simple shape check
    model = CactusResNet(num_classes=Config.NUM_CLASSES).to(device)
    model.eval()

    with torch.no_grad():
        output = model(images.to(device))

    print(f"Model Output Shape: {output.shape}")

    # Output should be (Batch_Size, Num_Classes) -> (8, 1)
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Expected output shape {(Config.BATCH_SIZE, 1)}, got {output.shape}"

    print("Model architecture validation passed.")

    # 4. Validate Metric Calculation
    print("\n[4] Validating Metric Utility...")
    # Synthetic ground truth and predictions
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])

    # Expected AUC:
    # Pairs: (0, 0.1), (0, 0.4), (1, 0.35), (1, 0.8)
    # Negative examples: 0.1, 0.4
    # Positive examples: 0.35, 0.8
    # Comparisons:
    # 0.35 > 0.1 (Win)
    # 0.35 < 0.4 (Loss)
    # 0.8 > 0.1 (Win)
    # 0.8 > 0.4 (Win)
    # Total Wins = 3, Total Comparisons = 4 => AUC = 0.75

    auc = calculate_roc_auc(y_true, y_pred)
    print(f"Calculated AUC: {auc}")

    assert np.isclose(auc, 0.75), f"Expected AUC 0.75, got {auc}"
    print("Metric utility validation passed.")

    # 5. Execute Training Pipeline
    print("\n[5] Executing Training Pipeline (Debug Mode)...")

    # run_training handles the loop, validation, and saving
    trained_model = run_training(debug=True, save_checkpoint_path=checkpoint_path)

    # Verify checkpoint creation
    assert os.path.exists(
        checkpoint_path
    ), f"Checkpoint file was not created at {checkpoint_path}"

    print("Training pipeline completed successfully.")

    # 6. Execute Prediction Pipeline
    print("\n[6] Executing Prediction Pipeline...")

    # Run inference using the checkpoint generated in step 5
    run_prediction(checkpoint_path=checkpoint_path, debug=True)

    # Verify submission creation
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file was not created at {Config.SUBMISSION_PATH}"

    print("Prediction pipeline completed successfully.")

    # 7. Validate Submission Content
    print("\n[7] Validating Submission File...")
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    print(f"Submission Head:\n{submission_df.head()}")
    print(f"Submission Shape: {submission_df.shape}")

    # Check columns
    assert list(submission_df.columns) == [
        "id",
        "has_cactus",
    ], "Submission columns should be ['id', 'has_cactus']"

    # Check value range
    preds = submission_df["has_cactus"]
    assert (
        preds.min() >= 0.0 and preds.max() <= 1.0
    ), "Predictions should be probabilities between 0 and 1"

    # Check ID format (should be filenames ending in .jpg)
    sample_id = submission_df.iloc[0]["id"]
    assert str(sample_id).endswith(
        ".jpg"
    ), f"ID should be a filename ending in .jpg, got {sample_id}"

    print("Submission file validation passed.")

    print("\n==== Demo Completed Successfully ====")


if __name__ == "__main__":
    main()
