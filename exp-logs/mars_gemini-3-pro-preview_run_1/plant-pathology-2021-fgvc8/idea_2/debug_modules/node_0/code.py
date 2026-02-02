import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config, seed_everything
from library.dataset import AppleDataset, get_loaders, get_test_loader
from library.model import create_model
from library.trainer import Trainer
from library.utils import calculate_f1_score

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("--- Starting Library Demonstration ---")

    # 1. Modify Configuration for Speed and Demo Purposes
    print("\n[1] Configuring environment for rapid demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 32  # Small subset for quick execution
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for this small demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Re-seed to ensure changes take effect if needed
    seed_everything(Config.SEED)

    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Subset Size: {Config.DEBUG_SUBSET_SIZE}")
    print(f"    Epochs: {Config.EPOCHS}")
    print(f"    Working Dir: {Config.WORKING_DIR}")

    # 2. Verify Dataset and DataLoaders
    print("\n[2] Verifying Dataset and DataLoaders...")

    # Test Dataset instantiation
    train_dataset = AppleDataset(Config.TRAIN_METADATA, transform=None, mode="train")
    assert (
        len(train_dataset) == Config.DEBUG_SUBSET_SIZE
    ), f"Dataset length mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {len(train_dataset)}"

    # Test item retrieval
    img, target = train_dataset[0]
    assert isinstance(img, torch.Tensor), "Image should be a torch Tensor"
    assert isinstance(target, torch.Tensor), "Target should be a torch Tensor"
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Unexpected image shape: {img.shape}"
    assert target.shape == (
        Config.NUM_CLASSES,
    ), f"Unexpected target shape: {target.shape}"

    print("    Dataset item verification passed.")

    # Test Loaders
    train_loader, val_loader = get_loaders()

    # Fetch one batch from train_loader to test MixupCutmixCollate
    images, targets = next(iter(train_loader))

    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Batch image shape mismatch: {images.shape}"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Batch target shape mismatch: {targets.shape}"

    print("    DataLoader batch verification passed.")

    # 3. Verify Model Architecture
    print("\n[3] Verifying Model Architecture...")
    model = create_model(
        pretrained=False
    )  # False for speed, we just check architecture
    model.eval()

    with torch.no_grad():
        output = model(images.to(Config.DEVICE))

    assert output.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {output.shape}"

    print("    Model forward pass verification passed.")

    # 4. Verify Metric Calculation
    print("\n[4] Verifying Metric Calculation...")
    # Create synthetic predictions and targets
    # Preds: [0.8, 0.1], [0.2, 0.9] -> Binary (thresh 0.5): [1, 0], [0, 1]
    # Targets: [1, 0], [0, 1] -> Perfect match, F1 should be 1.0
    dummy_preds = torch.tensor([[2.0, -2.0], [-2.0, 2.0]])  # Logits
    dummy_targets = torch.tensor([[1, 0], [0, 1]])

    # Note: calculate_f1_score computes Macro F1
    score = calculate_f1_score(dummy_preds, dummy_targets, threshold=0.5)
    assert np.isclose(
        score, 1.0
    ), f"F1 Score calculation failed. Expected 1.0, got {score}"

    print("    Metric calculation verification passed.")

    # 5. Verify Trainer (Fit and Predict)
    print("\n[5] Executing Trainer Pipeline (Fit & Predict)...")
    trainer = Trainer()

    # Run training
    print("    Starting training (1 epoch)...")
    trainer.fit()

    # Check if model checkpoint was saved
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print("    Training complete. Checkpoint verified.")

    # Run inference
    print("    Starting inference...")
    trainer.predict()

    # Check if submission file was saved
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"
    print("    Inference complete. Submission file verified.")

    # 6. Verify Submission Format
    print("\n[6] Verifying Submission Format...")
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    expected_cols = ["image", "labels"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Got {df_sub.columns}"

    # Check length (should match DEBUG_SUBSET_SIZE because test loader also respects Config.DEBUG)
    assert (
        len(df_sub) == Config.DEBUG_SUBSET_SIZE
    ), f"Submission length mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {len(df_sub)}"

    # Check content (labels should be strings or NaN if empty, but code ensures string)
    # We check if the image IDs match the test metadata subset
    df_test_meta = pd.read_csv(Config.TEST_METADATA).head(Config.DEBUG_SUBSET_SIZE)
    assert df_sub["image"].equals(
        df_test_meta["image"]
    ), "Submission image IDs do not match test metadata order."

    print("    Submission format verification passed.")
    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
