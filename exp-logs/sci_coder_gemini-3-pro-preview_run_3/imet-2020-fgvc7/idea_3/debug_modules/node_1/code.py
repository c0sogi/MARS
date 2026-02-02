import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, calculate_micro_f1
from library.dataset import get_dataloaders
from library.model import get_artwork_model
from library.loss import AsymmetricLoss
from library.train import run_training
from library.inference import run_inference


def main():
    print("=== Starting Artwork Attribute Labeling Demo ===\n")

    # --- 1. Configuration Setup ---
    print("1. Configuring environment for rapid demonstration...")
    # Override Config attributes for speed and isolation
    Config.debug = True  # Use small subset (1000 train, 500 val, 100 test)
    Config.epochs = 2  # Run only 2 epochs
    Config.batch_size = 16  # Small batch size
    Config.num_workers = 2  # Reduce workers overhead
    Config.pretrained = False  # Skip downloading ImageNet weights
    Config.working_dir = "./working/demo_run"
    Config.model_save_path = os.path.join(Config.working_dir, "demo_model.pth")
    Config.submission_dir = os.path.join(Config.working_dir, "submission")
    Config.submission_path = os.path.join(Config.submission_dir, "submission.csv")

    # Clean working directory if it exists
    if os.path.exists(Config.working_dir):
        shutil.rmtree(Config.working_dir)
    os.makedirs(Config.working_dir, exist_ok=True)
    os.makedirs(Config.submission_dir, exist_ok=True)

    seed_everything(Config.seed)
    print(f"   Debug Mode: {Config.debug}")
    print(f"   Device: {Config.device}")
    print(f"   Working Directory: {Config.working_dir}\n")

    # --- 2. Data Pipeline Verification ---
    print("2. Verifying Data Pipeline...")
    # Force load_cached_data=False to test raw CSV processing logic
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch a single batch
    images, targets = next(iter(train_loader))

    print(f"   Batch Image Shape: {images.shape}")
    print(f"   Batch Target Shape: {targets.shape}")

    # Assertions
    expected_img_shape = (Config.batch_size, 3, Config.img_size, Config.img_size)
    expected_target_shape = (Config.batch_size, Config.num_classes)

    assert (
        images.shape == expected_img_shape
    ), f"Image shape mismatch. Expected {expected_img_shape}, got {images.shape}"
    assert (
        targets.shape == expected_target_shape
    ), f"Target shape mismatch. Expected {expected_target_shape}, got {targets.shape}"
    assert targets.dtype == torch.float32, "Targets must be float32"
    print("   Data Pipeline verified successfully.\n")

    # --- 3. Model Architecture Verification ---
    print("3. Verifying Model Architecture...")
    model = get_artwork_model(num_classes=Config.num_classes, pretrained=False)
    model.to(Config.device)
    model.eval()

    # Run forward pass (inference)
    with torch.no_grad():
        # Move images to device
        input_images = images.to(Config.device)
        logits = model(input_images)

    print(f"   Logits Shape: {logits.shape}")
    assert logits.shape == expected_target_shape, "Model output shape mismatch"
    print("   Model Architecture verified successfully.\n")

    # --- 4. Loss Function Verification ---
    print("4. Verifying Loss Function (AsymmetricLoss)...")
    criterion = AsymmetricLoss()

    # Move targets to device
    target_gpu = targets.to(Config.device)
    loss = criterion(logits, target_gpu)

    print(f"   Calculated Loss: {loss.item():.6f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss must be non-negative"
    print("   Loss Function verified successfully.\n")

    # --- 5. Metric Verification ---
    print("5. Verifying Metric Calculation (Micro F1)...")
    # Synthetic ground truth: 2 samples, 3 classes
    y_true = np.array([[0, 1, 1], [1, 0, 0]])
    # Synthetic predictions (probabilities)
    y_pred_probs = np.array([[0.1, 0.9, 0.8], [0.8, 0.2, 0.1]])

    # With threshold 0.5, predictions become [[0, 1, 1], [1, 0, 0]] -> Perfect Match
    f1_perfect = calculate_micro_f1(y_pred_probs, y_true, threshold=0.5)

    # With threshold 0.85, predictions become [[0, 1, 0], [0, 0, 0]]
    # True Positives: 1 (class 1, sample 0)
    # False Positives: 0
    # False Negatives: 2 (class 2 sample 0, class 0 sample 1)
    # Micro F1 = 2*TP / (2*TP + FP + FN) = 2*1 / (2*1 + 0 + 2) = 2/4 = 0.5
    f1_partial = calculate_micro_f1(y_pred_probs, y_true, threshold=0.85)

    print(f"   Perfect Match F1: {f1_perfect}")
    print(f"   Partial Match F1: {f1_partial}")

    assert f1_perfect == 1.0, "F1 Score calculation incorrect for perfect match"
    assert f1_partial == 0.5, "F1 Score calculation incorrect for partial match"
    print("   Metric Calculation verified successfully.\n")

    # --- 6. Full Training Loop Execution ---
    print("6. Executing Training Loop (Debug Mode)...")
    # This calls library.train.run_training which uses the Config we modified
    best_val_f1 = run_training()

    print(f"   Training completed. Best Validation F1: {best_val_f1:.4f}")
    assert os.path.exists(
        Config.model_save_path
    ), f"Model checkpoint missing at {Config.model_save_path}"
    print("   Training Loop verified successfully.\n")

    # --- 7. Full Inference Execution ---
    print("7. Executing Inference Pipeline...")
    # This calls library.inference.run_inference
    # It loads the model we just trained, optimizes threshold, and generates submission
    run_inference()

    # Verify Submission
    assert os.path.exists(
        Config.submission_path
    ), f"Submission file missing at {Config.submission_path}"

    sub_df = pd.read_csv(Config.submission_path)
    print(f"   Submission File loaded. Shape: {sub_df.shape}")
    print(f"   Columns: {list(sub_df.columns)}")

    # In debug mode, test set is sliced to 100 samples
    assert (
        len(sub_df) == 100
    ), f"Expected 100 rows in debug submission, found {len(sub_df)}"
    assert (
        "id" in sub_df.columns and "attribute_ids" in sub_df.columns
    ), "Invalid submission format"

    # Check content of first row
    print(
        f"   First row example: ID={sub_df.iloc[0]['id']}, Attrs={sub_df.iloc[0]['attribute_ids']}"
    )

    print("   Inference Pipeline verified successfully.\n")

    print("=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
