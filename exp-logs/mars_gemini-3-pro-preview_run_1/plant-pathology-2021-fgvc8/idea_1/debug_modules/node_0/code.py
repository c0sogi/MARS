import os
import sys
import torch
import numpy as np
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_label_map, calculate_score
from library.data_loader import get_dataloaders
from library.network import AppleDiseaseModel
from library.trainer import Trainer


def run_pipeline_demonstration():
    print("=== Starting Apple Disease Detection Pipeline Demonstration ===")

    # 1. Setup Reproducibility
    seed_everything(Config.SEED)
    print(f"Random seed set to {Config.SEED}")

    # 2. Define Demo Hyperparameters (Optimized for Speed)
    DEMO_EPOCHS = 1
    DEMO_BATCH_SIZE = 4
    DEMO_DEBUG = True  # Uses only 100 samples per dataset

    print(
        f"Configuration: Device={Config.DEVICE} | Epochs={DEMO_EPOCHS} | Batch Size={DEMO_BATCH_SIZE} | Debug={DEMO_DEBUG}"
    )

    # 3. Verify Utility Functions
    print("\n[1/5] Verifying Utility Functions...")

    # Test Label Mapping
    str2int, int2str = get_label_map()
    assert (
        len(str2int) == Config.NUM_CLASSES
    ), f"Expected {Config.NUM_CLASSES} classes, found {len(str2int)}"
    assert "healthy" in str2int, "Class 'healthy' missing from label map."

    # Test Score Calculation (F1 Macro)
    # Scenario: 2 samples, 3 classes. Sample 1: Perfect match. Sample 2: Complete miss.
    # Note: calculate_score expects (N, C) inputs.
    y_true_dummy = np.array([[1, 0, 1], [0, 1, 0]])
    y_pred_dummy = np.array([[0.9, 0.1, 0.8], [0.8, 0.2, 0.8]])  # Threshold is 0.5
    # Sample 1 Pred: [1, 0, 1] (Match), Sample 2 Pred: [1, 0, 1] (Miss: True is [0, 1, 0])
    # This is just a structural test, not a math test for sklearn, but we ensure it runs without error.
    score = calculate_score(y_true_dummy, y_pred_dummy, threshold=0.5)
    assert 0.0 <= score <= 1.0, "Score calculation returned value out of range [0, 1]"
    print("Utils verification passed.")

    # 4. Verify Data Loading
    print("\n[2/5] Verifying Data Loaders...")

    # Force load_cached_data=False to ensure processing logic runs
    train_loader, val_loader, test_loader = get_dataloaders(
        train_batch_size=DEMO_BATCH_SIZE,
        val_batch_size=DEMO_BATCH_SIZE,
        num_workers=2,  # Reduced workers for demo script stability
        load_cached_data=False,
        debug=DEMO_DEBUG,
    )

    # Fetch one batch to verify shapes
    images, targets = next(iter(train_loader))

    # Assertions
    # Image: (B, 3, 256, 256)
    expected_img_shape = (DEMO_BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert (
        images.shape == expected_img_shape
    ), f"Image shape mismatch. Expected {expected_img_shape}, got {images.shape}"

    # Target: (B, 6)
    expected_target_shape = (DEMO_BATCH_SIZE, Config.NUM_CLASSES)
    assert (
        targets.shape == expected_target_shape
    ), f"Target shape mismatch. Expected {expected_target_shape}, got {targets.shape}"

    print(f"Data Loader verification passed. Batch shape: {images.shape}")

    # 5. Verify Model Architecture
    print("\n[3/5] Verifying Model Architecture...")

    model = AppleDiseaseModel(pretrained=True)
    model.to(Config.DEVICE)

    # Perform forward pass
    images = images.to(Config.DEVICE)
    logits = model(images)

    # Check output dimensions
    assert (
        logits.shape == expected_target_shape
    ), f"Model output shape mismatch. Expected {expected_target_shape}, got {logits.shape}"
    print("Model forward pass verification passed.")

    # 6. Verify Training Loop
    print("\n[4/5] Verifying Training Loop...")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=Config.DEVICE,
        learning_rate=1e-4,
        num_epochs=DEMO_EPOCHS,  # Override config for speed
        patience=1,
    )

    # Run training
    trainer.fit()

    # Check if model checkpoint was created
    assert os.path.exists(
        trainer.best_model_path
    ), f"Best model not found at {trainer.best_model_path}"
    print("Training loop verification passed.")

    # 7. Verify Inference and Submission
    print("\n[5/5] Verifying Inference & Submission...")

    trainer.predict(test_loader)

    # Check submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not created at {Config.SUBMISSION_PATH}"
        )

    # Validate content format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    assert "image" in df_sub.columns, "Submission missing 'image' column"
    assert "labels" in df_sub.columns, "Submission missing 'labels' column"

    # Check row count (Debug mode uses 100 samples)
    # Note: test_loader in debug mode has 100 samples
    assert (
        len(df_sub) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} predictions, got {len(df_sub)}"

    # Check label format (should be string, even if empty or single class)
    assert pd.api.types.is_string_dtype(
        df_sub["labels"]
    ) or pd.api.types.is_object_dtype(
        df_sub["labels"]
    ), "Labels column is not string/object type"

    print(f"Submission verification passed. File saved to {Config.SUBMISSION_PATH}")
    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_pipeline_demonstration()
