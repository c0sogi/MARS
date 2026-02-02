import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import shutil

# Ensure the library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library import utils, data, model, train


def run_demonstration():
    print("============================================================")
    print("       Animal Classification Library Demonstration          ")
    print("============================================================")

    # -------------------------------------------------------------------------
    # 1. Patch Configuration for Fast Execution
    # -------------------------------------------------------------------------
    print("\n[Step 1] Patching Config for rapid prototyping...")

    # Enable debug mode to use a tiny subset of data
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 images for this demo

    # Reduce training duration
    Config.EPOCHS_STAGE1 = 1
    Config.EPOCHS_STAGE2 = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2  # Reduce worker overhead for small data

    # Disable pretrained weights to avoid downloading heavy files during demo
    Config.PRETRAINED = False

    print(f"   Debug Mode: {Config.DEBUG}")
    print(f"   Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"   Batch Size: {Config.BATCH_SIZE}")

    # -------------------------------------------------------------------------
    # 2. Verify Utility Functions
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Utility Functions...")

    # Test F1 Score Calculation
    y_true_dummy = [0, 1, 2, 0, 1, 2]
    y_pred_dummy = [0, 1, 2, 0, 1, 0]  # Introduce one error
    f1 = utils.calculate_macro_f1(y_true_dummy, y_pred_dummy)
    print(f"   Calculated Macro F1 (Dummy): {f1:.4f}")
    assert 0.0 <= f1 <= 1.0, "F1 score must be between 0 and 1"

    # Test Class Weight Computation
    # We check if it returns a tensor of the correct size
    if os.path.exists(Config.TRAIN_METADATA_PATH):
        print("   Computing class weights from metadata...")
        df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
        weights = utils.compute_class_weights(df_train)
        print(f"   Class Weights Shape: {weights.shape}")

        assert isinstance(weights, torch.Tensor), "Weights must be a torch Tensor"
        assert (
            weights.shape[0] == Config.NUM_CLASSES
        ), f"Expected {Config.NUM_CLASSES} weights, got {weights.shape[0]}"
    else:
        print("   Warning: Train metadata not found, skipping weight check.")

    # -------------------------------------------------------------------------
    # 3. Verify Data Loading Pipeline
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Data Loading Pipeline...")

    # Initialize DataLoaders with debug=True
    train_loader, val_loader, test_loader = data.get_dataloaders(debug=True)

    # Check Train Loader
    try:
        images, labels = next(iter(train_loader))
        print(f"   Train Batch - Images: {images.shape}, Labels: {labels.shape}")

        assert images.shape == (
            Config.BATCH_SIZE,
            3,
            Config.IMAGE_SIZE,
            Config.IMAGE_SIZE,
        ), "Incorrect train image batch shape"
        assert labels.shape == (Config.BATCH_SIZE,), "Incorrect train label batch shape"
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # Check Test Loader (returns image, id)
    try:
        test_images, test_ids = next(iter(test_loader))
        print(f"   Test Batch  - Images: {test_images.shape}, IDs: {len(test_ids)}")

        assert test_images.shape == (
            Config.BATCH_SIZE,
            3,
            Config.IMAGE_SIZE,
            Config.IMAGE_SIZE,
        ), "Incorrect test image batch shape"
        assert len(test_ids) == Config.BATCH_SIZE, "Incorrect number of test IDs"
    except StopIteration:
        raise AssertionError("Test loader is empty!")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[Step 4] Verifying Model Architecture...")

    # Initialize model (CPU for quick check)
    net = model.EfficientNetClassifier(num_classes=Config.NUM_CLASSES, pretrained=False)
    net.eval()

    # Create dummy input
    dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)

    # Forward pass
    with torch.no_grad():
        logits = net(dummy_input)

    print(f"   Input Shape: {dummy_input.shape}")
    print(f"   Output Logits Shape: {logits.shape}")

    assert logits.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Model output shape mismatch. Expected (2, {Config.NUM_CLASSES}), got {logits.shape}"

    # -------------------------------------------------------------------------
    # 5. Verify Training Loop (Trainer)
    # -------------------------------------------------------------------------
    print("\n[Step 5] Executing Full Training Cycle (Trainer)...")

    # Initialize Trainer
    # This will use the patched Config values (Debug=True, Epochs=1)
    trainer = train.Trainer(debug=True)

    # Run the fit method which orchestrates Stage 1, Stage 2, and Submission
    print("   Starting Trainer.fit()...")
    trainer.fit()

    print("   Training cycle completed.")

    # -------------------------------------------------------------------------
    # 6. Verify Submission Output
    # -------------------------------------------------------------------------
    print("\n[Step 6] Verifying Submission File...")

    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"   Submission File Found: {Config.SUBMISSION_PATH}")
        print(f"   Rows: {len(sub_df)}")
        print(f"   Columns: {list(sub_df.columns)}")

        assert len(sub_df) > 0, "Submission file is empty"
        assert "Id" in sub_df.columns, "Id column missing"
        assert "Predicted" in sub_df.columns, "Predicted column missing"

        # Check if predictions are valid integers
        assert pd.api.types.is_integer_dtype(
            sub_df["Predicted"]
        ), "Predicted column should contain integers"

        print("   Submission file is valid.")
    else:
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    print("\n============================================================")
    print("       Demonstration Completed Successfully                 ")
    print("============================================================")


if __name__ == "__main__":
    # Suppress warnings for cleaner output during demo
    warnings.filterwarnings("ignore")

    # Set global seed for reproducibility of the demo script itself
    utils.set_seed(42)

    run_demonstration()
