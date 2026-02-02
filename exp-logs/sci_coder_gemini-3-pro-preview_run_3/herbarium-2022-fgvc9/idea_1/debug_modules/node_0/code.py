import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Add the current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, compute_metrics
from library.dataset import get_dataloaders
from library.model import PlantClassifier
from library.trainer import Trainer
from library.inference import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting Library Usage Demonstration...")

    # -------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Set seed for reproducibility
    set_seed(42)

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Very small subset for demo
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_EPOCHS = 1  # Single epoch
    Config.NUM_WORKERS = 2  # Reduce worker overhead

    # Ensure working directories exist
    Config.setup()

    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Device: {Config.DEVICE}")

    # -------------------------------------------------------------------------
    # 2. Dataset and DataLoader Verification
    # -------------------------------------------------------------------------
    print("\n[2] Verifying DataLoaders...")

    train_loader, val_loader, test_loader = get_dataloaders(
        train_batch_size=Config.BATCH_SIZE,
        val_batch_size=Config.BATCH_SIZE,
        debug=Config.DEBUG,
        debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    )

    # Verify Train Loader
    try:
        images, labels = next(iter(train_loader))
        print(f"    Train Batch - Images: {images.shape}, Labels: {labels.shape}")

        # Assertions
        assert images.shape == (
            Config.BATCH_SIZE,
            3,
            Config.IMAGE_SIZE,
            Config.IMAGE_SIZE,
        ), f"Expected image shape {(Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)}, got {images.shape}"
        assert labels.shape == (
            Config.BATCH_SIZE,
        ), f"Expected label shape {(Config.BATCH_SIZE,)}, got {labels.shape}"
        assert labels.dtype == torch.long, "Labels should be of type torch.long"
    except StopIteration:
        raise AssertionError("Train loader is empty!")

    # Verify Test Loader (returns images and IDs)
    try:
        test_images, test_ids = next(iter(test_loader))
        print(f"    Test Batch  - Images: {test_images.shape}, IDs: {len(test_ids)}")
        assert len(test_ids) == Config.BATCH_SIZE, "Mismatch in test batch IDs length"
    except StopIteration:
        raise AssertionError("Test loader is empty!")

    print("    DataLoaders verified successfully.")

    # -------------------------------------------------------------------------
    # 3. Model Architecture Verification
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Model Architecture...")

    model = PlantClassifier(num_classes=Config.NUM_CLASSES)
    model.to(Config.DEVICE)
    model.eval()

    # Check output layer dimensions
    # MobileNetV3 classifier is a Sequential block. The last layer is replaced in PlantClassifier.
    # We check if the last linear layer has the correct out_features.
    last_layer = model.model.classifier[-1]
    print(f"    Final Layer: {last_layer}")
    assert isinstance(last_layer, torch.nn.Linear), "Last layer should be Linear"
    assert (
        last_layer.out_features == Config.NUM_CLASSES
    ), f"Expected {Config.NUM_CLASSES} output features, got {last_layer.out_features}"

    # Verify Forward Pass
    dummy_input = torch.randn(2, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE).to(
        Config.DEVICE
    )
    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Forward Pass Output Shape: {output.shape}")
    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), "Output shape mismatch during forward pass"

    print("    Model architecture verified successfully.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Demonstrating Training Loop (Trainer)...")

    trainer = Trainer(device=Config.DEVICE)

    # Run fit (1 epoch, small subset)
    trainer.fit(train_loader, val_loader, epochs=Config.NUM_EPOCHS)

    # Verify checkpoint creation
    checkpoint_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    assert os.path.exists(
        checkpoint_path
    ), f"Checkpoint file not found at {checkpoint_path}"
    print(f"    Training complete. Checkpoint saved at: {checkpoint_path}")

    # -------------------------------------------------------------------------
    # 5. Inference and Submission Verification
    # -------------------------------------------------------------------------
    print("\n[5] Demonstrating Inference and Submission...")

    # Run inference using the generated checkpoint
    generate_submission(
        checkpoint_path=checkpoint_path,
        batch_size=Config.BATCH_SIZE,
        device=Config.DEVICE,
        debug=Config.DEBUG,
    )

    # Verify submission file
    submission_path = Config.SUBMISSION_PATH
    assert os.path.exists(
        submission_path
    ), f"Submission file not found at {submission_path}"

    df_sub = pd.read_csv(submission_path)
    print(f"    Submission File Loaded. Shape: {df_sub.shape}")
    print(f"    Columns: {df_sub.columns.tolist()}")

    # Validate format
    assert (
        "Id" in df_sub.columns and "Predicted" in df_sub.columns
    ), "Missing required columns in submission"
    assert len(df_sub) > 0, "Submission file is empty"
    assert (
        df_sub["Id"].dtype == "int64" or df_sub["Id"].dtype == "int32"
    ), "Id column should be integer"
    assert (
        df_sub["Predicted"].dtype == "int64" or df_sub["Predicted"].dtype == "int32"
    ), "Predicted column should be integer"

    print("    Inference pipeline verified successfully.")

    # -------------------------------------------------------------------------
    # 6. Metric Calculation Verification
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Metric Calculation...")

    # Create dummy data
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 2, 0, 1, 1])

    # Calculate Macro F1
    score = compute_metrics(y_true, y_pred)
    print(f"    Dummy Macro F1 Score: {score:.4f}")

    assert isinstance(score, float), "Metric should be a float"
    assert 0.0 <= score <= 1.0, "F1 score must be between 0 and 1"

    print("    Metric calculation verified successfully.")

    print("\nAll demonstrations and verifications passed successfully!")


if __name__ == "__main__":
    main()
