import os
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import Config
from library.dataset import get_dataloaders
from library.model import FrozenResNetLinear
from library.engine import train_model, predict_and_submit

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demonstration():
    print("Starting Library Demonstration...")

    # ==========================================
    # 1. Configuration Setup for Demo
    # ==========================================
    print("\n[1] Setting up Configuration for Rapid Execution...")

    # Modify Config for speed (override defaults)
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Only use 50 images
    Config.BATCH_SIZE = 8
    Config.NUM_EPOCHS = 1  # Only 1 epoch
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny dataset

    # Redirect outputs to a demo directory to keep things clean
    demo_dir = "./working/demo_run"
    os.makedirs(demo_dir, exist_ok=True)
    Config.MODEL_PATH = os.path.join(demo_dir, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "demo_submission.csv")

    Config.print_config()

    # ==========================================
    # 2. Dataset & DataLoader Verification
    # ==========================================
    print("\n[2] Verifying Dataset and DataLoaders...")

    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Check batch structure
    images, metadata, targets = next(iter(train_loader))

    print(f"  Batch Image Shape: {images.shape}")
    print(f"  Batch Metadata Shape: {metadata.shape}")
    print(f"  Batch Targets Shape: {targets.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Image tensor shape mismatch"
    assert metadata.shape == (
        Config.BATCH_SIZE,
        12,
    ), "Metadata tensor shape mismatch (expected 12 features)"
    assert targets.shape == (Config.BATCH_SIZE,), "Target tensor shape mismatch"

    # Check normalization (approximate range check)
    # ImageNet normalization results in values roughly between -2 and 2
    assert (
        images.max() <= 3.0 and images.min() >= -3.0
    ), "Image normalization seems incorrect"

    print("  Dataset verification passed.")

    # ==========================================
    # 3. Model Architecture Verification
    # ==========================================
    print("\n[3] Verifying Model Architecture...")

    device = torch.device(Config.DEVICE)
    model = FrozenResNetLinear().to(device)

    # Move batch to device
    images = images.to(device)
    metadata = metadata.to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(images, metadata)

    print(f"  Model Output Shape: {outputs.shape}")

    # Assertions
    assert outputs.shape == (
        Config.BATCH_SIZE,
        1,
    ), "Model output shape mismatch (expected [Batch, 1])"

    # Check if backbone is frozen
    backbone_param = next(model.backbone.parameters())
    assert backbone_param.requires_grad is False, "ResNet backbone should be frozen"

    # Check if head is trainable
    head_param = next(model.head.parameters())
    assert head_param.requires_grad is True, "Linear head should be trainable"

    print("  Model verification passed.")

    # ==========================================
    # 4. Training Engine Verification
    # ==========================================
    print("\n[4] Running Training Loop (1 Epoch)...")

    # Run training using the engine function
    # This will use the modified Config (1 epoch, debug subset)
    best_rmse = train_model(debug=True)

    print(f"  Training finished. Best RMSE: {best_rmse:.4f}")

    # Verify model artifact creation
    assert os.path.exists(
        Config.MODEL_PATH
    ), f"Model file was not created at {Config.MODEL_PATH}"

    print("  Training engine verification passed.")

    # ==========================================
    # 5. Inference & Submission Verification
    # ==========================================
    print("\n[5] Running Inference and Generating Submission...")

    predict_and_submit(debug=True)

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file was not created at {Config.SUBMISSION_PATH}"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"  Submission Shape: {df_sub.shape}")
    print("  Submission Head:")
    print(df_sub.head(3))

    # Assertions
    # In debug mode, test set is subset to DEBUG_SUBSET_SIZE
    assert (
        len(df_sub) == Config.DEBUG_SUBSET_SIZE
    ), f"Submission row count mismatch. Expected {Config.DEBUG_SUBSET_SIZE}, got {len(df_sub)}"

    assert (
        "Id" in df_sub.columns and "Pawpularity" in df_sub.columns
    ), "Submission columns mismatch"

    # Check value range
    preds = df_sub["Pawpularity"].values
    assert np.all(preds >= 1.0) and np.all(
        preds <= 100.0
    ), "Predictions contain values outside valid range [1, 100]"

    print("  Inference verification passed.")

    print("\nDemonstration Completed Successfully.")


if __name__ == "__main__":
    run_demonstration()
