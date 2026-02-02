import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Import classes and functions from the provided library files
from library.config import Config
from library.utils import set_seed
from library.dataset import AnimalDataset, get_transforms
from library.model import create_model
from library.loss import ClassBalancedFocalLoss
import importlib
import library.trainer

importlib.reload(library.trainer)
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demonstration of Animal Classification Library ===\n")

    # ---------------------------------------------------------
    # 1. Configuration Setup for Demonstration
    # ---------------------------------------------------------
    print("[1/6] Configuring environment for rapid demonstration...")

    # Modify Config attributes to run a fast, minimal version of the pipeline.
    # We override defaults to ensure execution finishes within seconds/minutes.
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8  # Small batch size for speed
    Config.NUM_WORKERS = 2  # Reduce multiprocessing overhead
    Config.PRETRAINED = (
        False  # Disable weight download to ensure offline execution safety
    )
    Config.USE_EMA = False  # Disable EMA to simplify the graph for this quick test

    # Redirect outputs to a demo directory
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    Config.BEST_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.EMA_MODEL_PATH = os.path.join(Config.WORKING_DIR, "ema_model.pth")

    # Clean up previous demo runs if any
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    Config.setup()

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Batch Size: {Config.BATCH_SIZE}, Epochs: {Config.EPOCHS}")

    # ---------------------------------------------------------
    # 2. Dataset and Transforms Verification
    # ---------------------------------------------------------
    print("\n[2/6] Verifying Dataset and Transforms...")

    # Verify metadata existence
    assert os.path.exists(Config.TRAIN_META_PATH), "Train metadata file is missing."

    # Initialize Dataset
    train_transform = get_transforms(mode="train")
    train_dataset = AnimalDataset(Config.TRAIN_META_PATH, transform=train_transform)

    # Check dataset length
    print(f"Training Dataset Length: {len(train_dataset)}")
    assert len(train_dataset) > 0, "Dataset should not be empty."

    # Fetch a single sample to verify loading and transforms
    sample_img, sample_target = train_dataset[0]

    print(f"Sample Image Shape: {sample_img.shape}")  # Should be [3, 224, 224]
    print(f"Sample Target: {sample_target} (Type: {type(sample_target)})")

    # Assertions
    assert sample_img.shape == (
        3,
        224,
        224,
    ), f"Expected shape (3, 224, 224), got {sample_img.shape}"
    assert isinstance(sample_target, torch.Tensor), "Target should be a torch.Tensor"
    assert sample_img.dtype == torch.float32, "Image tensor should be float32"

    # ---------------------------------------------------------
    # 3. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n[3/6] Verifying Model Architecture...")

    device = Config.get_device()
    print(f"Device: {device}")

    # Create model (randomly initialized due to Config.PRETRAINED = False)
    model = create_model(num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED)
    model.to(device)
    model.eval()

    # Create a dummy batch
    dummy_input = torch.randn(2, 3, 224, 224).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Assertions
    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Expected output shape (2, {Config.NUM_CLASSES}), got {output.shape}"
    assert not torch.isnan(output).any(), "Model output contains NaNs"

    # ---------------------------------------------------------
    # 4. Loss Function Verification
    # ---------------------------------------------------------
    print("\n[4/6] Verifying Class Balanced Focal Loss...")

    # Initialize Loss
    criterion = ClassBalancedFocalLoss(beta=Config.CLASS_BETA, gamma=Config.FOCAL_GAMMA)
    criterion.to(device)

    # Create dummy logits and targets
    # Batch size 4, Num classes 23
    dummy_logits = torch.randn(4, Config.NUM_CLASSES).to(device)
    dummy_targets = torch.tensor([0, 5, 10, 22]).to(
        device
    )  # Random valid class indices

    # Compute loss
    loss_val = criterion(dummy_logits, dummy_targets)

    print(f"Calculated Loss: {loss_val.item():.4f}")

    # Assertions
    assert loss_val.dim() == 0, "Loss should be a scalar"
    assert loss_val.item() > 0, "Loss should be positive"

    # ---------------------------------------------------------
    # 5. Trainer Integration (Training & Validation Loop)
    # ---------------------------------------------------------
    print("\n[5/6] Running Trainer (Debug Mode)...")
    print(
        "Initializing Trainer with debug=True to limit batches (approx 10 batches)..."
    )

    # Initialize Trainer
    # debug=True ensures the loops break early (after ~10 batches) to save time
    trainer = Trainer(debug=True)

    # Run Training
    # This will run for 1 epoch (truncated by debug mode)
    trainer.fit(epochs=Config.EPOCHS)

    # Verify Best Model was saved
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not created."
    print("Training cycle completed. Best model saved.")

    # ---------------------------------------------------------
    # 6. Inference and Submission Generation
    # ---------------------------------------------------------
    print("\n[6/6] Generating Predictions on Test Set...")

    # Run Inference
    # This uses the saved best model to predict on the test set
    trainer.predict_test_set()

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission CSV was not created."

    # Load and inspect submission
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File Created: {Config.SUBMISSION_PATH}")
    print(f"Number of Predictions: {len(df_sub)}")
    print("First 3 rows:")
    print(df_sub.head(3))

    # Assertions on submission content
    assert (
        "Id" in df_sub.columns and "Predicted" in df_sub.columns
    ), "Submission columns missing"
    assert len(df_sub) > 0, "Submission dataframe is empty"
    # Check data types
    assert pd.api.types.is_integer_dtype(
        df_sub["Predicted"]
    ), "Predictions should be integers"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
