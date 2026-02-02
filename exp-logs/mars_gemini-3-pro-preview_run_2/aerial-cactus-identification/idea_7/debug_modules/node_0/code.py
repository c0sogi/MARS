import os
import sys
import shutil
import numpy as np
import torch
import pandas as pd

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, calculate_roc_auc
from library.dataset import process_data, CactusDataset, get_transforms
from library.model import CactusResUNet
from library.train import run_training
from library.inference import generate_submission


def main():
    print("Initializing Demonstration...")

    # 1. Configuration & Setup
    # Override Config for a fast demonstration run
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce compute load for demo
    Config.NUM_EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.SEEDS = [42]  # Only run one seed
    Config.DEBUG = True

    # Create directories
    Config.setup()

    # Set global seed
    set_seed(42)
    print(f"Configuration set. Working directory: {Config.WORKING_DIR}")

    # 2. Data Loading & Processing
    print("\n--- Step 1: Data Loading ---")
    # We force loading from metadata (ignoring any existing cache in the main working dir)
    # to demonstrate the loading logic.
    train_data_full, val_data_full, test_data_full = process_data(
        load_cached_data=False
    )

    train_imgs, train_lbls = train_data_full
    val_imgs, val_lbls = val_data_full
    test_imgs, test_ids = test_data_full

    print(f"Original Train shape: {train_imgs.shape}")
    print(f"Original Val shape:   {val_imgs.shape}")
    print(f"Original Test shape:  {test_imgs.shape}")

    # Verify data types and ranges
    assert train_imgs.dtype == np.uint8, "Images should be uint8"
    assert train_imgs.max() <= 255, "Pixel values should be <= 255"
    assert len(train_imgs) == len(train_lbls), "Mismatch in train images and labels"

    # Create a small subset for speed
    subset_size = 50
    train_subset = (train_imgs[:subset_size], train_lbls[:subset_size])
    val_subset = (val_imgs[:subset_size], val_lbls[:subset_size])
    test_subset = (test_imgs[:subset_size], test_ids[:subset_size])

    print(f"Subset created with {subset_size} samples for demo.")

    # 3. Dataset & Transforms Verification
    print("\n--- Step 2: Dataset & Transforms ---")
    # Instantiate dataset
    ds = CactusDataset(
        images=train_subset[0],
        labels=train_subset[1],
        transform=get_transforms("train"),
    )

    # Check __getitem__
    img_tensor, label_tensor = ds[0]

    # Verify Tensor shapes
    # Image: [C, H, W] -> [3, 32, 32]
    assert img_tensor.shape == (
        3,
        32,
        32,
    ), f"Unexpected image shape: {img_tensor.shape}"
    assert isinstance(img_tensor, torch.Tensor), "Output should be a torch Tensor"
    # Label: scalar tensor
    assert label_tensor.shape == (), f"Label should be scalar, got {label_tensor.shape}"

    print("Dataset verification passed.")

    # 4. Model Verification
    print("\n--- Step 3: Model Architecture ---")
    device = torch.device("cpu")  # Use CPU for simple shape check
    model = CactusResUNet().to(device)

    # Create dummy batch [Batch, Channel, Height, Width]
    dummy_input = torch.randn(4, 3, 32, 32).to(device)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    # Verify output shape: [Batch, 1] (binary classification logits)
    assert output.shape == (4, 1), f"Expected output shape (4, 1), got {output.shape}"
    print("Model forward pass successful.")

    # 5. Training Loop Demonstration
    print("\n--- Step 4: Training Loop ---")
    # Run training for the defined seed on the subset data
    # This will train for Config.NUM_EPOCHS (2)
    run_training(seed=42, train_data=train_subset, val_data=val_subset)

    # Verify model checkpoint was saved
    model_path = Config.get_model_path(42)
    if os.path.exists(model_path):
        print(f"Training complete. Model saved to: {model_path}")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    # 6. Inference & Submission
    print("\n--- Step 5: Inference & Submission ---")
    # Generate submission using the trained model and test subset
    generate_submission(test_subset)

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        df_sub = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission generated at: {Config.SUBMISSION_PATH}")
        print("Head of submission:")
        print(df_sub.head())

        # Verify format
        assert list(df_sub.columns) == [
            "id",
            "has_cactus",
        ], "Incorrect columns in submission"
        assert (
            len(df_sub) == subset_size
        ), f"Expected {subset_size} rows, got {len(df_sub)}"
        assert (
            df_sub["has_cactus"].min() >= 0.0 and df_sub["has_cactus"].max() <= 1.0
        ), "Probabilities out of range"
    else:
        raise FileNotFoundError("Submission file was not created.")

    # 7. Metric Utility Check
    print("\n--- Step 6: Metric Utility ---")
    # Test ROC AUC calculation
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    score = calculate_roc_auc(y_true, y_scores)
    print(f"Calculated ROC AUC for dummy data: {score}")
    assert 0 <= score <= 1, "ROC AUC score invalid"

    print("\n========================================")
    print("       DEMONSTRATION COMPLETE           ")
    print("========================================")


if __name__ == "__main__":
    # Ensure clean execution
    try:
        main()
    except Exception as e:
        print(f"\nERROR: Demonstration failed with exception: {e}")
        raise e
