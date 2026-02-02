import os
import sys
import pandas as pd
import numpy as np
import torch
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, load_middle_slice
from library.dataset import MGMTDataset
from library.model import MGMTClassifier
from library.train import run_training
from library.predict import generate_submission


def create_mini_metadata(original_path, save_path, n_samples=10):
    """
    Creates a subset of the metadata file for quick demonstration.
    """
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original metadata not found: {original_path}")

    df = pd.read_csv(original_path)
    # Take a small subset
    df_subset = df.head(n_samples).copy()
    df_subset.to_csv(save_path, index=False)
    print(f"Created mini metadata at {save_path} with {len(df_subset)} samples.")
    return len(df_subset)


def verify_dataset_logic(mini_train_path):
    """
    Verifies that MGMTDataset loads and transforms data correctly.
    """
    print("\n=== Verifying Dataset Logic ===")

    # Instantiate dataset with a unique split name to avoid cache conflicts during this check
    dataset = MGMTDataset(
        metadata_path=mini_train_path,
        split="demo_check",
        transform=None,  # Get raw tensors first (or use default ToTensorV2)
        load_cached_data=False,
    )

    # Check length
    assert len(dataset) > 0, "Dataset should not be empty."
    print(f"Dataset length: {len(dataset)}")

    # Check item structure
    img, target, subject_id = dataset[0]

    # Check Image Shape: (Channels, Height, Width) -> (3, 224, 224)
    # Note: Albumentations ToTensorV2 converts (H, W, C) to (C, H, W)
    print(f"Sample image shape: {img.shape}")
    assert img.shape == (
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    ), f"Expected shape (3, {Config.IMG_SIZE}, {Config.IMG_SIZE}), got {img.shape}"

    # Check Target Shape
    # It returns a tensor scalar
    print(f"Sample target: {target}")
    assert isinstance(target, torch.Tensor), "Target should be a tensor."

    print("Dataset verification successful.")


def verify_model_logic():
    """
    Verifies that the model instantiates and accepts input correctly.
    """
    print("\n=== Verifying Model Logic ===")

    model = MGMTClassifier(
        model_name="efficientnet_b0", pretrained=False, num_classes=1  # Speed up init
    )
    model.eval()

    # Create dummy batch: (Batch_Size, Channels, H, W)
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, Config.IMG_SIZE, Config.IMG_SIZE)

    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model output shape: {output.shape}")

    # Assertions
    assert output.shape == (
        batch_size,
        1,
    ), f"Expected output shape ({batch_size}, 1), got {output.shape}"

    print("Model verification successful.")


def main():
    # 1. Setup and Reproducibility
    seed_everything(42)

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Create Mini Datasets for Speed
    # We define paths for our mini metadata
    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test.csv")

    # Create the files
    n_train = create_mini_metadata(
        Config.TRAIN_METADATA_PATH, mini_train_path, n_samples=8
    )
    n_val = create_mini_metadata(Config.VAL_METADATA_PATH, mini_val_path, n_samples=4)
    n_test = create_mini_metadata(
        Config.TEST_METADATA_PATH, mini_test_path, n_samples=4
    )

    # 3. Override Config to use Mini Datasets and Fast Training Settings
    # We modify the class attributes directly
    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path
    Config.TEST_METADATA_PATH = mini_test_path

    Config.EPOCHS = 2
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # 4. Verify Components
    verify_dataset_logic(mini_train_path)
    verify_model_logic()

    # 5. Run Training
    print("\n=== Running Training Demo ===")
    # We set load_cached_data=False to ensure it processes our new mini metadata
    # instead of looking for existing cache files from a previous full run.
    best_auc = run_training(
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,
    )

    # Verify model checkpoint exists
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(
            f"Training finished but model not found at {Config.MODEL_PATH}"
        )
    print(f"Training successful. Best AUC: {best_auc}")

    # 6. Run Inference
    print("\n=== Running Inference Demo ===")
    df_submission = generate_submission(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,
    )

    # Verify Submission
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    assert (
        len(df_submission) == n_test
    ), f"Submission rows {len(df_submission)} mismatch expected test size {n_test}"

    assert (
        "BraTS21ID" in df_submission.columns and "MGMT_value" in df_submission.columns
    ), "Submission columns mismatch."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
