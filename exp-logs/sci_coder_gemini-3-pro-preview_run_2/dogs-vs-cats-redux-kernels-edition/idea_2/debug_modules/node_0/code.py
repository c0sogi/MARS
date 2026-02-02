import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.dataset import DogCatDataset, get_fold_dataloaders, get_test_dataloader
from library.model import EfficientNetClassifier
from library.engine import train_kfold, generate_submission
from library.utils import seed_everything


def create_mini_metadata():
    """
    Creates smaller versions of the metadata CSVs to allow for rapid testing
    of the pipeline without processing the entire dataset.
    """
    print("Creating mini-datasets for rapid demonstration...")

    # Load original metadata
    train_df = pd.read_csv("./metadata/train.csv")
    val_df = pd.read_csv("./metadata/val.csv")
    test_df = pd.read_csv("./metadata/test.csv")

    # Sample a small subset (ensuring class balance for StratifiedKFold)
    # We take 50 samples from train and 20 from val to ensure enough data for 2 folds
    mini_train = (
        train_df.groupby("label").apply(lambda x: x.head(25)).reset_index(drop=True)
    )
    mini_val = (
        val_df.groupby("label").apply(lambda x: x.head(10)).reset_index(drop=True)
    )
    mini_test = test_df.head(20)

    # Save to working directory
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    return mini_train_path, mini_val_path, mini_test_path


def configure_for_demo(train_csv, val_csv, test_csv):
    """
    Overrides Config parameters to optimize for speed during demonstration.
    """
    print("Overriding configuration for speed...")

    # Point to mini datasets
    Config.TRAIN_METADATA_CSV = train_csv
    Config.VAL_METADATA_CSV = val_csv
    Config.TEST_METADATA_CSV = test_csv

    # Reduce training intensity
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.N_FOLDS = 2  # Run only 2 folds
    Config.BATCH_SIZE = 8  # Small batch size for the small dataset
    Config.NUM_WORKERS = 2  # Reduce workers overhead

    # Ensure checkpoint directory is clean/ready
    if not os.path.exists(Config.CHECKPOINT_DIR):
        os.makedirs(Config.CHECKPOINT_DIR)


def test_dataset_logic():
    """
    Validates the DogCatDataset and DataLoader logic.
    """
    print("\nTesting Dataset and DataLoader...")

    # Test loading a specific fold
    train_loader, val_loader = get_fold_dataloaders(fold_idx=0, n_folds=Config.N_FOLDS)

    # Fetch one batch
    images, labels = next(iter(train_loader))

    # Verify shapes
    # Images: (Batch_Size, 3, 224, 224)
    expected_img_shape = (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    assert (
        images.shape == expected_img_shape
    ), f"Image batch shape mismatch. Expected {expected_img_shape}, got {images.shape}"

    # Labels: (Batch_Size)
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Label batch shape mismatch. Expected {(Config.BATCH_SIZE,)}, got {labels.shape}"

    print(
        f"  Batch loaded successfully. Image shape: {images.shape}, Label shape: {labels.shape}"
    )

    # Test Test-Loader
    test_loader = get_test_dataloader()
    test_images, test_ids = next(iter(test_loader))

    assert test_images.shape == expected_img_shape, "Test image shape mismatch"
    assert test_ids.shape == (Config.BATCH_SIZE,), "Test ID shape mismatch"
    print("  Test DataLoader verified.")


def test_model_logic():
    """
    Validates the EfficientNetClassifier architecture.
    """
    print("\nTesting Model Architecture...")

    model = EfficientNetClassifier(
        pretrained=False
    )  # No need to download weights for shape check
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE)

    with torch.no_grad():
        output = model(dummy_input)

    # Expected output: (2, 1) since num_classes=1
    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"

    print(f"  Model forward pass successful. Output shape: {output.shape}")


def run_training_pipeline():
    """
    Runs the training engine (train_kfold) with the reduced configuration.
    """
    print("\nRunning Training Pipeline (Mini-KFold)...")

    # This calls the engine's train_kfold which trains, validates, and saves checkpoints
    train_kfold(n_folds=Config.N_FOLDS, epochs=Config.EPOCHS)

    # Verify checkpoints were created
    for fold in range(Config.N_FOLDS):
        ckpt_path = os.path.join(Config.CHECKPOINT_DIR, f"fold_{fold}.pth")
        assert os.path.exists(
            ckpt_path
        ), f"Checkpoint for fold {fold} was not created at {ckpt_path}"

    print("  Training completed and checkpoints verified.")


def run_inference_pipeline():
    """
    Runs the inference engine (generate_submission).
    """
    print("\nRunning Inference Pipeline...")

    generate_submission()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created."

    df = pd.read_csv(Config.SUBMISSION_FILE)

    # Check columns
    assert (
        "id" in df.columns and "label" in df.columns
    ), "Submission file missing required columns."

    # Check row count (should match mini_test size created earlier)
    # We created mini_test with 20 rows
    expected_rows = 20
    assert (
        len(df) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df)}"

    # Check value ranges
    assert (
        df["label"].min() >= 0.0 and df["label"].max() <= 1.0
    ), "Probabilities out of range [0, 1]."

    print(f"  Submission generated successfully at {Config.SUBMISSION_FILE}")
    print(f"  First 3 rows:\n{df.head(3)}")


if __name__ == "__main__":
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # 1. Setup Data
    train_csv, val_csv, test_csv = create_mini_metadata()

    # 2. Configure
    configure_for_demo(train_csv, val_csv, test_csv)

    # 3. Verify Components
    test_dataset_logic()
    test_model_logic()

    # 4. Execute Pipeline
    run_training_pipeline()
    run_inference_pipeline()

    print("\nAll demonstration steps completed successfully.")
