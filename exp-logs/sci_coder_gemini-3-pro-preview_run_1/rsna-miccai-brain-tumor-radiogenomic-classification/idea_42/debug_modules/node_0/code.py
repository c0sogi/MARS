import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import library modules
from library import config, utils, preprocessing, dataset, model, train, inference


def create_mini_metadata():
    """
    Creates a small subset of the metadata to allow the demo to run quickly.
    Updates config paths to point to these new mini-files.
    """
    print("Creating mini-datasets for rapid demonstration...")

    # Load original metadata
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(config.VAL_METADATA_PATH)
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    # Sample subset (e.g., 10 samples for train/val, 5 for test)
    mini_train = df_train.head(10).copy()
    mini_val = df_val.head(5).copy()
    mini_test = df_test.head(5).copy()

    # Define paths for mini metadata
    mini_train_path = os.path.join(config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(config.WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(config.WORKING_DIR, "mini_test.csv")

    # Save mini metadata
    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    # Override config paths to use mini metadata
    config.TRAIN_METADATA_PATH = mini_train_path
    config.VAL_METADATA_PATH = mini_val_path
    config.TEST_METADATA_PATH = mini_test_path

    # Override cache filenames to avoid overwriting or loading real full data
    config.CACHE_TRAIN_IMAGES = "mini_train_images.npy"
    config.CACHE_TRAIN_LABELS = "mini_train_labels.npy"
    config.CACHE_VAL_IMAGES = "mini_val_images.npy"
    config.CACHE_VAL_LABELS = "mini_val_labels.npy"
    config.CACHE_TEST_IMAGES = "mini_test_images.npy"
    config.CACHE_TEST_IDS = "mini_test_ids.npy"

    print(f"Mini-datasets created at {config.WORKING_DIR}")


def verify_preprocessing():
    """
    Runs the preprocessing pipeline and asserts output shapes.
    """
    print("\n--- Verifying Preprocessing ---")

    # Force reload to process the mini dataset
    # load_cached_data=False ensures we process the DICOMs from scratch
    (train_data, val_data, test_data) = preprocessing.prepare_datasets(
        load_cached_data=False
    )

    train_X, train_y = train_data
    val_X, val_y = val_data
    test_X, test_ids = test_data

    # Assertions
    # Shape: (N, 224, 224, 3)
    assert train_X.ndim == 4, f"Train images should be 4D, got {train_X.ndim}"
    assert train_X.shape[1:] == (
        224,
        224,
        3,
    ), f"Unexpected image shape: {train_X.shape}"
    assert len(train_X) == 10, f"Expected 10 training samples, got {len(train_X)}"
    assert len(train_y) == 10, "Mismatch between training images and labels"

    assert val_X.shape[1:] == (224, 224, 3)
    assert len(val_X) == 5

    assert test_X.shape[1:] == (224, 224, 3)
    assert len(test_X) == 5

    print("Preprocessing verification passed. Data shapes are correct.")
    return train_data


def verify_dataset_and_loader(train_data):
    """
    Verifies the MGMTDataset and DataLoader functionality.
    """
    print("\n--- Verifying Dataset & DataLoader ---")
    train_X, train_y = train_data

    # Instantiate Dataset
    ds = dataset.MGMTDataset(
        train_X, train_y, transform=dataset.get_transforms("train")
    )

    # Check length
    assert len(ds) == 10

    # Instantiate DataLoader
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    # Fetch one batch
    images, targets = next(iter(dl))

    # Verify Batch Shapes
    # Images: (Batch, Channels, H, W) -> (4, 3, 224, 224)
    assert images.shape == (
        4,
        3,
        224,
        224,
    ), f"Batch image shape mismatch: {images.shape}"
    # Targets: (Batch,) -> (4,)
    assert targets.shape == (4,), f"Batch target shape mismatch: {targets.shape}"
    assert targets.dtype == torch.float32, "Targets should be float32"

    print("Dataset and DataLoader verification passed.")


def verify_model():
    """
    Verifies model instantiation and forward pass.
    """
    print("\n--- Verifying Model Architecture ---")

    # Instantiate Model
    net = model.MGMTNet(
        pretrained=False
    )  # False for speed, no download needed if cached

    # Create dummy input: (Batch=2, Channels=3, H=224, W=224)
    dummy_input = torch.randn(2, 3, 224, 224)

    # Forward pass
    net.eval()
    with torch.no_grad():
        output = net(dummy_input)

    # Check output shape: (Batch, Num_Classes) -> (2, 1)
    assert output.shape == (2, 1), f"Model output shape mismatch: {output.shape}"

    print("Model architecture verification passed.")


def verify_training_pipeline():
    """
    Runs a short training loop using the library's train module.
    """
    print("\n--- Verifying Training Pipeline ---")

    # Override config hyperparameters for speed
    # We want to run 2 folds, 1 epoch each

    # Note: train.run_kfold uses config constants internally for defaults,
    # but accepts arguments. We will pass arguments to override.

    try:
        train.run_kfold(
            num_folds=2, epochs=1, batch_size=4, learning_rate=1e-3, patience=1
        )
    except Exception as e:
        print(f"Training pipeline failed with error: {e}")
        raise e

    # Check if models were saved
    expected_model_0 = os.path.join(config.WORKING_DIR, "best_model_fold0.pth")
    expected_model_1 = os.path.join(config.WORKING_DIR, "best_model_fold1.pth")

    assert os.path.exists(expected_model_0), "Model for fold 0 was not saved."
    assert os.path.exists(expected_model_1), "Model for fold 1 was not saved."

    print("Training pipeline verification passed.")


def verify_inference():
    """
    Runs the inference pipeline and checks submission file.
    """
    print("\n--- Verifying Inference Pipeline ---")

    # Define output path for demo submission
    demo_submission_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")

    # Run inference
    # Note: This will load the models saved in the previous step
    inference.predict(
        model_dir=config.WORKING_DIR,
        output_path=demo_submission_path,
        batch_size=4,
        device=config.DEVICE,
    )

    # Verify Submission File
    assert os.path.exists(demo_submission_path), "Submission file was not created."

    df = pd.read_csv(demo_submission_path)

    # Check columns
    assert "BraTS21ID" in df.columns
    assert "MGMT_value" in df.columns

    # Check length (should match mini_test size = 5)
    assert len(df) == 5, f"Submission length mismatch. Expected 5, got {len(df)}"

    # Check values are probabilities
    assert df["MGMT_value"].min() >= 0.0
    assert df["MGMT_value"].max() <= 1.0

    print("Inference verification passed.")
    print(f"Submission generated at: {demo_submission_path}")
    print(df)


if __name__ == "__main__":
    # Ensure reproducibility
    utils.seed_everything(42)

    # 1. Setup Mini Data
    create_mini_metadata()

    # 2. Verify Preprocessing
    train_data = verify_preprocessing()

    # 3. Verify Dataset
    verify_dataset_and_loader(train_data)

    # 4. Verify Model
    verify_model()

    # 5. Verify Training
    verify_training_pipeline()

    # 6. Verify Inference
    verify_inference()

    print("\nAll verifications passed successfully!")
