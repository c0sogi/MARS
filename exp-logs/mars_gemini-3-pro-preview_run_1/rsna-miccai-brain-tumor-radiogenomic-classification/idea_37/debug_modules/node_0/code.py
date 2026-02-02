import os
import shutil
import pandas as pd
import torch
import numpy as np
import random

# Import from the provided library
from library.config import Config
from library.geometry_utils import process_subject_geometry
from library.dataset import get_dataloader, ARVSDataset
from library.model import ARVSNet
from library.engine import train_model
from library.inference import predict_test_set


def setup_demo_environment():
    """
    Sets up a temporary environment for the demo to avoid modifying
    the main working directory or processing the full dataset.
    """
    print(">>> Setting up demo environment...")

    # Define demo directories
    demo_dir = os.path.join(Config.WORKING_DIR, "demo_execution_custom")
    demo_cache_dir = os.path.join(demo_dir, "cache")
    demo_models_dir = os.path.join(demo_dir, "models")
    demo_submission_dir = os.path.join(demo_dir)

    # Clean up previous runs if they exist
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_cache_dir, exist_ok=True)
    os.makedirs(demo_models_dir, exist_ok=True)

    # Override Config to use these directories and speed up execution
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = demo_models_dir  # Store models here
    Config.SUBMISSION_DIR = demo_submission_dir
    Config.SUBMISSION_PATH = os.path.join(demo_submission_dir, "submission.csv")

    # Override Hyperparameters for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.PRETRAINED = False  # Skip downloading weights for speed/offline safety

    # Create Mini-Metadata files to avoid processing full dataset
    # 1. Mini Train
    full_train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    mini_train_df = full_train_df.head(10).copy()
    mini_train_path = os.path.join(demo_dir, "mini_train.csv")
    mini_train_df.to_csv(mini_train_path, index=False)
    Config.TRAIN_METADATA_PATH = mini_train_path

    # 2. Mini Val (Use same as train for demo purposes to ensure data exists)
    mini_val_path = os.path.join(demo_dir, "mini_val.csv")
    mini_train_df.to_csv(mini_val_path, index=False)
    Config.VAL_METADATA_PATH = mini_val_path

    # 3. Mini Test
    full_test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    mini_test_df = full_test_df.head(5).copy()
    mini_test_path = os.path.join(demo_dir, "mini_test.csv")
    mini_test_df.to_csv(mini_test_path, index=False)
    Config.TEST_METADATA_PATH = mini_test_path

    print(f"Demo environment configured in {demo_dir}")
    print(
        f"Using {len(mini_train_df)} training samples and {len(mini_test_df)} test samples."
    )


def demonstrate_geometry_processing():
    """
    Demonstrates and verifies the geometry calculation logic.
    """
    print("\n>>> Demonstrating Geometry Processing...")

    # Load the mini metadata
    df = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Process geometry (calculates CoM and slice indices)
    # We disable loading from cache to force calculation
    geometry_df = process_subject_geometry(df, load_cached_data=False)

    # Validation
    assert len(geometry_df) == len(df), "Geometry DF length mismatch"
    assert "BraTS21ID" in geometry_df.columns, "Missing ID column"

    # Check for expected index columns
    # Expect: flair_idx_0, flair_idx_1, flair_idx_2, t1wce_idx_0, ...
    expected_cols = []
    for mod in Config.MODALITIES:
        for i in range(len(Config.RELATIVE_OFFSETS)):
            expected_cols.append(f"{mod}_idx_{i}")

    for col in expected_cols:
        assert col in geometry_df.columns, f"Missing geometry column: {col}"
        # Ensure indices are integers
        assert pd.api.types.is_integer_dtype(
            geometry_df[col]
        ) or pd.api.types.is_float_dtype(
            geometry_df[col]
        ), f"Column {col} is not numeric"

    print("Geometry processing verified successfully.")
    print(f"Sample geometry row:\n{geometry_df.iloc[0].to_dict()}")


def demonstrate_dataset_and_loader():
    """
    Demonstrates and verifies Dataset and DataLoader.
    """
    print("\n>>> Demonstrating Dataset and DataLoader...")

    # Create DataLoader for training
    # This internally calls process_subject_geometry
    loader = get_dataloader(
        "train", batch_size=Config.BATCH_SIZE, num_workers=0, load_cached_geometry=False
    )

    # Fetch one batch
    images, labels = next(iter(loader))

    # Validation
    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    # Expected shape: (Batch, Channels, Height, Width)
    # Channels = 3 modalities * 3 offsets = 9
    expected_channels = len(Config.MODALITIES) * len(Config.RELATIVE_OFFSETS)
    assert images.shape == (
        Config.BATCH_SIZE,
        expected_channels,
        Config.IMAGE_SIZE[0],
        Config.IMAGE_SIZE[1],
    ), f"Incorrect image tensor shape. Expected {(Config.BATCH_SIZE, expected_channels, Config.IMAGE_SIZE[0], Config.IMAGE_SIZE[1])}, got {images.shape}"

    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect label tensor shape. Expected {(Config.BATCH_SIZE,)}, got {labels.shape}"

    print("Dataset and DataLoader verified successfully.")
    return images  # Return for model demo


def demonstrate_model(input_tensor):
    """
    Demonstrates and verifies the ARVSNet model.
    """
    print("\n>>> Demonstrating ARVSNet Model...")

    model = ARVSNet()

    # Move to configured device
    device = torch.device(Config.DEVICE)
    model.to(device)
    input_tensor = input_tensor.to(device)

    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)

    # Validation
    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Incorrect output shape. Expected {(Config.BATCH_SIZE, 1)}, got {output.shape}"

    # Verify input layer modification (Weight Inflation)
    # EfficientNet B0 stem usually has 32 output channels
    first_conv = model.model.conv_stem
    print(f"Input Conv Layer Weight Shape: {first_conv.weight.shape}")

    expected_in_channels = len(Config.MODALITIES) * len(Config.RELATIVE_OFFSETS)
    assert (
        first_conv.in_channels == expected_in_channels
    ), f"Model input channels mismatch. Expected {expected_in_channels}, got {first_conv.in_channels}"

    print("Model initialization and forward pass verified successfully.")


def demonstrate_training_engine():
    """
    Demonstrates the training loop using the engine.
    """
    print("\n>>> Demonstrating Training Engine...")

    # Run training for 1 epoch with max 2 batches per epoch to be fast
    # This will save 'best_model.pth' in Config.CACHE_DIR
    best_model_path = train_model(num_epochs=1, max_batches_per_epoch=2)

    assert os.path.exists(
        best_model_path
    ), f"Model checkpoint not found at {best_model_path}"
    print(f"Training verified. Model saved to {best_model_path}")


def demonstrate_inference():
    """
    Demonstrates the inference pipeline.
    """
    print("\n>>> Demonstrating Inference...")

    # Run inference on the mini test set
    # This loads the model saved in the previous step
    submission_df = predict_test_set(load_cached_geometry=False)

    # Validation
    assert isinstance(
        submission_df, pd.DataFrame
    ), "Inference did not return a DataFrame"
    assert "BraTS21ID" in submission_df.columns, "Submission missing BraTS21ID"
    assert "MGMT_value" in submission_df.columns, "Submission missing MGMT_value"
    assert (
        len(submission_df) == 5
    ), f"Expected 5 predictions (mini test), got {len(submission_df)}"

    # Check file existence
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not saved to disk"

    print("Inference verified successfully.")
    print(submission_df.head())


if __name__ == "__main__":
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    try:
        # 1. Setup
        setup_demo_environment()

        # 2. Geometry
        demonstrate_geometry_processing()

        # 3. Data Loading
        sample_batch = demonstrate_dataset_and_loader()

        # 4. Model
        demonstrate_model(sample_batch)

        # 5. Training
        demonstrate_training_engine()

        # 6. Inference
        demonstrate_inference()

        print("\n>>> All demonstrations completed successfully.")

    except AssertionError as e:
        print(f"\n!!! Validation Failed: {e}")
        exit(1)
    except Exception as e:
        print(f"\n!!! An error occurred: {e}")
        import traceback

        traceback.print_exc()
        exit(1)
