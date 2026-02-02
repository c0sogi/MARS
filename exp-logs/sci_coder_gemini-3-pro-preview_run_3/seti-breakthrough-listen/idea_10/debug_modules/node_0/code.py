import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, pad_image, mixup_data, get_score
from library.dataset import SETIDataset
from library.model import SiameseEfficientNet
from library.engine import run_training, generate_submission


def create_mini_metadata():
    """
    Creates small subsets of the original metadata files to allow for
    rapid testing and demonstration of the pipeline.
    """
    print("Creating mini metadata files for demonstration...")
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # Read original metadata
    df_train = pd.read_csv("./metadata/train.csv")
    df_val = pd.read_csv("./metadata/val.csv")
    df_test = pd.read_csv("./metadata/test.csv")

    # Create subsets (e.g., 16 samples to fit a few batches)
    # We ensure we have at least one positive and one negative sample if possible
    df_train_mini = df_train.head(32)
    df_val_mini = df_val.head(16)
    df_test_mini = df_test.head(16)

    # Save to working directory
    train_mini_path = os.path.join(Config.WORK_DIR, "train_mini.csv")
    val_mini_path = os.path.join(Config.WORK_DIR, "val_mini.csv")
    test_mini_path = os.path.join(Config.WORK_DIR, "test_mini.csv")

    df_train_mini.to_csv(train_mini_path, index=False)
    df_val_mini.to_csv(val_mini_path, index=False)
    df_test_mini.to_csv(test_mini_path, index=False)

    return train_mini_path, val_mini_path, test_mini_path


def test_utils():
    """
    Verifies utility functions: padding, mixup, and scoring.
    """
    print("\n=== Testing Utils ===")

    # 1. Test pad_image
    # Original shape: (6, 273, 256)
    dummy_img = np.random.randn(6, 273, 256).astype(np.float32)
    padded_img = pad_image(dummy_img)

    # Expected shape: (6, 288, 256) based on Config.IMG_HEIGHT=288
    assert padded_img.shape == (
        6,
        288,
        256,
    ), f"pad_image failed: expected (6, 288, 256), got {padded_img.shape}"

    # Check padding content (bottom rows should be 0)
    # The original had 273 rows. Rows 273 to 287 should be 0.
    assert np.all(padded_img[:, 273:, :] == 0), "Padding values are not zero."
    print("pad_image: OK")

    # 2. Test mixup_data
    # Create dummy batch: (Batch=4, Channels=6, H=32, W=32)
    batch_size = 4
    x = torch.randn(batch_size, 6, 32, 32)
    y = torch.tensor([0.0, 1.0, 0.0, 1.0])

    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0, device="cpu")

    assert mixed_x.shape == x.shape, "Mixup output shape mismatch."
    assert y_a.shape == y.shape, "Mixup target A shape mismatch."
    assert y_b.shape == y.shape, "Mixup target B shape mismatch."
    assert 0.0 <= lam <= 1.0, "Mixup lambda out of range."
    print("mixup_data: OK")

    # 3. Test get_score (AUC)
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.35, 0.8])
    score = get_score(y_true, y_pred)
    assert 0.0 <= score <= 1.0, "AUC score out of range."
    print(f"get_score: OK (Score={score:.4f})")


def test_dataset(mini_meta_path):
    """
    Verifies the SETIDataset class.
    """
    print("\n=== Testing Dataset ===")

    # Initialize dataset with mini metadata
    dataset = SETIDataset(mini_meta_path, mode="train")

    assert len(dataset) > 0, "Dataset is empty."

    # Fetch one item
    inputs, target = dataset[0]
    on_target, off_target = inputs

    # Check return types
    assert isinstance(on_target, torch.Tensor), "On-target is not a tensor."
    assert isinstance(off_target, torch.Tensor), "Off-target is not a tensor."
    assert isinstance(target, torch.Tensor), "Target is not a tensor."

    # Check shapes
    # Config.IN_CHANNELS is 3 (A, C, E)
    # Config.IMG_HEIGHT is 288 (after padding)
    expected_shape = (3, Config.IMG_HEIGHT, Config.IMG_WIDTH)

    assert (
        on_target.shape == expected_shape
    ), f"On-target shape mismatch. Got {on_target.shape}, expected {expected_shape}"
    assert (
        off_target.shape == expected_shape
    ), f"Off-target shape mismatch. Got {off_target.shape}, expected {expected_shape}"

    print("SETIDataset: OK")
    return on_target.unsqueeze(0), off_target.unsqueeze(
        0
    )  # Return batch for model test


def test_model(sample_inputs):
    """
    Verifies the SiameseEfficientNet model.
    """
    print("\n=== Testing Model ===")

    device = torch.device("cpu")  # Use CPU for simple logic check
    model = SiameseEfficientNet().to(device)
    model.eval()

    on_input, off_input = sample_inputs

    # Forward pass
    with torch.no_grad():
        output = model((on_input, off_input))

    # Check output shape: (Batch_Size, 1)
    assert output.shape == (1, 1), f"Model output shape mismatch. Got {output.shape}"

    print("SiameseEfficientNet: OK")


def demonstrate_engine_training(train_path, val_path, test_path):
    """
    Demonstrates the training loop and inference using the engine.
    Overrides Config to run a very short cycle.
    """
    print("\n=== Demonstrating Training Engine ===")

    # --- Override Config for Demo ---
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.TRAIN_METADATA = train_path
    Config.VAL_METADATA = val_path
    Config.TEST_METADATA = test_path

    # Set a specific working directory for the demo
    Config.WORK_DIR = os.path.join("./working", "demo_run")
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    Config.MODEL_PATH = os.path.join(Config.WORK_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORK_DIR, "submission.csv")

    # Force CPU or GPU based on availability (usually handled by Config, but ensuring here)
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {Config.DEVICE}")

    # --- Run Training ---
    # We pass debug=False because we are already providing mini datasets via Config override.
    # If we passed debug=True, it would try to subset the dataset again, which is fine but unnecessary.
    print("Starting training run...")
    run_training(debug=False)

    # Verify model was saved
    assert os.path.exists(Config.MODEL_PATH), "Model file was not created."
    print("Training run completed. Model saved.")

    # --- Run Inference ---
    print("Generating submission...")
    generate_submission()

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "id" in df_sub.columns and "target" in df_sub.columns
    ), "Submission columns missing."
    assert (
        len(df_sub) == 16
    ), f"Submission length mismatch. Expected 16, got {len(df_sub)}"

    print("Inference completed. Submission generated.")
    print(df_sub.head())


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)

    # 2. Create Mini Data
    # This creates small CSVs in ./working pointing to real files in ./input
    train_csv, val_csv, test_csv = create_mini_metadata()

    # 3. Test Components
    test_utils()
    sample_batch = test_dataset(train_csv)
    test_model(sample_batch)

    # 4. Run Full Pipeline Demo
    demonstrate_engine_training(train_csv, val_csv, test_csv)

    print("\nAll demonstrations passed successfully.")
