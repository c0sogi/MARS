import os
import torch
import pandas as pd
import numpy as np
import random

# Import from the provided library files
from library.config import Config
from library.dataset import SETIDataset
from library.model import BaselineCNN
from library.engine import run_training, generate_submission


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_dataset_logic():
    print("\n=== Testing Dataset Logic ===")

    # Load the training metadata directly to pass to the Dataset
    # We use a small sample for verification
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(
            f"Metadata file not found: {Config.TRAIN_METADATA_PATH}"
        )

    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH).head(10)

    # Instantiate dataset
    dataset = SETIDataset(df_train, input_dir=Config.INPUT_DIR)

    # Check length
    assert (
        len(dataset) == 10
    ), f"Dataset length mismatch. Expected 10, got {len(dataset)}"

    # Fetch first item
    image_tensor, target_tensor = dataset[0]

    # Verify Image Tensor
    # Expected shape: (6, 273, 256)
    expected_shape = (6, 273, 256)
    assert (
        image_tensor.shape == expected_shape
    ), f"Image tensor shape mismatch. Expected {expected_shape}, got {image_tensor.shape}"
    assert (
        image_tensor.dtype == torch.float32
    ), f"Image tensor dtype mismatch. Expected float32, got {image_tensor.dtype}"

    # Verify Target Tensor
    # Expected shape: (1,)
    assert target_tensor.shape == (
        1,
    ), f"Target tensor shape mismatch. Expected (1,), got {target_tensor.shape}"
    assert (
        target_tensor.dtype == torch.float32
    ), f"Target tensor dtype mismatch. Expected float32, got {target_tensor.dtype}"

    print("Dataset logic verified successfully.")


def test_model_logic():
    print("\n=== Testing Model Logic ===")

    device = torch.device("cpu")  # Use CPU for simple logic verification
    model = BaselineCNN().to(device)
    model.eval()

    # Create a dummy batch of size 2
    # Shape: (Batch, Channels, Freq, Time) -> (2, 6, 273, 256)
    dummy_input = torch.randn(2, 6, 273, 256).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    # Expected output shape: (Batch, Num_Classes) -> (2, 1)
    expected_shape = (2, 1)
    assert (
        output.shape == expected_shape
    ), f"Model output shape mismatch. Expected {expected_shape}, got {output.shape}"

    print("Model logic verified successfully.")


def run_pipeline_demo():
    print("\n=== Running Training & Inference Pipeline Demo ===")

    # Define fast parameters for demonstration
    demo_epochs = 1
    demo_batch_size = 4
    demo_debug_size = 20  # Small subset

    # 1. Run Training
    print("Starting training run (demo mode)...")
    trained_model = run_training(
        epochs=demo_epochs,
        batch_size=demo_batch_size,
        debug=True,
        debug_sample_size=demo_debug_size,
        save_path=Config.MODEL_SAVE_PATH,
        patience=1,  # minimal patience
    )

    # Verify model file was created
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model file was not saved at {Config.MODEL_SAVE_PATH}")
    print(f"Model successfully saved to {Config.MODEL_SAVE_PATH}")

    # 2. Run Inference / Submission Generation
    print("Starting submission generation (demo mode)...")
    generate_submission(
        batch_size=demo_batch_size,
        debug=True,
        debug_sample_size=demo_debug_size,
        model_path=Config.MODEL_SAVE_PATH,
        output_path=Config.SUBMISSION_SAVE_PATH,
    )

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_SAVE_PATH):
        raise FileNotFoundError(
            f"Submission file was not saved at {Config.SUBMISSION_SAVE_PATH}"
        )

    # Check submission content
    df_sub = pd.read_csv(Config.SUBMISSION_SAVE_PATH)

    # Check columns
    expected_cols = ["id", "target"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Check length (should match debug_sample_size)
    assert (
        len(df_sub) == demo_debug_size
    ), f"Submission length mismatch. Expected {demo_debug_size}, got {len(df_sub)}"

    # Check values are probabilities (between 0 and 1)
    assert (
        df_sub["target"].min() >= 0.0 and df_sub["target"].max() <= 1.0
    ), "Submission contains values outside probability range [0, 1]."

    print(f"Submission successfully generated at {Config.SUBMISSION_SAVE_PATH}")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(Config.SEED)

    # Execute verification steps
    try:
        test_dataset_logic()
        test_model_logic()
        run_pipeline_demo()
        print("\nAll demonstrations completed successfully.")
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        raise e
