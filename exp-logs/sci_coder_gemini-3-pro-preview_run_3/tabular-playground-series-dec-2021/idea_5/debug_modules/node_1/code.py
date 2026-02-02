import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch

# Import provided library modules
from library.config import Config
from library.model import build_model
from library.data_utils import get_dataloaders
from library.train_eval import train_model


def setup_demo_environment():
    """
    Configures the environment for a fast demonstration run.
    Modifies the Config class in-place to use a small subset of data
    and separate working directories.
    """
    print("Setting up demo configuration...")

    # 1. Define a separate project name for the demo
    Config.PROJECT_NAME = "demo_task"

    # 2. Update working directories
    Config.WORKING_DIR = os.path.join("./working", Config.PROJECT_NAME)
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # 3. Update file paths to point to the new working directory
    Config.TRAIN_X_PATH = os.path.join(Config.WORKING_DIR, "train_X.npy")
    Config.TRAIN_Y_PATH = os.path.join(Config.WORKING_DIR, "train_y.npy")
    Config.VAL_X_PATH = os.path.join(Config.WORKING_DIR, "val_X.npy")
    Config.VAL_Y_PATH = os.path.join(Config.WORKING_DIR, "val_y.npy")
    Config.TEST_X_PATH = os.path.join(Config.WORKING_DIR, "test_X.npy")
    Config.TEST_IDS_PATH = os.path.join(Config.WORKING_DIR, "test_ids.npy")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # 4. Set Hyperparameters for Speed
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 2000  # Small subset for quick execution
    Config.EPOCHS = 2  # Minimal epochs to prove training loop works
    Config.BATCH_SIZE = 128
    Config.HIDDEN_DIM = 64  # Smaller model for speed
    Config.NUM_LAYERS = 2

    # 5. Ensure directories exist
    Config.setup()

    # Clean up previous demo runs if they exist to ensure fresh execution
    if os.path.exists(Config.TRAIN_X_PATH):
        try:
            # We remove one key file to force data reloading/processing
            os.remove(Config.TRAIN_X_PATH)
        except OSError:
            pass

    print(f"Config configured. Working dir: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG} (Subset size: {Config.DEBUG_SUBSET_SIZE})")


def verify_model_architecture():
    """
    Verifies that the model can be built and accepts input tensors correctly.
    """
    print("\n=== Verifying Model Architecture ===")

    # Simulate input dimensions (e.g., 54 original + 5 engineered = 59 features)
    dummy_input_dim = 59
    batch_size = 32

    # Build model
    model = build_model(input_dim=dummy_input_dim, config=Config)

    # Check if model is instance of torch.nn.Module
    assert isinstance(model, torch.nn.Module), "Model is not a torch.nn.Module"

    # Create dummy input
    dummy_input = torch.randn(batch_size, dummy_input_dim)

    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    # Check output shape: (Batch_Size, Num_Classes)
    expected_shape = (batch_size, Config.NUM_CLASSES)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, got {output.shape}"

    print("Model architecture verification passed.")


def verify_data_pipeline():
    """
    Verifies that DataLoaders are generated correctly and yield valid batches.
    """
    print("\n=== Verifying Data Pipeline ===")

    # Load data loaders (this triggers feature engineering and preprocessing)
    # We set load_cached_data=False to force processing logic execution
    train_loader, val_loader, test_loader = get_dataloaders(
        config=Config, load_cached_data=False
    )

    # Verify Train Loader
    X_batch, y_batch = next(iter(train_loader))
    print(f"Train Batch Shape - X: {X_batch.shape}, y: {y_batch.shape}")

    assert X_batch.dim() == 2, "Train input should be 2D"
    assert y_batch.dim() == 1, "Train target should be 1D"
    assert (
        X_batch.shape[0] == Config.BATCH_SIZE
        or X_batch.shape[0] <= Config.DEBUG_SUBSET_SIZE
    )

    # Verify Test Loader (should only have features)
    test_batch = next(iter(test_loader))
    # TensorDataset with one tensor returns a list/tuple of [tensor]
    if isinstance(test_batch, (list, tuple)):
        test_batch = test_batch[0]

    print(f"Test Batch Shape - X: {test_batch.shape}")
    assert test_batch.dim() == 2, "Test input should be 2D"

    # Verify Feature Engineering added columns
    # Original continuous cols (10) + Binary (44) = 54.
    # Engineered adds 5. Total should be 59.
    input_dim = X_batch.shape[1]
    print(f"Processed Input Feature Dimension: {input_dim}")
    assert input_dim > 54, "Feature engineering did not add columns as expected."

    print("Data pipeline verification passed.")


def run_full_training_process():
    """
    Runs the complete training loop using the provided train_model function.
    """
    print("\n=== Running Full Training Process ===")

    # Run training (this handles training, validation, saving model, and predicting)
    # We use load_cached_data=True because verify_data_pipeline already generated the cache
    model = train_model(config=Config, load_cached_data=True)

    # 1. Verify Model File Exists
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), f"Model file not found at {Config.MODEL_SAVE_PATH}"
    print(f"Verified model saved at: {Config.MODEL_SAVE_PATH}")

    # 2. Verify Submission File Exists
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"
    print(f"Verified submission saved at: {Config.SUBMISSION_PATH}")

    # 3. Verify Submission Content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission Head:")
    print(df_sub.head())

    # Check columns
    assert Config.ID_COL in df_sub.columns, f"Missing ID column {Config.ID_COL}"
    assert (
        Config.TARGET_COL in df_sub.columns
    ), f"Missing Target column {Config.TARGET_COL}"

    # Check row count (should match debug subset size)
    assert (
        len(df_sub) == Config.DEBUG_SUBSET_SIZE
    ), f"Expected {Config.DEBUG_SUBSET_SIZE} predictions, found {len(df_sub)}"

    # Check value range (Classes are 1-7)
    valid_classes = set(range(1, 8))  # 1 to 7
    preds = df_sub[Config.TARGET_COL].unique()
    assert all(
        p in valid_classes for p in preds
    ), f"Found invalid class labels: {preds}"

    print("Training process verification passed.")


if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    try:
        # 1. Setup
        setup_demo_environment()

        # 2. Unit Tests
        verify_model_architecture()
        verify_data_pipeline()

        # 3. Integration Test
        run_full_training_process()

        print("\nSUCCESS: All demonstrations and verifications completed successfully.")

    except AssertionError as e:
        print(f"\nFAILURE: Assertion failed - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nFAILURE: An unexpected error occurred - {e}")
        # Print traceback for debugging
        import traceback

        traceback.print_exc()
        sys.exit(1)
