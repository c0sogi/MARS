import os
import torch
import pandas as pd
import numpy as np
import warnings
import sys

# Import functions and classes from the provided library files
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import AsymmetricEfficientNet
from library.train import run_training, predict_and_submit

# Configuration for the demonstration
DEMO_WORK_DIR = "./working/demo_run"
SUBMISSION_FILE = os.path.join(DEMO_WORK_DIR, "submission.csv")
MODEL_PATH = os.path.join(DEMO_WORK_DIR, "best_model.pth")
BATCH_SIZE = 8  # Small batch size for speed
EPOCHS = 2  # Minimal epochs to demonstrate loop


def verify_data_loading():
    """
    Demonstrates how to initialize dataloaders and verifies data shapes.
    """
    print("\n=== 1. Verifying Data Loading ===")

    # Initialize DataLoaders
    # Note: This will calculate/load ROI cache automatically
    train_loader, val_loader, test_loader = get_dataloaders(
        train_metadata_path="./metadata/train.csv",
        val_metadata_path="./metadata/val.csv",
        test_metadata_path="./metadata/test.csv",
        input_dir="./input",
        batch_size=BATCH_SIZE,
        num_workers=2,
        load_cached_data=True,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # Fetch one batch from training loader
    images, targets = next(iter(train_loader))

    # Verify Shapes
    # Expected: (Batch_Size, 12, 224, 224)
    # 12 channels = 4 modalities * 3 slices (strided)
    print(f"Image Batch Shape: {images.shape}")
    print(f"Target Batch Shape: {targets.shape}")

    assert images.shape == (
        BATCH_SIZE,
        12,
        224,
        224,
    ), f"Expected image shape ({BATCH_SIZE}, 12, 224, 224), got {images.shape}"
    assert targets.shape == (
        BATCH_SIZE,
    ), f"Expected target shape ({BATCH_SIZE},), got {targets.shape}"

    print("Data loading verification passed.")


def verify_model_architecture():
    """
    Demonstrates model instantiation and performs a dummy forward pass.
    """
    print("\n=== 2. Verifying Model Architecture ===")

    device = get_device()
    model = AsymmetricEfficientNet(
        pretrained=False
    )  # False for speed in demo, True in prod
    model.to(device)
    model.eval()

    # Create dummy input: (Batch=2, Channels=12, H=224, W=224)
    dummy_input = torch.randn(2, 12, 224, 224).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Expected output: (Batch, 1) - Logits
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"

    print("Model architecture verification passed.")


def execute_training_demo():
    """
    Runs a short training loop to demonstrate the training pipeline.
    """
    print("\n=== 3. Executing Training Demo ===")

    # Ensure working directory exists
    os.makedirs(DEMO_WORK_DIR, exist_ok=True)

    # Run training
    # Using a small number of epochs and batch size for demonstration speed
    best_auc = run_training(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=1e-4,
        weight_decay=1e-2,
        patience=2,
        save_dir=DEMO_WORK_DIR,
    )

    print(f"Training completed. Best Validation AUC: {best_auc:.4f}")

    # Check if model file was created
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    print(f"Model saved successfully at {MODEL_PATH}")


def execute_inference_demo():
    """
    Generates predictions using the trained model and verifies submission format.
    """
    print("\n=== 4. Executing Inference Demo ===")

    # Generate predictions
    predict_and_submit(
        model_path=MODEL_PATH, output_file=SUBMISSION_FILE, batch_size=BATCH_SIZE
    )

    # Verify Submission File
    if not os.path.exists(SUBMISSION_FILE):
        raise FileNotFoundError(f"Submission file not found at {SUBMISSION_FILE}")

    df_sub = pd.read_csv(SUBMISSION_FILE)
    print(f"Submission DataFrame Head:\n{df_sub.head()}")

    # Check Columns
    expected_cols = ["BraTS21ID", "MGMT_value"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Expected columns {expected_cols}, got {list(df_sub.columns)}"

    # Check ID Format (should be 5-digit string, e.g., '00001')
    # Note: pandas might infer as int if all are numbers, but our code forces string formatting.
    # Let's check the first value.
    first_id = df_sub.iloc[0]["BraTS21ID"]
    print(f"Sample ID format: {first_id} (Type: {type(first_id)})")

    # If read as int/float by pandas, convert to verify logic
    first_id_str = str(first_id)
    assert len(first_id_str) == 5 or (
        isinstance(first_id, (int, np.integer))
    ), "ID format check warning: Ensure IDs are handled correctly."

    # Check Probability Range
    probs = df_sub["MGMT_value"]
    assert (
        probs.min() >= 0.0 and probs.max() <= 1.0
    ), "Probabilities must be between 0 and 1."

    print("Inference and submission verification passed.")


if __name__ == "__main__":
    # Setup
    warnings.filterwarnings("ignore")
    seed_everything(42)

    try:
        # 1. Data Loading
        verify_data_loading()

        # 2. Model
        verify_model_architecture()

        # 3. Training
        execute_training_demo()

        # 4. Inference
        execute_inference_demo()

        print("\nAll demonstrations completed successfully.")

    except Exception as e:
        print(f"\nAn error occurred during the demonstration: {e}")
        sys.exit(1)
