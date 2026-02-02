import os
import sys
import torch
import pandas as pd
import numpy as np

# =============================================================================
# 1. Configuration Override for Fast Demonstration
# =============================================================================
# We import the config module first and modify its attributes.
# Subsequent imports of 'library.trainer' etc. will pick up these modified values.
import library.config as config

print("[Demo] Configuring parameters for rapid execution...")
config.DEBUG_DATA_LIMIT = 12  # Use a tiny subset of data (divisible by batch size)
config.NUM_EPOCHS = 2  # Run only 2 epochs to verify the loop
config.BATCH_SIZE = 4  # Small batch size
config.WORKING_DIR = "./working/demo_execution"  # Isolated working directory
config.SUBMISSION_PATH = os.path.join(config.WORKING_DIR, "demo_submission.csv")

# Ensure working directory exists
os.makedirs(config.WORKING_DIR, exist_ok=True)

# =============================================================================
# 2. Import Library Modules
# =============================================================================
from library.utils import seed_everything, get_device
from library.dataset import BraTSDataset, get_dataloader
from library.model import SIRVEfficientNet
from library.trainer import Trainer


def verify_dataset_logic():
    print("\n[Demo] Verifying Dataset Logic...")

    # Load metadata
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    if config.DEBUG_DATA_LIMIT:
        df_train = df_train.head(config.DEBUG_DATA_LIMIT)

    # Instantiate Dataset
    dataset = BraTSDataset(df_train, phase="train")

    # Fetch one sample
    image, label = dataset[0]

    # Assertions
    # Expected shape: (Channels, Height, Width) -> (9, 224, 224)
    expected_shape = (9, config.IMG_SIZE, config.IMG_SIZE)
    if image.shape != expected_shape:
        raise AssertionError(
            f"Dataset image shape mismatch. Expected {expected_shape}, got {image.shape}"
        )

    if not isinstance(label, torch.Tensor):
        raise AssertionError("Dataset label should be a torch.Tensor")

    print(f"  - Sample shape verified: {image.shape}")
    print(f"  - Label verified: {label.item()}")
    print("  - Dataset logic passed.")


def verify_model_logic():
    print("\n[Demo] Verifying Model Logic...")

    device = get_device()
    model = SIRVEfficientNet(
        pretrained=False
    )  # No need to download weights for shape check
    model.to(device)
    model.eval()

    # Create dummy input batch: (Batch, Channels, Height, Width)
    dummy_input = torch.randn(2, 9, config.IMG_SIZE, config.IMG_SIZE).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    # Assertions
    if output.shape != (2, 1):
        raise AssertionError(
            f"Model output shape mismatch. Expected (2, 1), got {output.shape}"
        )

    print(f"  - Input shape: {dummy_input.shape}")
    print(f"  - Output shape: {output.shape}")
    print("  - Model logic passed.")


def execute_training_pipeline():
    print("\n[Demo] Executing Training Pipeline...")

    # Initialize Trainer
    # This will load the model and prepare optimizers
    trainer = Trainer()

    # Override model save path to our demo directory
    trainer.model_save_path = os.path.join(config.WORKING_DIR, "demo_model.pth")

    # Run Training
    # This uses the modified config.DEBUG_DATA_LIMIT and config.NUM_EPOCHS
    trainer.fit()

    # Verify model checkpoint creation
    if not os.path.exists(trainer.model_save_path):
        raise AssertionError("Training failed to save the best model checkpoint.")

    print("  - Training loop completed.")
    print(f"  - Model saved to {trainer.model_save_path}")


def execute_inference_pipeline():
    print("\n[Demo] Executing Inference Pipeline...")

    trainer = Trainer()
    trainer.model_save_path = os.path.join(config.WORKING_DIR, "demo_model.pth")

    # Run Prediction
    # This generates the submission file at config.SUBMISSION_PATH
    trainer.predict()

    # Verify Submission
    if not os.path.exists(config.SUBMISSION_PATH):
        raise AssertionError("Inference failed to generate submission file.")

    df_sub = pd.read_csv(config.SUBMISSION_PATH)

    # Check columns
    required_cols = ["BraTS21ID", "MGMT_value"]
    if not all(col in df_sub.columns for col in required_cols):
        raise AssertionError(f"Submission file missing columns. Found {df_sub.columns}")

    # Check row count (should match DEBUG_DATA_LIMIT if applied to test,
    # but Trainer applies it to test as well in this setup)
    print(f"  - Submission generated with {len(df_sub)} rows.")
    print(f"  - File location: {config.SUBMISSION_PATH}")
    print("  - Inference logic passed.")


if __name__ == "__main__":
    # Set seeds for reproducibility
    seed_everything(config.SEED)

    try:
        # 1. Verify Data Loading
        verify_dataset_logic()

        # 2. Verify Model Architecture
        verify_model_logic()

        # 3. Run Training Loop
        execute_training_pipeline()

        # 4. Run Inference Loop
        execute_inference_pipeline()

        print("\n" + "=" * 40)
        print(" DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("=" * 40)

    except Exception as e:
        print(f"\n[ERROR] Demonstration failed: {e}")
        # Re-raise to ensure the task is marked as failed if something goes wrong
        raise e
