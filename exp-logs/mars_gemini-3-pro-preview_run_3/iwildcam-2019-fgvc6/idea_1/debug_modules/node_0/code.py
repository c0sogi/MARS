import os
import sys
import torch
import pandas as pd
import numpy as np

# Import the library components
# Note: We must import Config first to modify it before other modules use it.
from library.config import Config

# ==========================================
# 1. Configuration Override for Speed/Demo
# ==========================================
print("[Demo] Configuring environment for rapid demonstration...")

# Enable Debug mode to use a tiny subset of data (50 samples)
Config.DEBUG = True
Config.DEBUG_SUBSET_SIZE = 50

# Reduce training parameters for speed
Config.EPOCHS = 1
Config.BATCH_SIZE = 8
Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny datasets

# Ensure reproducible seeds
Config.SEED = 999

# Import remaining components after config update
from library.dataset import get_dataloaders
from library.model import get_model
from library.trainer import Trainer
from library.inference import predict_and_submit


def verify_data_loading():
    print("\n[Demo] Verifying Data Loading...")

    # Get dataloaders (this will trigger metadata processing for the debug subset)
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Check 1: Verify DataLoader lengths
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    assert len(train_loader) > 0, "Train loader is empty"

    # Check 2: Fetch a batch and verify shapes
    images, labels, ids = next(iter(train_loader))

    print(f"  Image Batch Shape: {images.shape}")
    print(f"  Label Batch Shape: {labels.shape}")

    # Expected: (Batch_Size, 3, 224, 224)
    expected_shape = (Config.BATCH_SIZE, 3, 224, 224)
    # Note: The last batch might be smaller, but since we set subset=50 and batch=8,
    # the first batch should be full size 8.

    assert (
        images.shape == expected_shape
    ), f"Expected images shape {expected_shape}, got {images.shape}"
    assert labels.shape[0] == Config.BATCH_SIZE, "Label batch size mismatch"

    # Check 3: Verify label range
    max_label = labels.max().item()
    min_label = labels.min().item()
    print(f"  Labels in batch: min={min_label}, max={max_label}")

    assert min_label >= 0, "Found negative label"
    assert (
        max_label < Config.NUM_CLASSES
    ), f"Found label {max_label} exceeding num_classes {Config.NUM_CLASSES}"

    print("  Data Loading verification successful.")
    return images  # Return images for model verification


def verify_model(sample_images):
    print("\n[Demo] Verifying Model Architecture...")

    # Instantiate model
    model = get_model(pretrained=False, num_classes=Config.NUM_CLASSES)
    model.eval()

    # Forward pass
    with torch.no_grad():
        outputs = model(sample_images)

    print(f"  Output Shape: {outputs.shape}")

    # Expected: (Batch_Size, Num_Classes)
    expected_output_shape = (Config.BATCH_SIZE, Config.NUM_CLASSES)

    assert (
        outputs.shape == expected_output_shape
    ), f"Expected output shape {expected_output_shape}, got {outputs.shape}"

    print("  Model verification successful.")


def run_training():
    print("\n[Demo] Running Training Loop...")

    # Initialize Trainer
    trainer = Trainer()

    # Run training (fit)
    # This uses the modified Config (1 Epoch, Debug subset)
    trainer.fit()

    # Verify checkpoint creation
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"  Checkpoint found at: {Config.MODEL_SAVE_PATH}")
    else:
        # If the model didn't improve (unlikely with 1 epoch and random init vs random init),
        # it might not save 'best'. However, for this demo, we want to ensure the code ran.
        # Let's force a save if it doesn't exist just for the inference step,
        # though the Trainer logic usually saves if validation runs.
        # In the provided Trainer code, it saves if val_f1 > -1.0 + delta.
        # Initial best_f1 is -1.0. So it should save on epoch 1.
        raise FileNotFoundError(
            f"Model checkpoint was not created at {Config.MODEL_SAVE_PATH}"
        )

    print("  Training loop execution successful.")


def run_inference():
    print("\n[Demo] Running Inference and Submission Generation...")

    # Run prediction pipeline
    predict_and_submit()

    # Verify submission file
    sub_path = Config.SUBMISSION_PATH
    if not os.path.exists(sub_path):
        raise FileNotFoundError(f"Submission file not found at {sub_path}")

    print(f"  Submission file generated at: {sub_path}")

    # Validate content
    df = pd.read_csv(sub_path)
    print(f"  Submission Head:\n{df.head()}")

    # Check columns
    expected_cols = ["Id", "Predicted"]
    assert (
        list(df.columns) == expected_cols
    ), f"Expected columns {expected_cols}, got {list(df.columns)}"

    # Check length (should match debug subset size)
    # Note: Test set is also subsetted in DEBUG mode
    print(f"  Submission Rows: {len(df)}")
    assert (
        len(df) == Config.DEBUG_SUBSET_SIZE
    ), f"Expected {Config.DEBUG_SUBSET_SIZE} rows, got {len(df)}"

    # Check values
    assert (
        df["Predicted"].dtype == np.int64 or df["Predicted"].dtype == np.int32
    ), "Predicted column should be integers"

    print("  Inference verification successful.")


if __name__ == "__main__":
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    try:
        # 1. Verify Data Loading
        sample_images = verify_data_loading()

        # 2. Verify Model
        verify_model(sample_images)

        # 3. Run Training
        run_training()

        # 4. Run Inference
        run_inference()

        print("\n[Demo] All steps completed successfully.")

    except AssertionError as e:
        print(f"\n[Error] Assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[Error] An unexpected error occurred: {e}")
        # Print traceback for debugging
        import traceback

        traceback.print_exc()
        sys.exit(1)
