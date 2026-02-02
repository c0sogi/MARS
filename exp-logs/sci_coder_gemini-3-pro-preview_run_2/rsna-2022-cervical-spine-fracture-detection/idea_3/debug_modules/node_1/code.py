import os
import torch
import pandas as pd
import numpy as np
import warnings
import shutil
from library.config import Config
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders, CervicalSpineDataset, get_transforms
from library.model import CervicalFractureNet
from library.losses import WeightedMultiLabelLoss
from library.trainer import Trainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Cervical Spine Fracture Detection Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed and Isolation
    # -------------------------------------------------------------------------
    print("1. Configuring environment for rapid demonstration...")

    # Create a unique working directory for this demo
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR)

    # Override Config parameters
    Config.WORKING_DIR = DEMO_DIR
    Config.OUTPUT_DIR = os.path.join(DEMO_DIR, "output")
    Config.MODEL_CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.OUTPUT_DIR, "submission.csv")

    # Cache files specific to demo
    Config.TRAIN_CACHE_PATH = os.path.join(Config.WORKING_DIR, "train_cache.parquet")
    Config.VAL_CACHE_PATH = os.path.join(Config.WORKING_DIR, "val_cache.parquet")
    Config.TEST_CACHE_PATH = os.path.join(Config.WORKING_DIR, "test_cache.parquet")

    # Hyperparameters for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 6  # Very small subset
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.GRAD_ACCUM_STEPS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data
    Config.SEQ_LEN = 16  # Reduced sequence length for speed (default is 48)
    Config.IMAGE_SIZE = 256  # Reduced image size for speed (default is 384)

    # Initialize directories
    Config.setup()

    # Set seeds
    seed_everything(Config.SEED)
    print("   Configuration updated successfully.")

    # -------------------------------------------------------------------------
    # 2. Data Loader Verification
    # -------------------------------------------------------------------------
    print("\n2. Verifying Data Loaders...")

    # Get loaders
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Fetch one batch
    batch = next(iter(train_loader))
    images = batch["image"]
    targets = batch["target"]
    study_ids = batch["study_id"]

    print(f"   Batch fetched. Study IDs: {study_ids}")
    print(f"   Image Tensor Shape: {images.shape}")
    print(f"   Target Tensor Shape: {targets.shape}")

    # Assertions
    # Shape: (Batch, Seq_Len, Channels, H, W) -> Permuted in Dataset to (Batch, Seq_Len, 3, H, W)
    # Wait, the Dataset class permutes it to (SEQ_LEN, 3, H, W).
    # DataLoader stacks them -> (Batch, SEQ_LEN, 3, H, W).
    expected_image_shape = (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    )
    expected_target_shape = (Config.BATCH_SIZE, Config.NUM_CLASSES)

    assert (
        images.shape == expected_image_shape
    ), f"Image shape mismatch. Expected {expected_image_shape}, got {images.shape}"
    assert (
        targets.shape == expected_target_shape
    ), f"Target shape mismatch. Expected {expected_target_shape}, got {targets.shape}"

    print("   Data Loader shapes verified.")

    # -------------------------------------------------------------------------
    # 3. Model Verification
    # -------------------------------------------------------------------------
    print("\n3. Verifying Model Architecture...")

    device = get_device()
    model = CervicalFractureNet().to(device)

    # Move batch to device
    images = images.to(device)

    # Forward pass
    logits = model(images)

    print(f"   Logits Shape: {logits.shape}")

    # Assertions
    assert (
        logits.shape == expected_target_shape
    ), f"Model output shape mismatch. Expected {expected_target_shape}, got {logits.shape}"

    print("   Model forward pass successful.")

    # -------------------------------------------------------------------------
    # 4. Loss Function Verification
    # -------------------------------------------------------------------------
    print("\n4. Verifying Loss Function...")

    criterion = WeightedMultiLabelLoss().to(device)
    targets = targets.to(device)

    loss = criterion(logits, targets)

    print(f"   Loss Value: {loss.item():.4f}")

    assert torch.is_tensor(loss) and loss.dim() == 0, "Loss should be a scalar tensor."
    assert not torch.isnan(loss), "Loss is NaN."

    print("   Loss calculation successful.")

    # -------------------------------------------------------------------------
    # 5. Trainer Execution (Fit)
    # -------------------------------------------------------------------------
    print("\n5. Executing Training Loop (Trainer.fit)...")

    # Re-initialize trainer to ensure clean state
    trainer = Trainer(load_cached_data=True)  # Use cache generated in step 2

    # Run training
    # This will run for 1 epoch on the tiny debug dataset
    trainer.fit()

    # Check if checkpoint exists
    assert os.path.exists(
        Config.MODEL_CHECKPOINT_PATH
    ), f"Checkpoint not found at {Config.MODEL_CHECKPOINT_PATH}"

    print("   Training loop completed and checkpoint saved.")

    # -------------------------------------------------------------------------
    # 6. Inference and Submission
    # -------------------------------------------------------------------------
    print("\n6. Generating Submission...")

    trainer.generate_submission()

    # Verify submission file
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"   Submission file loaded. Rows: {len(df_sub)}")
    print(f"   First 5 rows:\n{df_sub.head()}")

    # Validate submission format
    assert (
        "row_id" in df_sub.columns and "fractured" in df_sub.columns
    ), "Submission missing required columns."

    # Check row count
    # Test set debug size is 6. Each study has 8 predictions. Total rows = 6 * 8 = 48.
    expected_rows = Config.DEBUG_SAMPLE_SIZE * 8
    assert (
        len(df_sub) == expected_rows
    ), f"Submission row count mismatch. Expected {expected_rows}, got {len(df_sub)}"

    print("   Submission generated and verified successfully.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
