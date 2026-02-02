import os
import torch
import pandas as pd
import numpy as np
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_dataloaders
from library.model import MGMTClassifier
from library.train import run_training


def main():
    print("Initializing Demonstration Script...")

    # ==========================================
    # 1. Configuration Overrides for Speed/Demo
    # ==========================================
    print("Configuring parameters for rapid execution...")

    # Use a separate working directory for the demo to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config constants
    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")

    # Cache paths specific to demo
    Config.CACHE_TRAIN_PATH = os.path.join(DEMO_DIR, "cache_train.parquet")
    Config.CACHE_VAL_PATH = os.path.join(DEMO_DIR, "cache_val.parquet")
    Config.CACHE_TEST_PATH = os.path.join(DEMO_DIR, "cache_test.parquet")

    # Training hyperparameters for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 2
    Config.PRETRAINED = False  # Disable download for speed/offline safety

    # Use a very small subset of data
    # We use enough subjects to ensure train/val split works and we have > 1 batch
    DEBUG_SIZE = 10

    # Set seeds
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Running on device: {device}")

    # ==========================================
    # 2. Verify Data Loading Pipeline
    # ==========================================
    print("\n--- Verifying Data Loading ---")

    # Generate dataloaders with debug size and force cache regeneration (load_cached_data=False)
    train_loader, val_loader, test_loader, test_df = get_dataloaders(
        debug_sample_size=DEBUG_SIZE, load_cached_data=False
    )

    # Assertions for Train Loader
    assert len(train_loader) > 0, "Train loader should not be empty."

    # Fetch one batch
    images, labels = next(iter(train_loader))

    print(f"Batch Image Shape: {images.shape}")
    print(f"Batch Label Shape: {labels.shape}")

    # Verify Shapes
    # Expected: (Batch_Size, 3, 224, 224)
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), f"Incorrect image shape. Expected {(Config.BATCH_SIZE, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)}, got {images.shape}"

    # Expected: (Batch_Size,)
    assert labels.shape == (
        Config.BATCH_SIZE,
    ), f"Incorrect label shape. Expected {(Config.BATCH_SIZE,)}, got {labels.shape}"

    # Verify Data Expansion (Idea 9)
    # The dataset should have 3x samples as the input dataframe due to 3 slices per subject
    # Note: debug_sample_size in prepare_data slices the metadata DF.
    # If we requested 10 subjects, we expect roughly 30 samples (minus any missing files).
    dataset_len = len(train_loader.dataset)
    print(f"Train Dataset Length (Expanded): {dataset_len}")
    assert (
        dataset_len > DEBUG_SIZE
    ), "Dataset expansion (3 slices per subject) does not seem to be working."

    # ==========================================
    # 3. Verify Model Architecture
    # ==========================================
    print("\n--- Verifying Model Architecture ---")

    model = MGMTClassifier().to(device)

    # Forward pass with the batch fetched earlier
    images = images.to(device)
    output = model(images)

    print(f"Model Output Shape: {output.shape}")

    # Expected: (Batch_Size, 1) -> Binary Logits
    assert output.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Incorrect model output shape. Expected {(Config.BATCH_SIZE, 1)}, got {output.shape}"

    # Check if gradients are available (model is trainable)
    assert (
        output.requires_grad
    ), "Model output does not require gradients. Check model definition."

    # ==========================================
    # 4. Verify Full Training Loop
    # ==========================================
    print("\n--- Verifying Training Loop Execution ---")

    # Run the training function
    # This handles training, validation, checkpointing, and submission generation
    best_model_path = run_training(
        debug_sample_size=DEBUG_SIZE,
        epochs=Config.EPOCHS,
        load_cached_data=True,  # Use the cache we just generated in step 2
    )

    # ==========================================
    # 5. Verify Outputs
    # ==========================================
    print("\n--- Verifying Outputs ---")

    # 1. Check Model Checkpoint
    assert os.path.exists(
        best_model_path
    ), f"Best model file not found at {best_model_path}"
    print(f"Confirmed: Best model saved at {best_model_path}")

    # 2. Check Submission File
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"
    print(f"Confirmed: Submission file saved at {Config.SUBMISSION_PATH}")

    # 3. Validate Submission Content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission File Head:")
    print(df_sub.head())

    # Check columns
    expected_cols = ["BraTS21ID", "MGMT_value"]
    assert (
        list(df_sub.columns) == expected_cols
    ), f"Submission columns mismatch. Expected {expected_cols}, got {list(df_sub.columns)}"

    # Check values (probabilities should be between 0 and 1)
    # Note: run_training applies sigmoid before saving
    preds = df_sub["MGMT_value"].values
    assert np.all(preds >= 0.0) and np.all(
        preds <= 1.0
    ), "Predictions are not valid probabilities (0-1)."

    # Check that we have predictions for the test subjects
    # Since we used DEBUG_SIZE on test data too, we expect min(DEBUG_SIZE, total_test_subjects) rows
    assert len(df_sub) > 0, "Submission file is empty."

    print("\nAll verification steps passed successfully!")


if __name__ == "__main__":
    main()
