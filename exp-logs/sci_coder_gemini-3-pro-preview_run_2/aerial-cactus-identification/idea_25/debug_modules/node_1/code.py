import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np

# Import provided library modules
import library.config
import library.dataset
import library.model
import library.train
import library.utils


def main():
    print("============================================")
    print("      Cactus Classification Demo Script     ")
    print("============================================")

    # ---------------------------------------------------------
    # 1. Configuration for Speed Optimization & Isolation
    # ---------------------------------------------------------
    # We define a separate working directory for this demo to avoid
    # interfering with any main training runs.
    DEMO_WORKING_DIR = "./working/demo_execution"
    DEMO_SUBMISSION_DIR = os.path.join(DEMO_WORKING_DIR, "submission")
    DEMO_SUBMISSION_PATH = os.path.join(DEMO_SUBMISSION_DIR, "submission_demo.csv")

    # Clean up demo dir if it exists to ensure a fresh run
    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)
    os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)

    print(f"Working Directory: {DEMO_WORKING_DIR}")

    # Monkey-patching configuration constants across all modules.
    # This is necessary because the modules import these constants into their global namespace.
    # We set DEBUG=True to use only 100 images, and EPOCHS=1 for a quick run.

    # Patch library.config (Source of truth, though others have already imported from it)
    library.config.WORKING_DIR = DEMO_WORKING_DIR
    library.config.SUBMISSION_PATH = DEMO_SUBMISSION_PATH
    library.config.DEBUG = True
    library.config.EPOCHS = 1
    library.config.SEEDS = [42]  # Single seed for speed

    # Patch library.dataset
    library.dataset.WORKING_DIR = DEMO_WORKING_DIR
    library.dataset.DEBUG = True

    # Patch library.model
    library.model.WORKING_DIR = DEMO_WORKING_DIR
    library.model.SUBMISSION_PATH = DEMO_SUBMISSION_PATH
    library.model.EPOCHS = 1
    library.model.SEEDS = [42]

    # Patch library.train
    library.train.WORKING_DIR = DEMO_WORKING_DIR
    library.train.SUBMISSION_PATH = DEMO_SUBMISSION_PATH
    library.train.EPOCHS = 1
    library.train.SEEDS = [42]

    # Set global seed for reproducibility
    library.utils.set_seed(42)

    print("Configuration patched: DEBUG=True, EPOCHS=1, SEEDS=[42]")

    # ---------------------------------------------------------
    # 2. Data Loading & Verification
    # ---------------------------------------------------------
    print("\n[Step 1] Verifying Data Loading...")

    # We force `load_cached_data=False` to trigger the processing logic
    # which will observe our patched DEBUG=True flag and create small cache files.
    train_loader, val_loader, test_loader = library.dataset.get_dataloaders(
        load_cached_data=False
    )

    # In DEBUG mode, the dataset is sliced to 100 images.
    # BATCH_SIZE is 128 (default in config), so we expect exactly 1 batch per loader.
    print(f"Train Loader Batches: {len(train_loader)}")
    print(f"Val Loader Batches:   {len(val_loader)}")
    print(f"Test Loader Batches:  {len(test_loader)}")

    assert len(train_loader) == 1, "Expected 1 batch for training in debug mode"

    # Check batch dimensions
    images, labels, ids = next(iter(train_loader))
    print(f"Sample Batch Shape - Images: {images.shape}, Labels: {labels.shape}")

    # Verify shape: (Batch_Size, Channels, Height, Width)
    # With 100 samples and batch_size 128, the batch size is 100.
    assert images.shape == (
        100,
        3,
        32,
        32,
    ), f"Expected (100, 3, 32, 32), got {images.shape}"
    assert labels.shape == (100,), f"Expected (100,), got {labels.shape}"

    print("Data loading verification passed.")

    # ---------------------------------------------------------
    # 3. Model Architecture Verification
    # ---------------------------------------------------------
    print("\n[Step 2] Verifying Model Architecture...")

    model = library.model.CactusNet(num_classes=1)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    # Expect (Batch_Size, Num_Classes)
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"

    print("Model architecture verification passed.")

    # ---------------------------------------------------------
    # 4. Training Pipeline Execution
    # ---------------------------------------------------------
    print("\n[Step 3] Executing Training Pipeline...")
    print("Running training for 1 epoch on 100 images...")

    # This function handles the loop, validation, early stopping, and submission generation.
    library.train.run_training()

    print("Training pipeline execution finished.")

    # ---------------------------------------------------------
    # 5. Output Verification
    # ---------------------------------------------------------
    print("\n[Step 4] Verifying Outputs...")

    # 5.1 Verify Model Checkpoint
    model_path = os.path.join(DEMO_WORKING_DIR, "model_seed_42.pth")
    if os.path.exists(model_path):
        print(f"SUCCESS: Model checkpoint found at {model_path}")
        file_size = os.path.getsize(model_path)
        print(f"Model size: {file_size / 1024 / 1024:.2f} MB")
    else:
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

    # 5.2 Verify Submission File
    if os.path.exists(DEMO_SUBMISSION_PATH):
        print(f"SUCCESS: Submission file found at {DEMO_SUBMISSION_PATH}")

        df_sub = pd.read_csv(DEMO_SUBMISSION_PATH)
        print("Submission Head:")
        print(df_sub.head())

        # Check Columns
        assert list(df_sub.columns) == [
            "id",
            "has_cactus",
        ], "Submission columns mismatch"

        # Check Length (Should be 100 in DEBUG mode)
        assert len(df_sub) == 100, f"Expected 100 predictions, got {len(df_sub)}"

        # Check Values (Probabilities between 0 and 1)
        assert df_sub["has_cactus"].min() >= 0.0, "Probabilities below 0 found"
        assert df_sub["has_cactus"].max() <= 1.0, "Probabilities above 1 found"

        print("Submission file format verification passed.")
    else:
        raise FileNotFoundError(f"Submission file not found at {DEMO_SUBMISSION_PATH}")

    print("\n============================================")
    print("      Demonstration Completed Successfully  ")
    print("============================================")


if __name__ == "__main__":
    main()
