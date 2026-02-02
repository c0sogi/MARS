import os
import pandas as pd
import numpy as np
import torch
import shutil

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, log_message
from library.model import MGSHDNetwork
from library.train import run_training
from library.predict import run_inference


def main():
    # ==========================================
    # 1. Configuration Setup for Demo
    # ==========================================
    log_message("Setting up Demo Configuration...")

    # Define paths for the demo execution to avoid overwriting real work
    demo_dir = "./working/demo_execution"
    demo_meta_dir = os.path.join(demo_dir, "metadata")
    os.makedirs(demo_meta_dir, exist_ok=True)

    # Override Config class attributes for a lightweight run
    Config.WORKING_DIR = demo_dir
    Config.CACHE_DIR = demo_dir
    Config.SUBMISSION_DIR = demo_dir
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission.csv")
    Config.MODEL_PATH = os.path.join(demo_dir, "best_model.pth")

    # Reduce computational load
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4

    # Reduce input dimensionality for speed (8 slices instead of 32)
    Config.NUM_SLICES_PER_MODALITY = 8
    # Important: Update IN_CHANS as it depends on NUM_SLICES
    Config.IN_CHANS = Config.NUM_SLICES_PER_MODALITY * Config.NUM_MODALITIES

    # ==========================================
    # 2. Create Subset Metadata
    # ==========================================
    log_message("Creating subset metadata for rapid demonstration...")

    # Load original metadata files
    # Note: These files are guaranteed to exist per the prompt
    orig_train = pd.read_parquet("./metadata/train.parquet")
    orig_val = pd.read_parquet("./metadata/val.parquet")
    orig_test = pd.read_parquet("./metadata/test.parquet")

    # Sample a small number of subjects (e.g., 12 train, 4 val, 4 test)
    # This ensures data loading and preprocessing takes seconds, not hours
    n_train = min(12, len(orig_train))
    n_val = min(4, len(orig_val))
    n_test = min(4, len(orig_test))

    demo_train = orig_train.head(n_train)
    demo_val = orig_val.head(n_val)
    demo_test = orig_test.head(n_test)

    # Save these subsets to the demo metadata directory
    demo_train_path = os.path.join(demo_meta_dir, "train.parquet")
    demo_val_path = os.path.join(demo_meta_dir, "val.parquet")
    demo_test_path = os.path.join(demo_meta_dir, "test.parquet")

    demo_train.to_parquet(demo_train_path)
    demo_val.to_parquet(demo_val_path)
    demo_test.to_parquet(demo_test_path)

    # Point Config to these new subset files
    Config.TRAIN_META_PATH = demo_train_path
    Config.VAL_META_PATH = demo_val_path
    Config.TEST_META_PATH = demo_test_path

    log_message(f"Demo Metadata: Train={n_train}, Val={n_val}, Test={n_test}")

    # ==========================================
    # 3. Verify Model Architecture
    # ==========================================
    log_message("Verifying Model Architecture...")

    # Instantiate model with the updated Config
    model = MGSHDNetwork()

    # Create dummy input: (Batch=2, Channels=32, H=256, W=256)
    # Channels = 4 modalities * 8 slices = 32
    dummy_input = torch.randn(2, Config.IN_CHANS, Config.IMG_SIZE, Config.IMG_SIZE)

    try:
        output = model(dummy_input)
        # Expect output shape (Batch, 1)
        assert output.shape == (
            2,
            1,
        ), f"Expected output shape (2, 1), got {output.shape}"
        log_message("Model forward pass successful. Output shape verified.")
    except Exception as e:
        log_message(f"Model verification failed: {e}")
        raise e

    # ==========================================
    # 4. Run Training Pipeline
    # ==========================================
    log_message("\nStarting Training Demo...")

    # load_cached_data=False ensures we process our new subset metadata
    # instead of loading potentially existing full-dataset cache files.
    run_training(load_cached_data=False, patience=1)

    # Verify that the model file was created
    if not os.path.exists(Config.MODEL_PATH):
        raise FileNotFoundError(f"Model file was not created at {Config.MODEL_PATH}")

    log_message("Training demo completed successfully.")

    # ==========================================
    # 5. Run Inference Pipeline
    # ==========================================
    log_message("\nStarting Inference Demo...")

    run_inference(load_cached_data=False)

    # Verify submission file
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file was not created at {Config.SUBMISSION_PATH}"
        )

    # Check content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    log_message(f"Submission generated with {len(sub_df)} rows.")

    # Validate row count matches test subset
    assert (
        len(sub_df) == n_test
    ), f"Expected {n_test} rows in submission, got {len(sub_df)}"

    # Validate columns
    expected_cols = ["BraTS21ID", "MGMT_value"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Expected columns {expected_cols}, got {list(sub_df.columns)}"

    log_message("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
