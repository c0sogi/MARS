import os
import sys
import pandas as pd
import torch
import numpy as np

# Import library modules
from library import config
from library import utils
from library import model
from library import data_loader
from library import train
from library import predict


def run_demo():
    print("Initializing Demo...")

    # =========================================================================
    # 1. SETUP & CONFIGURATION OVERRIDES
    # =========================================================================
    # We modify the config parameters at runtime to speed up the demo.
    # Original: 256x256, 32 slices. Demo: 64x64, 4 slices.

    DEMO_WORKING_DIR = "./working/demo_execution"
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    # Update Config Paths
    config.WORKING_DIR = DEMO_WORKING_DIR
    config.MODEL_PATH = os.path.join(DEMO_WORKING_DIR, "best_model.pth")
    config.SUBMISSION_PATH = os.path.join(DEMO_WORKING_DIR, "submission.csv")

    # Update Data Hyperparameters for Speed
    config.IMG_SIZE = 64
    config.NUM_SLICES = 4
    config.NUM_MODALITIES = 4
    # Important: Recalculate total input channels as it is a derived variable
    config.TOTAL_INPUT_CHANNELS = config.NUM_SLICES * config.NUM_MODALITIES

    # Update Training Hyperparameters
    config.BATCH_SIZE = 2
    config.EPOCHS = 1  # Run only 1 epoch
    config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny demo

    # Set seed for reproducibility
    utils.set_seed(config.SEED)

    print(
        f"Config updated: Image Size={config.IMG_SIZE}, Slices={config.NUM_SLICES}, Epochs={config.EPOCHS}"
    )

    # =========================================================================
    # 2. CREATE MINI DATASETS (Subsetting Metadata)
    # =========================================================================
    # Instead of processing the full dataset, we create mini parquet files
    # pointing to a few real samples.

    print("Creating mini metadata files...")

    # Load original metadata
    full_train_df = pd.read_parquet(config.TRAIN_META_PATH)
    full_val_df = pd.read_parquet(config.VAL_META_PATH)
    full_test_df = pd.read_parquet(config.TEST_META_PATH)

    # Subset (Take top 4 for train, 2 for val, 2 for test)
    mini_train_df = full_train_df.head(4).copy()
    mini_val_df = full_val_df.head(2).copy()
    mini_test_df = full_test_df.head(2).copy()

    # Save mini metadata to demo working directory
    mini_train_path = os.path.join(DEMO_WORKING_DIR, "train.parquet")
    mini_val_path = os.path.join(DEMO_WORKING_DIR, "val.parquet")
    mini_test_path = os.path.join(DEMO_WORKING_DIR, "test.parquet")

    mini_train_df.to_parquet(mini_train_path)
    mini_val_df.to_parquet(mini_val_path)
    mini_test_df.to_parquet(mini_test_path)

    # Point config to these new mini files
    config.TRAIN_META_PATH = mini_train_path
    config.VAL_META_PATH = mini_val_path
    config.TEST_META_PATH = mini_test_path

    print(f"Mini datasets created at {DEMO_WORKING_DIR}")

    # =========================================================================
    # 3. VERIFY MODEL ARCHITECTURE
    # =========================================================================
    print("Verifying model architecture...")
    net = model.SHDVNet()

    # Create dummy input: (Batch, Channels, H, W)
    # Channels = NUM_SLICES * NUM_MODALITIES = 4 * 4 = 16
    dummy_input = torch.randn(
        2, config.TOTAL_INPUT_CHANNELS, config.IMG_SIZE, config.IMG_SIZE
    )

    # Forward pass
    output = net(dummy_input)

    # Check output shape (Batch, Num_Classes) -> (2, 1)
    assert output.shape == (
        2,
        1,
    ), f"Model output shape mismatch. Expected (2, 1), got {output.shape}"
    print("Model architecture verified.")

    # =========================================================================
    # 4. EXECUTE TRAINING PIPELINE
    # =========================================================================
    print("Starting training pipeline...")

    # This function handles data loading (using our mini metadata),
    # model init, training loop, validation, and saving the best model.
    train.run_training(
        epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, lr=config.LEARNING_RATE
    )

    # Verify model file was created
    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Training failed to produce model file at {config.MODEL_PATH}"
        )

    print("Training pipeline completed successfully.")

    # =========================================================================
    # 5. EXECUTE INFERENCE PIPELINE
    # =========================================================================
    print("Starting inference pipeline...")

    # This function loads the model we just trained and predicts on the mini test set
    predict.generate_submission(
        model_path=config.MODEL_PATH,
        metadata_path=config.TEST_META_PATH,
        output_path=config.SUBMISSION_PATH,
        device=config.DEVICE,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    # Verify submission file
    if not os.path.exists(config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Inference failed to produce submission file at {config.SUBMISSION_PATH}"
        )

    # Check submission content
    submission_df = pd.read_csv(config.SUBMISSION_PATH)
    print("Submission Head:")
    print(submission_df.head())

    assert (
        len(submission_df) == 2
    ), f"Submission length mismatch. Expected 2, got {len(submission_df)}"
    assert "BraTS21ID" in submission_df.columns, "Submission missing BraTS21ID column"
    assert "MGMT_value" in submission_df.columns, "Submission missing MGMT_value column"

    print("Inference pipeline completed successfully.")
    print("Demo execution finished.")


if __name__ == "__main__":
    run_demo()
