import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

# Import from the provided library
from library.config import Config
from library.utils import seed_everything
from library.data_processing import prepare_data
from library.model import AVPFEModel
from library.training import train_one_epoch, validate, generate_submission


def run_demo():
    # 1. Configuration Setup for Demo
    print("--- 1. Configuring Environment ---")

    # Define a separate working directory for this demo to avoid conflicts
    demo_working_dir = "./working/demo_execution"
    if os.path.exists(demo_working_dir):
        shutil.rmtree(demo_working_dir)
    os.makedirs(demo_working_dir, exist_ok=True)

    # Override Config to use the demo directory and debug mode
    # Note: Modifying class attributes here affects the Config used by other modules
    Config.WORKING_DIR = demo_working_dir
    Config.SUBMISSION_DIR = demo_working_dir

    # Update paths that were initialized based on the original WORKING_DIR
    Config.TRAIN_PROCESSED_PATH = os.path.join(
        Config.WORKING_DIR, "train_processed.parquet"
    )
    Config.VAL_PROCESSED_PATH = os.path.join(
        Config.WORKING_DIR, "val_processed.parquet"
    )
    Config.TEST_PROCESSED_PATH = os.path.join(
        Config.WORKING_DIR, "test_processed.parquet"
    )
    Config.METADATA_CACHE_PATH = os.path.join(Config.WORKING_DIR, "metadata.npy")
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Set Debug parameters for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 2000  # Small subset for quick execution
    Config.BATCH_SIZE = 128
    Config.EPOCHS = 1  # Single epoch for demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Ensure directories exist
    Config.setup()

    # Set Seed for reproducibility
    seed_everything(Config.SEED)
    print(f"Working Directory: {Config.WORKING_DIR}")
    print(f"Debug Mode: {Config.DEBUG}")

    # 2. Data Processing Demonstration
    print("\n--- 2. Processing Data ---")
    # Force processing from scratch (load_cached_data=False) to ensure DEBUG sampling is applied
    # This creates parquet files in our demo_working_dir
    train_loader, val_loader, test_loader, vocab_sizes = prepare_data(
        load_cached_data=False
    )

    # Verify Data Integrity
    print(f"Train Loader Batches: {len(train_loader)}")
    sample_batch = next(iter(train_loader))
    x_cont = sample_batch["x_cont"]
    x_cat = sample_batch["x_cat"]
    targets = sample_batch["target"]

    # Assertions to ensure data loading is correct
    assert x_cont.shape[0] <= Config.BATCH_SIZE
    assert x_cat.shape[1] == len(vocab_sizes)
    assert targets.shape[0] == x_cont.shape[0]
    print("Data shapes verified.")

    # 3. Model Initialization Demonstration
    print("\n--- 3. Initializing Model ---")
    num_cont_features = x_cont.shape[1]
    model = AVPFEModel(vocab_sizes=vocab_sizes, num_cont_features=num_cont_features)
    model.to(Config.DEVICE)

    # Verify Forward Pass
    # The model should output 5 logits (one for each stream) per sample
    with torch.no_grad():
        logits = model(x_cont.to(Config.DEVICE), x_cat.to(Config.DEVICE))

    assert logits.shape == (
        x_cont.shape[0],
        5,
    ), f"Expected output shape ({x_cont.shape[0]}, 5), got {logits.shape}"
    print(f"Model Output Shape: {logits.shape} (Verified)")

    # 4. Training Loop Demonstration
    print("\n--- 4. Running Training Loop (1 Epoch) ---")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Simple scheduler for demo purposes
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, epochs=Config.EPOCHS, steps_per_epoch=len(train_loader)
    )

    # Run one epoch of training
    train_loss = train_one_epoch(
        model, train_loader, optimizer, scheduler, Config.DEVICE, criterion
    )
    print(f"Train Loss: {train_loss:.4f}")

    # Run validation
    val_auc = validate(model, val_loader, Config.DEVICE)
    print(f"Validation AUC: {val_auc:.4f}")

    # Verify metrics
    assert not np.isnan(train_loss), "Training loss is NaN"
    assert 0.0 <= val_auc <= 1.0, "AUC score is out of valid range [0, 1]"

    # Save Model (Required for the generate_submission function to work)
    torch.save(model.state_dict(), Config.MODEL_PATH)
    print(f"Model saved to {Config.MODEL_PATH}")

    # 5. Submission Generation Demonstration
    print("\n--- 5. Generating Submission ---")
    # generate_submission uses Config.MODEL_PATH and Config.SUBMISSION_PATH
    # It will reload the data from the cache we created in step 2
    generate_submission()

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {sub_df.shape}")
    print(f"Submission Head:\n{sub_df.head()}")

    # Verify submission size matches debug sample size
    # Note: prepare_data slices the test set to DEBUG_SAMPLE_SIZE when DEBUG=True
    assert (
        len(sub_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Expected {Config.DEBUG_SAMPLE_SIZE} rows in submission, got {len(sub_df)}"

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
