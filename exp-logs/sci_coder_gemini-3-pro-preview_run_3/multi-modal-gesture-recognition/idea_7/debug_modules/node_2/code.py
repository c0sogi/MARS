import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd

# Import provided library modules
from library import config
from library import utils
from library import model
from library import data_loader
from library import train
from library import predict


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup and Configuration Overrides
    print("\n[1] Configuring environment for fast demonstration...")

    # Set seed
    utils.set_seed(42)

    # Override config for speed and isolation
    config.WORKING_DIR = "./working/demo_execution"
    config.CACHE_DIR = os.path.join(config.WORKING_DIR, "cache")
    config.OUTPUT_DIR = os.path.join(config.WORKING_DIR, "outputs")
    config.MODEL_SAVE_PATH = os.path.join(config.OUTPUT_DIR, "demo_model.pth")
    config.SUBMISSION_DIR = config.WORKING_DIR
    config.SUBMISSION_PATH = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    # Ensure directories exist
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # Reduce compute load
    config.DEBUG_SUBSET_SIZE = 10  # Only use 10 sequences
    config.NUM_EPOCHS = 2  # Train for only 2 epochs
    config.BATCH_SIZE = 4  # Small batch size
    config.PATIENCE = 1  # Early stopping

    print(f"Working Directory: {config.WORKING_DIR}")
    print(f"Subset Size: {config.DEBUG_SUBSET_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")

    # 2. Verify Utilities
    print("\n[2] Verifying Utility Functions...")

    # Test Levenshtein
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist_eq = utils.compute_levenshtein(seq1, seq2)
    assert (
        dist_eq == 0
    ), f"Levenshtein distance for identical sequences should be 0, got {dist_eq}"

    seq3 = [1, 2]
    dist_diff = utils.compute_levenshtein(seq1, seq3)
    assert (
        dist_diff == 1
    ), f"Levenshtein distance for [1,2,3] vs [1,2] should be 1, got {dist_diff}"

    # Test Decoding
    # [1, 1, 1, 0, 0, 2, 2, 2, 2, 2] -> 1 (len 3), 0 (len 2), 2 (len 5)
    # Filter background (0) and min_len=4 -> Should only keep 2
    raw_preds = [1, 1, 1, 0, 0, 2, 2, 2, 2, 2]
    decoded = utils.decode_predictions_to_sequence(
        raw_preds, background_id=0, min_len=4
    )
    assert decoded == [2], f"Decoding logic failed. Expected [2], got {decoded}"

    print("Utilities verified successfully.")

    # 3. Verify Model Architecture
    print("\n[3] Verifying Model Architecture...")
    net = model.KC_IRN()
    # Input shape: (Batch, Time, InputDim)
    # config.INPUT_DIM is 193
    dummy_input = torch.randn(2, 100, config.INPUT_DIM)

    # Forward pass
    outputs = net(dummy_input)

    # Check outputs
    assert len(outputs) == 3, "Model should return outputs for 3 stages."
    for i, out in enumerate(outputs):
        # Expected shape: (Batch, NumClasses, Time)
        expected_shape = (2, config.NUM_CLASSES, 100)
        assert (
            out.shape == expected_shape
        ), f"Stage {i+1} output shape mismatch. Expected {expected_shape}, got {out.shape}"
        assert not torch.isnan(out).any(), f"Stage {i+1} output contains NaNs."

    print("Model architecture verified successfully.")

    # 4. Verify Data Loading
    print("\n[4] Verifying Data Loading...")
    # Force re-caching to test processing logic
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        load_cached=False
    )

    # Check Train Loader
    assert len(train_loader) > 0, "Train loader is empty."
    features, labels = next(iter(train_loader))

    # Features: (Batch, WindowSize, InputDim)
    assert features.shape == (
        config.BATCH_SIZE,
        config.WINDOW_SIZE,
        config.INPUT_DIM,
    ), f"Train batch features shape mismatch. Got {features.shape}"
    # Labels: (Batch, WindowSize)
    assert labels.shape == (
        config.BATCH_SIZE,
        config.WINDOW_SIZE,
    ), f"Train batch labels shape mismatch. Got {labels.shape}"

    print(f"Train Batch Shape: {features.shape}")
    print("Data Loading verified successfully.")

    # 5. Run Training Pipeline
    print("\n[5] Running Training Pipeline...")

    # Initialize Trainer manually to use our specific loaders and config
    logger = utils.setup_logger(os.path.join(config.WORKING_DIR, "demo.log"))
    trainer = train.Trainer(config.DEVICE, logger)

    # Run Fit
    trainer.fit(
        train_loader, val_loader, num_epochs=config.NUM_EPOCHS, patience=config.PATIENCE
    )

    # Check if model saved
    assert os.path.exists(config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    print("Training pipeline completed.")

    # 6. Run Inference Pipeline
    print("\n[6] Running Inference Pipeline...")

    # We can use the predict_test_set function, but we need to make sure it uses our modified config
    # predict.predict_test_set() re-loads dataloaders. Since we already cached data in step 4,
    # we can pass load_cached=True.

    # However, predict_test_set() inside library/predict.py uses config.MODEL_SAVE_PATH
    # and config.SUBMISSION_PATH which we overrode.

    predict.predict_test_set(load_cached_data=True)

    # Check submission file
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not generated."

    # Validate submission format
    with open(config.SUBMISSION_PATH, "r") as f:
        lines = f.readlines()

    assert len(lines) > 0, "Submission file is empty."

    # Check first line format: SessionID,label1,label2...
    first_line = lines[0].strip().split(",")
    assert len(first_line) >= 1, "Submission line format incorrect."
    # The first element should be a sample ID (string)
    # Subsequent elements should be integers (gesture IDs)
    if len(first_line) > 1:
        for label in first_line[1:]:
            assert label.isdigit(), f"Submission label is not an integer: {label}"

    print(f"Submission generated with {len(lines)} lines.")
    print("Inference pipeline completed.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
