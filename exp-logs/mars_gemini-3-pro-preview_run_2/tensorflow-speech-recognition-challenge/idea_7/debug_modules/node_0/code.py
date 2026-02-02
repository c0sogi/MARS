import os
import shutil
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, calculate_accuracy
from library.dataset import (
    process_and_cache_data,
    get_dataloaders,
    SpeechCommandsDataset,
)
from library.model import TimeResolvedEfficientNet
from library.trainer import Trainer


def main():
    # ==========================================
    # 0. Setup and Configuration Overrides
    # ==========================================
    print("=== 0. Configuration Setup ===")

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Override Config for a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Process only 50 samples
    Config.NUM_EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Ensure clean working directory
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # ==========================================
    # 1. Utility Verification
    # ==========================================
    print("\n=== 1. Utility Verification ===")

    # Test Reproducibility
    set_seed(42)
    rand_a = torch.randn(5)
    set_seed(42)
    rand_b = torch.randn(5)
    assert torch.equal(rand_a, rand_b), "set_seed did not ensure reproducibility"
    print("Seed verification passed.")

    # Test Accuracy Calculation
    # Preds: [Class 1, Class 0], Targets: [1, 0] -> 100% Accuracy
    dummy_outputs = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    dummy_targets = torch.tensor([1, 0])
    acc = calculate_accuracy(dummy_outputs, dummy_targets)
    assert acc == 1.0, f"Accuracy calculation failed. Expected 1.0, got {acc}"
    print(f"Accuracy verification passed: {acc}")

    # ==========================================
    # 2. Data Processing and Loading
    # ==========================================
    print("\n=== 2. Data Processing and Loading ===")

    # Manually trigger processing for the training set
    # This will create .npy files in Config.WORKING_DIR
    print("Processing training data (Debug Subset)...")
    train_features, train_labels = process_and_cache_data(
        Config.TRAIN_METADATA_PATH,
        "train",
        load_cached_data=False,  # Force processing
        is_test=False,
    )

    # Verify Shapes
    # Expected: (N, 1, 128, 101) for 1 second audio at 16kHz with hop_length 160
    # N might be slightly less than DEBUG_SUBSET_SIZE if files were missing (unlikely here)
    print(f"Processed Train Features Shape: {train_features.shape}")
    print(f"Processed Train Labels Shape: {train_labels.shape}")

    assert len(train_features) > 0, "No features processed."
    assert (
        train_features.ndim == 4
    ), f"Expected 4D features (N, C, F, T), got {train_features.ndim}"
    assert train_features.shape[1] == 1, "Expected 1 channel."
    assert (
        train_features.shape[2] == Config.N_MELS
    ), f"Expected {Config.N_MELS} Mel bands."

    # Verify Dataloaders
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # Fetch one batch
    features_batch, labels_batch = next(iter(train_loader))

    print(f"Batch Features Shape: {features_batch.shape}")
    print(f"Batch Labels Shape: {labels_batch.shape}")

    assert features_batch.shape[0] == Config.BATCH_SIZE, "Batch size mismatch."
    assert features_batch.shape[1] == 1, "Channel dimension mismatch."
    assert isinstance(features_batch, torch.Tensor), "Features should be a Tensor."
    assert isinstance(labels_batch, torch.Tensor), "Labels should be a Tensor."

    # ==========================================
    # 3. Model Initialization and Forward Pass
    # ==========================================
    print("\n=== 3. Model Initialization and Forward Pass ===")

    device = torch.device("cpu")  # Use CPU for simple demo assertion
    model = TimeResolvedEfficientNet().to(device)
    model.eval()

    # Create dummy input: (Batch, 1, F, T)
    # F=128, T=101 based on config
    dummy_input = torch.randn(2, 1, Config.N_MELS, 101).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")

    assert output.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Expected output shape (2, {Config.NUM_CLASSES}), got {output.shape}"
    print("Model forward pass successful.")

    # ==========================================
    # 4. Training and Inference Loop
    # ==========================================
    print("\n=== 4. Training and Inference Loop ===")

    trainer = Trainer()

    # 4.1 Run Training (Fit)
    # This uses the subset defined by Config.DEBUG_SUBSET_SIZE and runs for 1 Epoch
    print("Starting Trainer.fit()...")
    trainer.fit(load_cached_data=True)

    # Check if best model was saved
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    assert os.path.exists(best_model_path), "Best model checkpoint was not created."
    print("Training loop completed and model saved.")

    # 4.2 Run Inference (Predict)
    print("Starting Trainer.predict_and_submit()...")
    trainer.predict_and_submit(load_cached_data=True)

    # Check Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not found."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission File Rows: {len(df_sub)}")
    print("Sample Submission:")
    print(df_sub.head())

    # Validate Submission Format
    assert (
        "fname" in df_sub.columns and "label" in df_sub.columns
    ), "Submission columns missing."
    # In Debug mode, we process a subset of test data.
    # The length should match the debug subset size (or total test size if subset > total).
    # Config.DEBUG applies to test set loading in process_and_cache_data as well.
    expected_len = min(Config.DEBUG_SUBSET_SIZE, 6473)  # 6473 is total test files
    assert (
        len(df_sub) == expected_len
    ), f"Submission length mismatch. Expected {expected_len}, got {len(df_sub)}"

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()
