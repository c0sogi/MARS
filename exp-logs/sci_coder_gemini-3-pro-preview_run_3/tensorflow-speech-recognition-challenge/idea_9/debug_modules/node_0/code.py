import os
import sys
import shutil
import pandas as pd
import torch
import numpy as np
import warnings

# Import from the provided library
from library.config import TrainConfig, AudioConfig, ModelConfig
from library.preprocess import process_dataset
from library.dataset import CachedSpeechDataset, get_balanced_dataloader
from library.model import SKResNetConformer
from library.engine import run_training, predict_and_submit, set_seed

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("=== Starting Demo Execution ===")

    # 1. Setup Directories and Seeds
    # We use a specific demo directory to avoid conflicts
    DEMO_DIR = "./working/demo_execution"
    DEMO_CACHE_DIR = os.path.join(DEMO_DIR, "cache")
    DEMO_SUBMISSION_PATH = os.path.join(DEMO_DIR, "submission.csv")
    DEMO_MODEL_PATH = os.path.join(DEMO_DIR, "best_model.pth")

    # Clean up previous run if exists
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)
    os.makedirs(DEMO_CACHE_DIR, exist_ok=True)

    # Set seed
    set_seed(42)

    print(f"Working Directory: {DEMO_DIR}")

    # 2. Create Mini-Datasets for Speed
    # We load the full metadata but only save a tiny subset for this demo
    print("\n--- Preparing Mini-Datasets ---")

    full_train_df = pd.read_csv(TrainConfig.train_metadata_path)
    full_val_df = pd.read_csv(TrainConfig.val_metadata_path)
    full_test_df = pd.read_csv(TrainConfig.test_metadata_path)

    # Sample small subsets
    mini_train = full_train_df.head(32).copy()  # One batch size
    mini_val = full_val_df.head(16).copy()
    mini_test = full_test_df.head(16).copy()

    # Save to demo directory
    mini_train_path = os.path.join(DEMO_DIR, "train.csv")
    mini_val_path = os.path.join(DEMO_DIR, "val.csv")
    mini_test_path = os.path.join(DEMO_DIR, "test.csv")

    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    mini_test.to_csv(mini_test_path, index=False)

    print(f"Created mini train set: {len(mini_train)} samples")
    print(f"Created mini val set:   {len(mini_val)} samples")
    print(f"Created mini test set:  {len(mini_test)} samples")

    # 3. Override Configuration
    # We modify the TrainConfig class attributes directly to affect the library modules
    print("\n--- Overriding Configuration ---")
    TrainConfig.working_dir = DEMO_DIR
    TrainConfig.cache_dir = DEMO_CACHE_DIR
    TrainConfig.train_metadata_path = mini_train_path
    TrainConfig.val_metadata_path = mini_val_path
    TrainConfig.test_metadata_path = mini_test_path
    TrainConfig.model_save_path = DEMO_MODEL_PATH
    TrainConfig.submission_path = DEMO_SUBMISSION_PATH

    # Speed optimizations
    TrainConfig.epochs = 1
    TrainConfig.batch_size = 8
    TrainConfig.debug = True
    TrainConfig.debug_subset_size = 32  # Ensure we use all of our mini set
    TrainConfig.num_workers = 0  # Avoid multiprocessing overhead for tiny data

    # 4. Run Preprocessing (Caching)
    print("\n--- Running Preprocessing (Caching) ---")
    # Process Train
    process_dataset(mini_train_path, DEMO_CACHE_DIR, load_cached_data=False)
    # Process Val
    process_dataset(mini_val_path, DEMO_CACHE_DIR, load_cached_data=False)
    # Process Test
    process_dataset(mini_test_path, DEMO_CACHE_DIR, load_cached_data=False)

    # Verify cache population
    cached_files = os.listdir(DEMO_CACHE_DIR)
    print(f"Cached files count: {len(cached_files)}")
    assert len(cached_files) > 0, "Cache directory is empty after processing!"

    # 5. Verify Dataset Logic
    print("\n--- Verifying Dataset Logic ---")
    dataset = CachedSpeechDataset(mini_train, transform=None)
    features, label = dataset[0]

    print(f"Feature Shape: {features.shape}")
    print(f"Label: {label}")

    # Expected shape: (3 channels, 64 mels, 101 time steps)
    # Time steps = 1.0s * 16000sr / 160hop + 1 approx 101
    assert features.shape[0] == 3, f"Expected 3 channels, got {features.shape[0]}"
    assert (
        features.shape[1] == AudioConfig.n_mels
    ), f"Expected {AudioConfig.n_mels} mels, got {features.shape[1]}"
    # Allow slight variation in time dimension due to padding logic, but usually 101
    assert (
        features.shape[2] == 101
    ), f"Expected 101 time frames, got {features.shape[2]}"
    assert isinstance(label, torch.Tensor), "Label should be a torch Tensor"

    # 6. Verify Model Logic
    print("\n--- Verifying Model Logic ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SKResNetConformer().to(device)

    # Create dummy batch
    dummy_input = torch.randn(2, 3, 64, 101).to(device)
    dummy_output = model(dummy_input)

    print(f"Model Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (
        2,
        ModelConfig.num_classes,
    ), f"Expected output shape (2, {ModelConfig.num_classes}), got {dummy_output.shape}"

    # 7. Run Training Engine
    print("\n--- Executing Training Loop ---")
    # This will use the overridden TrainConfig paths and parameters
    run_training()

    # Verify model was saved
    assert os.path.exists(DEMO_MODEL_PATH), "Model checkpoint was not saved!"
    print("Training completed and model saved.")

    # 8. Run Inference Engine
    print("\n--- Executing Inference Loop ---")
    predict_and_submit()

    # Verify submission file
    assert os.path.exists(DEMO_SUBMISSION_PATH), "Submission file was not created!"

    # Check submission content
    df_sub = pd.read_csv(DEMO_SUBMISSION_PATH)
    print(f"Submission rows: {len(df_sub)}")
    assert len(df_sub) == len(
        mini_test
    ), f"Expected {len(mini_test)} predictions, got {len(df_sub)}"
    assert (
        "fname" in df_sub.columns and "label" in df_sub.columns
    ), "Submission columns missing"

    print("\n=== Demo Execution Completed Successfully ===")


if __name__ == "__main__":
    main()
