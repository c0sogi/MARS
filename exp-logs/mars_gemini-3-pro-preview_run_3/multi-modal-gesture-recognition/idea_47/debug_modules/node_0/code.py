import os
import shutil
import torch
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import (
    set_seeds,
    levenshtein_distance,
    run_length_encoding,
    filter_segments,
)
from library.data_loader import get_dataloaders
from library.model import DGC_KN
from library.loss import CascadedLoss
from library.train import train_model
from library.predict import generate_submission


def main():
    # ==========================================
    # 1. Setup Environment and Config for Demo
    # ==========================================
    print("=== Setting up Demo Configuration ===")
    demo_dir = "./working/demo_execution"

    # Clean up previous demo runs if they exist
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Override Config paths to use the demo directory
    # This ensures we don't overwrite or rely on existing files in the main working dir
    Config.WORKING_DIR = demo_dir
    Config.TRAIN_CACHE_PATH = os.path.join(demo_dir, "cache", "train.npz")
    Config.VAL_CACHE_PATH = os.path.join(demo_dir, "cache", "val.npz")
    Config.TEST_CACHE_PATH = os.path.join(demo_dir, "cache", "test.npz")
    Config.BEST_MODEL_PATH = os.path.join(demo_dir, "checkpoints", "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(demo_dir, "submission", "submission.csv")

    # Create necessary subdirectories manually since Config only does it at import time
    os.makedirs(os.path.dirname(Config.TRAIN_CACHE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(Config.BEST_MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # Set Debug Mode: Process only 20 samples and train for 1 epoch
    Config.set_debug_mode(subset_size=20, epochs=1)
    set_seeds(42)

    # ==========================================
    # 2. Verify Utility Functions
    # ==========================================
    print("\n=== Verifying Utility Functions ===")

    # Test Levenshtein Distance
    assert levenshtein_distance([1, 2, 3], [1, 2, 3]) == 0.0
    assert levenshtein_distance([1, 2, 3], [1, 2]) == 1.0
    assert levenshtein_distance([], [1]) == 1.0

    # Test Run Length Encoding
    # Sequence: 1,1 -> (1,0,1); 2 -> (2,2,2); 0,0 -> (0,3,4)
    frames = [1, 1, 2, 0, 0]
    segments = run_length_encoding(frames)
    assert segments == [(1, 0, 1), (2, 2, 2), (0, 3, 4)]

    print("Utilities verified successfully.")

    # ==========================================
    # 3. Data Loading Demonstration
    # ==========================================
    print("\n=== Verifying Data Loading ===")
    # This will trigger data processing and caching for the subset
    train_loader, val_loader, test_loader = get_dataloaders(debug=True)

    # Verify Train Loader (Batched, Windowed)
    features, labels = next(iter(train_loader))
    print(f"Train Batch Features Shape: {features.shape}")
    print(f"Train Batch Labels Shape: {labels.shape}")

    # Assertions for Train Loader
    assert features.dim() == 3  # (Batch, Time, Dim)
    assert features.shape[1] == Config.WINDOW_SIZE
    assert features.shape[2] == Config.TOTAL_INPUT_DIM
    assert labels.dim() == 2  # (Batch, Time)

    # Verify Val Loader (Batch=1, Full Sequence)
    val_features, val_labels, val_ids = next(iter(val_loader))
    print(f"Val Batch Features Shape: {val_features.shape}")
    assert val_features.shape[0] == 1

    print("Data Loaders verified.")

    # ==========================================
    # 4. Model Architecture & Loss Demonstration
    # ==========================================
    print("\n=== Verifying Model Architecture ===")
    model = DGC_KN().to(Config.DEVICE)

    # Create dummy input matching the window size
    dummy_input = torch.randn(2, Config.WINDOW_SIZE, Config.TOTAL_INPUT_DIM).to(
        Config.DEVICE
    )
    outputs = model(dummy_input)

    # Assertions for Model Output
    assert "logits_1" in outputs
    assert "logits_2" in outputs
    assert "logits_3" in outputs
    assert outputs["logits_3"].shape == (2, Config.WINDOW_SIZE, Config.NUM_CLASSES)
    print("Model forward pass verified.")

    print("\n=== Verifying Loss Function ===")
    criterion = CascadedLoss().to(Config.DEVICE)
    dummy_targets = torch.randint(0, Config.NUM_CLASSES, (2, Config.WINDOW_SIZE)).to(
        Config.DEVICE
    )

    loss, metrics = criterion(outputs, dummy_targets)
    print(f"Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss)
    assert loss.item() > 0
    print("Loss function verified.")

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("\n=== Running Training Loop (1 Epoch) ===")
    # train_model instantiates the Trainer and runs the fit method
    # We pass debug=True to ensure it uses the debug config settings
    train_model(debug=True, epochs=1)

    # Verify that the model checkpoint was saved
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not saved."
    print(f"Model saved to {Config.BEST_MODEL_PATH}")

    # ==========================================
    # 6. Inference and Submission Demonstration
    # ==========================================
    print("\n=== Running Inference and Generating Submission ===")
    # Explicitly pass the updated paths since default args are evaluated at definition time
    generate_submission(
        model_path=Config.BEST_MODEL_PATH, output_path=Config.SUBMISSION_PATH
    )

    # Verify submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."

    # Check content
    df_sub = pd.read_csv(Config.SUBMISSION_PATH, header=None)
    print(f"Submission file generated with {len(df_sub)} rows.")
    assert len(df_sub) > 0, "Submission file is empty."

    print("Submission generated successfully.")
    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    main()
