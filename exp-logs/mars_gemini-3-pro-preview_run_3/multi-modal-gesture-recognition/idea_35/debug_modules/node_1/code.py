import os
import torch
import numpy as np
import pandas as pd
import shutil
import logging

# Import from the provided library
from library.config import config
from library.utils import (
    levenshtein_distance,
    run_length_encoding,
    process_predictions_for_submission,
    compute_sequence_accuracy,
)
from library.dataset import GestureDataset, get_dataloaders
from library.model import RHCKN
from library.engine import Trainer
from library.predict import SlidingWindowPredictor


def setup_demo_environment():
    """
    Sets up a temporary environment for the demo to run quickly
    without affecting the main workspace or processing the full dataset.
    """
    print(">>> Setting up Demo Environment...")

    # Define demo paths
    demo_dir = "./working/demo_env"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    metadata_dir = os.path.join(demo_dir, "metadata")
    cache_dir = os.path.join(demo_dir, "cache")
    submission_dir = os.path.join(demo_dir, "submission")
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    # Create Mini Metadata (Top 5 samples from each split)
    # This ensures we don't process the entire dataset for this demo
    splits = {
        "train": config.TRAIN_METADATA,
        "val": config.VAL_METADATA,
        "test": config.TEST_METADATA,
    }

    new_paths = {}

    for split_name, original_path in splits.items():
        if os.path.exists(original_path):
            df = pd.read_csv(original_path)
            # Take a small subset (e.g., 5 samples)
            mini_df = df.head(6)
            save_path = os.path.join(metadata_dir, f"{split_name}.csv")
            mini_df.to_csv(save_path, index=False)
            new_paths[split_name] = save_path
            print(
                f"    Created mini {split_name} metadata with {len(mini_df)} samples."
            )
        else:
            # Fallback if original metadata doesn't exist (should not happen based on prompt)
            print(
                f"    WARNING: Original {split_name} metadata not found at {original_path}"
            )
            new_paths[split_name] = original_path

    # Override Config Global Object
    # We modify the singleton instance directly
    config.WORKING_DIR = demo_dir
    config.CACHE_DIR = cache_dir
    config.SUBMISSION_DIR = submission_dir
    config.SUBMISSION_FILE = os.path.join(submission_dir, "submission.csv")
    config.MODEL_SAVE_PATH = os.path.join(demo_dir, "best_model.pth")

    config.TRAIN_METADATA = new_paths["train"]
    config.VAL_METADATA = new_paths["val"]
    config.TEST_METADATA = new_paths["test"]

    # Reduce Model Complexity for Speed
    config.RNN_HIDDEN_SIZE = 32
    config.RNN_LAYERS = 1
    config.TCN_CHANNELS = 16
    config.TCN_KERNEL_SIZE = 3
    config.TCN_DILATIONS = [1, 2]  # Reduced depth

    # Training params for demo
    config.BATCH_SIZE = 2
    config.EPOCHS = 2
    config.EARLY_STOPPING_PATIENCE = 2

    print(">>> Demo Environment Configured.")


def demo_utils():
    """
    Demonstrates and verifies utility functions.
    """
    print("\n>>> Demonstrating Library Utils...")

    # 1. Levenshtein Distance
    seq1 = [1, 2, 3]
    seq2 = [1, 2, 3]
    dist_eq = levenshtein_distance(seq1, seq2)
    assert dist_eq == 0, f"Distance should be 0 for identical sequences, got {dist_eq}"

    seq3 = [1, 2]
    dist_diff = levenshtein_distance(seq1, seq3)
    assert dist_diff == 1, f"Distance should be 1 (deletion), got {dist_diff}"
    print("    Levenshtein Distance logic verified.")

    # 2. Run Length Encoding & Processing
    # Create a sequence: Background(0) -> Class(1) -> Background(0) -> Class(2)
    # Lengths: 5 frames of 0, 10 frames of 1, 3 frames of 0, 4 frames of 2
    # Note: MIN_GESTURE_DURATION is default 5. So Class 2 (4 frames) should be filtered out.
    preds = np.concatenate([np.zeros(5), np.ones(10), np.zeros(3), np.full(4, 2)])

    # Expected result: [1] (Class 2 is filtered because 4 < 5)
    result = process_predictions_for_submission(preds, background_class=0)

    # Check if filtering works (Class 2 should be gone)
    # Check if background is removed
    assert 1 in result, "Class 1 should be detected."
    assert 2 not in result, "Class 2 should be filtered out (duration 4 < 5)."
    assert 0 not in result, "Background should be removed."
    assert result == [1], f"Expected [1], got {result}"

    print("    Prediction post-processing logic verified.")


def demo_dataset_and_loader():
    """
    Demonstrates dataset loading, processing, and batching.
    """
    print("\n>>> Demonstrating Dataset & DataLoader...")

    # Force reload of cache since we changed metadata paths
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=config.BATCH_SIZE
    )

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches: {len(val_loader)}")

    # Fetch one batch
    features, labels = next(iter(train_loader))

    # Verify Shapes
    # Features: (Batch, Window, InputDim)
    # InputDim = 20 joints * 9 (pos,vel,acc) + 13 MFCC = 193
    expected_input_dim = 193

    print(f"    Batch Features Shape: {features.shape}")
    print(f"    Batch Labels Shape: {labels.shape}")

    assert features.dim() == 3, "Features should be 3D (Batch, Time, Feat)"
    assert (
        features.shape[2] == expected_input_dim
    ), f"Expected {expected_input_dim} features, got {features.shape[2]}"
    assert labels.dim() == 2, "Labels should be 2D (Batch, Time)"
    assert (
        labels.shape[1] == config.WINDOW_SIZE
    ), f"Labels time dim should match window size {config.WINDOW_SIZE}"

    print("    Dataset shapes verified.")
    return train_loader, val_loader, test_loader


def demo_model_forward():
    """
    Demonstrates model instantiation and forward pass.
    """
    print("\n>>> Demonstrating Model Architecture...")

    model = RHCKN().to(torch.device("cpu"))

    # Create dummy input: (Batch=2, Time=64, Feat=193)
    dummy_input = torch.randn(2, config.WINDOW_SIZE, 193)

    # Forward pass
    logits1, logits2, logits3 = model(dummy_input)

    print(f"    Logits1 Shape: {logits1.shape}")
    print(f"    Logits2 Shape: {logits2.shape}")
    print(f"    Logits3 Shape: {logits3.shape}")

    # Verify Deep Supervision Outputs
    for i, l in enumerate([logits1, logits2, logits3]):
        assert l.shape == (
            2,
            config.WINDOW_SIZE,
            config.NUM_CLASSES,
        ), f"Logits{i+1} shape mismatch. Expected (2, {config.WINDOW_SIZE}, {config.NUM_CLASSES})"

    print("    Model forward pass verified.")


def demo_training(train_loader, val_loader):
    """
    Demonstrates the training loop.
    """
    print("\n>>> Demonstrating Training Loop...")

    # Use CPU for demo stability/compatibility, or CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Training on device: {device}")

    trainer = Trainer(device=device)

    # Run fit for limited epochs (defined in setup as 2)
    trainer.fit(train_loader, val_loader)

    # Check if model was saved
    assert os.path.exists(
        config.MODEL_SAVE_PATH
    ), "Model checkpoint file was not created."
    print(f"    Training complete. Model saved to {config.MODEL_SAVE_PATH}")


def demo_prediction(test_loader):
    """
    Demonstrates inference and submission generation.
    """
    print("\n>>> Demonstrating Prediction & Submission...")

    predictor = SlidingWindowPredictor(model_path=config.MODEL_SAVE_PATH)

    # Generate predictions
    results = predictor.generate_predictions(test_loader)

    # Verify results
    assert len(results) > 0, "No predictions generated."
    print(f"    Generated {len(results)} prediction lines.")
    print(f"    Sample prediction: {results[0]}")

    # Save submission
    predictor.save_submission(results)
    assert os.path.exists(config.SUBMISSION_FILE), "Submission file not created."

    # Verify file content
    with open(config.SUBMISSION_FILE, "r") as f:
        lines = f.readlines()
        assert len(lines) == len(results), "Submission file line count mismatch."

    print(f"    Submission saved to {config.SUBMISSION_FILE}")


if __name__ == "__main__":
    # Ensure reproducibility
    config.set_seed()

    # 1. Setup Demo Environment (Mini Data, Fast Config)
    setup_demo_environment()

    # 2. Verify Utils
    demo_utils()

    # 3. Verify Data Loading
    # Note: This will process the mini dataset and cache it
    train_loader, val_loader, test_loader = demo_dataset_and_loader()

    # 4. Verify Model
    demo_model_forward()

    # 5. Run Training Demo
    demo_training(train_loader, val_loader)

    # 6. Run Prediction Demo
    demo_prediction(test_loader)

    print("\n>>> All demonstrations completed successfully.")
