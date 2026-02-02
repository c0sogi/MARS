import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library files
from library.config import Config, set_seed
from library.utils import get_weighted_sampler
from library.dataset import (
    get_spectrogram_transform,
    process_audio_file,
    get_dataloaders,
)
from library.model import AudioEfficientNet
from library.train import run_training

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def test_weighted_sampler():
    print("\n=== Testing Weighted Sampler ===")
    # Create a dummy dataframe with known imbalance
    # Class 'A': 2 samples, Class 'B': 1 sample
    data = {"label": ["A", "A", "B"], "file_path": ["f1", "f2", "f3"]}
    df = pd.DataFrame(data)

    sampler = get_weighted_sampler(df, label_col="label")

    # Extract weights assigned to samples
    weights = sampler.weights
    print(f"Sample weights: {weights}")

    # Logic check:
    # Count(A) = 2, Weight(A) = 1/2 = 0.5
    # Count(B) = 1, Weight(B) = 1/1 = 1.0
    # Weights should be [0.5, 0.5, 1.0]

    assert torch.isclose(
        weights[0], torch.tensor(0.5, dtype=torch.double)
    ), "Weight for class A incorrect"
    assert torch.isclose(
        weights[2], torch.tensor(1.0, dtype=torch.double)
    ), "Weight for class B incorrect"
    print("WeightedSampler logic verified.")


def test_data_processing():
    print("\n=== Testing Data Processing Pipeline ===")

    # 1. Test Transform Generation
    transform = get_spectrogram_transform()
    assert isinstance(
        transform, torch.nn.Sequential
    ), "Transform should be a Sequential module"

    # 2. Test Single File Processing
    # Load train metadata to get a valid file path
    df_train = pd.read_csv(Config.TRAIN_CSV)
    if len(df_train) == 0:
        raise ValueError("Train metadata is empty.")

    sample_row = df_train.iloc[0]
    full_path = os.path.join(Config.INPUT_DIR, sample_row["file_path"])

    print(f"Processing file: {full_path}")

    # Process the file
    spec = process_audio_file(full_path, transform)

    # Check shape: (1, n_mels, time)
    # Time dimension depends on hop_length and duration.
    # For 16000Hz, 1s, hop=160 -> ~101 frames
    print(f"Spectrogram shape: {spec.shape}")

    assert spec.dim() == 3, "Spectrogram must be 3D tensor (C, F, T)"
    assert spec.size(0) == 1, "Channel dimension must be 1"
    assert spec.size(1) == Config.N_MELS, f"Freq dimension must be {Config.N_MELS}"

    # Check normalization (mean should be close to 0, std close to 1)
    print(f"Mean: {spec.mean():.4f}, Std: {spec.std():.4f}")

    print("Data processing pipeline verified.")


def test_dataloaders_and_model():
    print("\n=== Testing DataLoaders and Model Forward Pass ===")

    # Use debug=True to load a tiny subset of data
    # This also tests the caching mechanism in get_data_cache
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Get a batch
    inputs, targets = next(iter(train_loader))
    print(f"Batch input shape: {inputs.shape}")
    print(f"Batch target shape: {targets.shape}")

    # Verify shapes
    # Batch size might be smaller if dataset is tiny, but Config.BATCH_SIZE is 64
    current_batch_size = inputs.size(0)
    assert inputs.shape == (
        current_batch_size,
        1,
        Config.N_MELS,
        101,
    ), f"Unexpected input shape: {inputs.shape}"

    # Initialize Model
    device = Config.DEVICE
    model = AudioEfficientNet(num_classes=Config.NUM_CLASSES)
    model.to(device)

    # Forward pass
    inputs = inputs.to(device)
    outputs = model(inputs)

    print(f"Model output shape: {outputs.shape}")

    assert outputs.shape == (
        current_batch_size,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"

    print("DataLoaders and Model architecture verified.")
    return test_loader


def test_training_loop():
    print("\n=== Testing Training Loop ===")

    # Run training for 1 epoch with debug data
    # This tests the Trainer class, optimizer, scheduler, and saving logic
    best_acc = run_training(
        debug=True,
        load_cached_data=True,  # Use the cache generated in previous step
        epochs=1,
        patience=1,
        batch_size=16,
    )

    print(f"Training run finished. Best Debug Accuracy: {best_acc}")

    # Verify model file was saved
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Model successfully saved to {Config.MODEL_SAVE_PATH}")
    else:
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")


def generate_submission(test_loader):
    print("\n=== Generating Sample Submission ===")

    device = Config.DEVICE

    # Load the best model
    model = AudioEfficientNet(num_classes=Config.NUM_CLASSES)
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()

    predictions = []
    fnames = []

    # We need to map the indices back to filenames.
    # In a real scenario, the Dataset should return fnames or we iterate the metadata.
    # For this demo, we will read the test metadata (debug version) to match order.
    df_test = pd.read_csv(Config.TEST_CSV).iloc[
        :100
    ]  # Matching debug size in dataset.py

    print("Running inference on test set...")
    with torch.no_grad():
        # Iterate through loader
        # Note: The loader order matches the dataframe order because shuffle=False for test
        batch_idx = 0
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)

            # Get predicted class indices
            _, preds = torch.max(outputs, 1)

            # Convert to labels
            pred_labels = [Config.ID2LABEL[idx.item()] for idx in preds]
            predictions.extend(pred_labels)

            batch_idx += 1

    # Ensure lengths match (handling potential drop_last=False implication)
    # In debug mode, we processed 100 samples.
    predictions = predictions[: len(df_test)]

    # Create submission DataFrame
    submission = pd.DataFrame({"fname": df_test["fname"], "label": predictions})

    print("Sample Submission Head:")
    print(submission.head())

    # Save submission
    sub_path = Config.SUBMISSION_PATH
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


if __name__ == "__main__":
    # 1. Set Seed
    set_seed(Config.SEED)

    # 2. Test Utility
    test_weighted_sampler()

    # 3. Test Data Processing
    test_data_processing()

    # 4. Test Components
    test_loader = test_dataloaders_and_model()

    # 5. Run Training Integration Test
    test_training_loop()

    # 6. Generate Submission
    generate_submission(test_loader)

    print("\nAll demonstrations completed successfully.")
