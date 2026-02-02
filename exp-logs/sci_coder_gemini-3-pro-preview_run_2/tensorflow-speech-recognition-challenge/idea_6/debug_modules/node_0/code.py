import os
import pandas as pd
import torch
import numpy as np
import warnings

# Import from the provided library
from library import config
from library import utils
from library import dataset
from library import model as model_lib
from library import train
from library import inference

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def create_subset_metadata(original_path, new_path, n=100):
    """Creates a subset of the metadata CSV for rapid demonstration."""
    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Original metadata not found: {original_path}")

    df = pd.read_csv(original_path)
    # Sample n rows or take all if len < n
    n_sample = min(n, len(df))
    df_subset = df.sample(n=n_sample, random_state=config.SEED).reset_index(drop=True)

    # Ensure directory exists
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    df_subset.to_csv(new_path, index=False)
    print(f"Created subset metadata at {new_path} with {len(df_subset)} samples.")


def main():
    print("=== Starting Speech Command Recognition Demo ===\n")

    # 1. Setup
    utils.set_seed(config.SEED)
    device = torch.device(config.DEVICE)
    print(f"Device: {device}")

    # Define paths for temporary subsets
    subset_dir = os.path.join(config.WORKING_DIR, "demo_subsets")
    train_subset_path = os.path.join(subset_dir, "train.csv")
    val_subset_path = os.path.join(subset_dir, "val.csv")
    test_subset_path = os.path.join(subset_dir, "test.csv")

    # 2. Create Data Subsets for Speed
    print("\n--- Preparing Data Subsets ---")
    create_subset_metadata(config.TRAIN_METADATA_PATH, train_subset_path, n=128)
    create_subset_metadata(config.VAL_METADATA_PATH, val_subset_path, n=64)
    create_subset_metadata(config.TEST_METADATA_PATH, test_subset_path, n=64)

    # Monkey-patch config paths to use subsets
    config.TRAIN_METADATA_PATH = train_subset_path
    config.VAL_METADATA_PATH = val_subset_path
    config.TEST_METADATA_PATH = test_subset_path

    # Adjust config for demo
    config.BATCH_SIZE = 32
    config.NUM_WORKERS = 2

    # 3. Verify Dataset and DataLoader
    print("\n--- Verifying Dataset & DataLoader ---")
    train_loader, val_loader = dataset.get_dataloaders(
        train_csv=config.TRAIN_METADATA_PATH,
        val_csv=config.VAL_METADATA_PATH,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    # Fetch one batch
    specs, labels = next(iter(train_loader))

    print(f"Batch Spectrogram Shape: {specs.shape}")
    print(f"Batch Labels Shape: {labels.shape}")

    # Assertions
    # Shape: (Batch, Channels, Freq, Time)
    # Channels = 1
    # Freq = N_MELS = 64
    # Time depends on padding/stft, roughly 101 for 1s audio with given params
    assert specs.shape[0] == config.BATCH_SIZE, "Incorrect batch size"
    assert specs.shape[1] == 1, "Spectrogram should have 1 channel"
    assert specs.shape[2] == config.N_MELS, f"Expected {config.N_MELS} mel bins"
    assert labels.max() < config.NUM_CLASSES, "Label index out of bounds"

    print("Dataset verification successful.")

    # 4. Verify Model Architecture
    print("\n--- Verifying Model Architecture ---")
    model = model_lib.ConvNeXtSpeech(num_classes=config.NUM_CLASSES, pretrained=False)
    model = model.to(device)

    num_params = utils.count_parameters(model)
    print(f"Model initialized. Trainable parameters: {num_params}")

    # Forward pass check
    specs = specs.to(device)
    with torch.no_grad():
        outputs = model(specs)

    print(f"Model Output Shape: {outputs.shape}")

    assert outputs.shape == (
        config.BATCH_SIZE,
        config.NUM_CLASSES,
    ), "Output shape mismatch"
    print("Model verification successful.")

    # 5. Run Training Demo
    print("\n--- Running Training Loop (Demo) ---")
    # We run for 2 epochs to verify the loop mechanics and checkpointing
    # We reduce patience to ensure it doesn't run long if logic is wrong
    train.run_training(
        epochs=2,
        batch_size=config.BATCH_SIZE,
        learning_rate=1e-3,  # Higher LR for demo convergence check
        patience=2,
    )

    # Check if model checkpoint was created
    if not os.path.exists(config.MODEL_CHECKPOINT_PATH):
        raise FileNotFoundError("Model checkpoint was not created after training.")
    print(
        f"Training demo completed. Checkpoint found at {config.MODEL_CHECKPOINT_PATH}"
    )

    # 6. Run Inference Demo
    print("\n--- Running Inference (Demo) ---")
    submission_path = os.path.join(config.WORKING_DIR, "demo_submission.csv")

    df_sub = inference.generate_submission(
        model_path=config.MODEL_CHECKPOINT_PATH,
        output_path=submission_path,
        batch_size=config.BATCH_SIZE,
        device=config.DEVICE,
    )

    # Verify Submission
    print(f"Submission shape: {df_sub.shape}")
    print("First 5 rows:")
    print(df_sub.head())

    # Assertions
    expected_rows = 64  # Size of our test subset
    assert (
        len(df_sub) == expected_rows
    ), f"Expected {expected_rows} predictions, got {len(df_sub)}"
    assert (
        "fname" in df_sub.columns and "label" in df_sub.columns
    ), "Missing columns in submission"
    assert (
        df_sub["label"].isin(config.TARGET_LABELS).all()
    ), "Invalid labels found in submission"

    print("Inference verification successful.")
    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()
