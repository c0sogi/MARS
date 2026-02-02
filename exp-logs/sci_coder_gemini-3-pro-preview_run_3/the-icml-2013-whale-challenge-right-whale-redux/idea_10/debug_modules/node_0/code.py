import os
import shutil
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed, calculate_auc
from library.dataset import (
    load_and_process_data,
    WhaleDataset,
    get_dataloaders,
    get_test_loader,
)
from library.models import WhaleClassifier
from library.train import train_one_epoch, validate


def setup_demo_environment():
    """
    Sets up a lightweight environment for demonstration by creating
    mini-metadata files and overriding Config parameters.
    """
    print(">>> Setting up demo environment...")

    # Define demo directories
    demo_dir = os.path.join(Config.WORKING_DIR, "demo_execution")
    demo_meta_dir = os.path.join(demo_dir, "metadata")

    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_meta_dir, exist_ok=True)

    # Override Config to point to demo directories
    Config.EXPERIMENT_NAME = "demo_execution"
    Config.OUTPUT_DIR = demo_dir
    Config.CACHE_DIR = demo_dir  # Cache npz files here
    Config.CHECKPOINT_DIR = os.path.join(demo_dir, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Initialize directories based on new config
    Config.setup()

    # Create Mini Metadata (Sample 20 rows from original files)
    # This ensures data processing is instant
    splits = {
        "train": (Config.TRAIN_CSV, os.path.join(demo_meta_dir, "train.csv"), 32),
        "val": (Config.VAL_CSV, os.path.join(demo_meta_dir, "val.csv"), 16),
        "test": (Config.TEST_CSV, os.path.join(demo_meta_dir, "test.csv"), 16),
    }

    for split_name, (src_path, dst_path, n_samples) in splits.items():
        if os.path.exists(src_path):
            df = pd.read_csv(src_path)
            # Ensure we have enough samples, or take all
            n = min(len(df), n_samples)
            # Stratify sample if label exists to ensure we have both classes for training
            if "label" in df.columns and split_name == "train":
                # Force at least some positives and negatives
                pos = df[df["label"] == 1].head(n // 2)
                neg = df[df["label"] == 0].head(n - len(pos))
                df_sample = pd.concat([pos, neg])
            else:
                df_sample = df.head(n)

            df_sample.to_csv(dst_path, index=False)
            print(
                f"    Created mini {split_name} metadata with {len(df_sample)} samples."
            )

            # Update Config to point to mini metadata
            if split_name == "train":
                Config.TRAIN_CSV = dst_path
            elif split_name == "val":
                Config.VAL_CSV = dst_path
            elif split_name == "test":
                Config.TEST_CSV = dst_path
        else:
            raise FileNotFoundError(f"Original metadata not found at {src_path}")

    # Override Hyperparameters for Speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8  # Small batch size for demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.SPECAUG_TIME_MASK = 5  # Reduce augmentation intensity
    Config.SPECAUG_FREQ_MASK = 5

    print(">>> Configuration updated for demo.")


def test_data_pipeline():
    """
    Demonstrates data loading, processing, and caching.
    """
    print("\n>>> Testing Data Pipeline...")

    # 1. Test load_and_process_data (Train)
    # This will read audio, compute spectrograms, and cache to .npz
    print("    Processing training data...")
    features, labels = load_and_process_data(
        Config.TRAIN_CSV, "train", load_cached_data=False
    )

    # Assertions
    assert features is not None, "Features should not be None"
    assert labels is not None, "Labels should not be None"
    assert len(features) == len(labels), "Features and labels length mismatch"
    assert features.ndim == 4, f"Expected 4D features (N, C, F, T), got {features.ndim}"
    # Expected shape: (N, 1, 320, T) based on Config.N_MELS=320
    assert features.shape[1] == 1, "Expected 1 channel"
    assert features.shape[2] == Config.N_MELS, f"Expected {Config.N_MELS} mels"

    print(f"    Train features shape: {features.shape}")
    print(f"    Train labels shape: {labels.shape}")

    # 2. Test DataLoader
    print("    Creating DataLoaders...")
    train_loader, val_loader = get_dataloaders(load_cached_data=True)

    # Fetch one batch
    batch_images, batch_labels = next(iter(train_loader))

    assert batch_images.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert batch_images.shape[1] == 1, "Channel dimension mismatch"
    assert batch_labels.shape[0] == Config.BATCH_SIZE, "Label batch size mismatch"

    print("    DataLoader verified successfully.")
    return train_loader, val_loader


def test_model_architecture():
    """
    Demonstrates model instantiation and forward pass.
    """
    print("\n>>> Testing Model Architecture...")

    device = torch.device("cpu")  # Use CPU for simple shape check
    model = WhaleClassifier().to(device)

    # Create dummy input: (Batch, Channel, Freq, Time)
    # Time dimension depends on Config.DURATION * Config.SR / Hop_Length
    # Approx 2.0 * 2000 / 16 = 250 frames.
    dummy_input = torch.randn(2, 1, Config.N_MELS, 250).to(device)

    print("    Running forward pass...")
    output = model(dummy_input)

    # Assertions
    assert output.shape == (2, 1), f"Expected output shape (2, 1), got {output.shape}"

    print("    Model forward pass successful.")
    return model


def test_training_loop(model, train_loader, val_loader):
    """
    Demonstrates the training loop for one epoch and validation.
    """
    print("\n>>> Testing Training Loop...")

    device = torch.device(Config.DEVICE)
    model = model.to(device)

    # Setup simple optimizer/loss for demo
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Train one epoch
    print("    Training for 1 epoch...")
    loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=0)

    assert not np.isnan(loss), "Training loss is NaN"
    print(f"    Epoch 0 Training Loss: {loss:.4f}")

    # Validate
    print("    Validating...")
    val_loss, val_auc = validate(model, val_loader, criterion, device)

    assert not np.isnan(val_loss), "Validation loss is NaN"
    print(f"    Validation Loss: {val_loss:.4f} | AUC: {val_auc:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
    assert os.path.exists(Config.BEST_MODEL_PATH), "Model checkpoint not saved"
    print("    Checkpoint saved successfully.")


def test_inference_pipeline():
    """
    Demonstrates inference on test set and submission file generation.
    """
    print("\n>>> Testing Inference Pipeline...")

    # Ensure test data is processed
    load_and_process_data(Config.TEST_CSV, "test", load_cached_data=False)

    # Get loader
    test_loader, clip_names = get_test_loader(load_cached_data=True)

    device = torch.device(Config.DEVICE)
    model = WhaleClassifier().to(device)

    # Load weights
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    all_probs = []

    print("    Predicting on test set...")
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy().flatten())

    # Verify predictions
    assert len(all_probs) == len(clip_names), "Mismatch between preds and clip names"
    assert all(0.0 <= p <= 1.0 for p in all_probs), "Probabilities out of range [0, 1]"

    # Create submission
    df_sub = pd.DataFrame({"clip": clip_names, "probability": all_probs})
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    # Check format
    df_check = pd.read_csv(Config.SUBMISSION_PATH)
    assert list(df_check.columns) == [
        "clip",
        "probability",
    ], "Submission columns incorrect"
    assert len(df_check) > 0, "Submission file is empty"

    print("    Submission generated and verified.")
    print(df_check.head())


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(Config.SEED)

    # 1. Setup Environment
    setup_demo_environment()

    # 2. Test Data Pipeline
    train_loader, val_loader = test_data_pipeline()

    # 3. Test Model
    model = test_model_architecture()

    # 4. Test Training
    test_training_loop(model, train_loader, val_loader)

    # 5. Test Inference
    test_inference_pipeline()

    print("\n>>> All demonstrations completed successfully.")
