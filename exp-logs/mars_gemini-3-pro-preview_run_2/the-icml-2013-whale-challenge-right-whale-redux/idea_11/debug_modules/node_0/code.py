import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, calculate_roc_auc, AverageMeter
from library.dataset import load_dataset_data, WhaleDataset
from library.models import get_model
from library.train import train_one_epoch, validate


def run_demo():
    print("=== Starting Right Whale Detection Demo ===")

    # 1. Setup & Configuration Overrides for Speed
    print("\n[1] Configuring environment for rapid demonstration...")
    seed_everything(Config.SEED)

    # Override Config for the demo to run fast
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Only use 50 samples
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.MODEL_NAMES = ["resnet34"]  # Use one model for demo
    Config.PRETRAINED = False  # Avoid downloading weights

    # Ensure output directory exists
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. Data Loading Demonstration
    print("\n[2] Demonstrating Data Loading...")
    # Load training data (subset due to DEBUG=True)
    train_data, train_labels, _ = load_dataset_data(
        Config.TRAIN_CSV, "train_demo", load_cached_data=False
    )

    print(f"Loaded train data shape: {train_data.shape}")
    print(f"Loaded train labels shape: {train_labels.shape}")

    # Assertions to verify data loading
    assert (
        len(train_data) == Config.DEBUG_SUBSET_SIZE
    ), f"Expected {Config.DEBUG_SUBSET_SIZE} samples, got {len(train_data)}"
    assert train_data.ndim == 2, "Audio data should be 2D (Samples, Time)"
    assert train_labels is not None, "Labels should not be None for training data"

    # 3. Dataset & Transform Demonstration
    print("\n[3] Demonstrating Dataset & Transforms...")
    train_dataset = WhaleDataset(train_data, train_labels, is_training=True)

    # Get a single item
    spec, label = train_dataset[0]
    print(f"Single item spectrogram shape: {spec.shape}")
    print(f"Single item label: {label}")

    # Assertions for Dataset
    # Expected shape: (1, n_mels, time_steps)
    # Time steps depends on audio length (2s * 2000Hz = 4000 samples) and hop_length (64)
    # Approx width = 4000 / 64 ≈ 63
    assert spec.ndim == 3, "Spectrogram should be 3D (C, F, T)"
    assert spec.shape[0] == 1, "Should have 1 channel"
    assert spec.shape[1] == Config.N_MELS, f"Should have {Config.N_MELS} Mel bands"
    assert isinstance(label, torch.Tensor), "Label should be a tensor"

    # 4. DataLoader & Batching
    print("\n[4] Demonstrating DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    # Fetch one batch
    batch_images, batch_labels = next(iter(train_loader))
    print(f"Batch images shape: {batch_images.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")

    assert batch_images.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert batch_images.shape[1] == 1, "Channel dimension mismatch"

    # 5. Model Instantiation & Forward Pass
    print("\n[5] Demonstrating Model & Forward Pass...")
    model_name = Config.MODEL_NAMES[0]
    model = get_model(model_name, pretrained=Config.PRETRAINED)
    model = model.to(device)

    # Run forward pass
    batch_images = batch_images.to(device)
    with torch.no_grad():
        logits = model(batch_images)

    print(f"Logits shape: {logits.shape}")

    # Assertions for Model
    assert logits.shape == (Config.BATCH_SIZE, 1), "Output shape should be (Batch, 1)"
    assert not torch.isnan(logits).any(), "Model output contains NaNs"

    # 6. Training Loop (Single Epoch)
    print("\n[6] Demonstrating Training Loop (1 Epoch)...")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Train one epoch
    train_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, device, epoch=0
    )
    print(f"Training Loss: {train_loss:.4f}")

    assert train_loss >= 0, "Training loss should be non-negative"

    # 7. Validation Loop
    print("\n[7] Demonstrating Validation Loop...")
    # Create a small validation set
    val_data, val_labels, _ = load_dataset_data(
        Config.VAL_CSV, "val_demo", load_cached_data=False
    )
    val_dataset = WhaleDataset(val_data, val_labels, is_training=False)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    val_loss, val_auc = validate(model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")

    # Note: AUC might be 0.5 if the small subset only has one class, but the function handles it.
    assert isinstance(val_loss, float), "Validation loss should be a float"
    assert 0.0 <= val_auc <= 1.0, "AUC should be between 0 and 1"

    # 8. Inference & Submission Generation
    print("\n[8] Demonstrating Inference & Submission...")
    # Load test data
    test_data, _, test_clips = load_dataset_data(
        Config.TEST_CSV, "test_demo", load_cached_data=False
    )
    test_dataset = WhaleDataset(test_data, None, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds.extend(probs)

    preds = np.array(preds)

    # Create submission dataframe
    df_sub = pd.DataFrame({"clip": test_clips, "probability": preds})

    print("Sample Submission Head:")
    print(df_sub.head())

    # Verify submission format
    assert "clip" in df_sub.columns and "probability" in df_sub.columns
    assert len(df_sub) == len(test_clips)
    assert df_sub["probability"].min() >= 0 and df_sub["probability"].max() <= 1

    # Save submission
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()
