import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import seed_everything, calculate_lwlrap, calculate_per_class_lwlrap
from library.dataset import AudioDataset
from library.model import HybridEfficientNet
from library.engine import fit_model


def run_demo():
    # ==========================================
    # 1. Configuration & Setup
    # ==========================================
    print("\n[1] Setting up Configuration for Demo...")

    # Override Config for speed and debugging
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 50  # Use only 50 samples
    Config.EPOCHS = 2  # Run only 2 epochs
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 2  # Reduce workers for simple script

    # Ensure reproducibility
    seed_everything(Config.SEED)
    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Epochs: {Config.EPOCHS}")
    print(f"Device: {Config.DEVICE}")

    # ==========================================
    # 2. Metric Verification
    # ==========================================
    print("\n[2] Verifying Metric Calculation (LWLRAP)...")

    # Create a simple deterministic case
    # 3 samples, 3 classes
    # Sample 0: True=[1, 0, 0], Pred=[0.8, 0.1, 0.1] -> Rank 1 correct -> Precision 1/1 = 1.0
    # Sample 1: True=[0, 1, 0], Pred=[0.2, 0.7, 0.1] -> Rank 1 correct -> Precision 1/1 = 1.0
    # Sample 2: True=[1, 1, 0], Pred=[0.6, 0.3, 0.1]
    #    -> Rank 1 (Class 0): Correct. Prec=1/1.
    #    -> Rank 2 (Class 1): Correct. Prec=2/2.
    #    -> Avg Prec for Sample 2 = (1.0 + 1.0) / 2 = 1.0

    y_true_dummy = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
    y_score_dummy = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.6, 0.3, 0.1]])

    score = calculate_lwlrap(y_true_dummy, y_score_dummy)
    print(f"Calculated LWLRAP: {score:.4f}")

    # Assert correctness (Float comparison)
    assert (
        abs(score - 1.0) < 1e-6
    ), f"Metric verification failed! Expected 1.0, got {score}"
    print("Metric verification passed.")

    # ==========================================
    # 3. Dataset Verification
    # ==========================================
    print("\n[3] Verifying AudioDataset...")

    # Initialize Train Dataset
    train_dataset = AudioDataset(split="train", debug=Config.DEBUG)
    print(f"Train Dataset Size: {len(train_dataset)}")

    # Fetch one sample
    image, target = train_dataset[0]

    # Verify Shapes
    # Image: (3, n_mels, time)
    # Time dimension depends on DURATION (30s) * SR (32000) / HOP (512) ≈ 1876
    expected_freq = Config.N_MELS
    expected_channels = 3

    print(f"Sample Image Shape: {image.shape}")
    print(f"Sample Target Shape: {target.shape}")

    assert image.dim() == 3, "Image must be 3D tensor (C, F, T)"
    assert image.shape[0] == expected_channels, f"Expected {expected_channels} channels"
    assert image.shape[1] == expected_freq, f"Expected {expected_freq} mel bins"
    assert (
        target.shape[0] == Config.NUM_CLASSES
    ), f"Expected {Config.NUM_CLASSES} classes in target"
    assert torch.is_tensor(image), "Image must be a tensor"
    assert torch.is_tensor(target), "Target must be a tensor"

    print("Dataset verification passed.")

    # ==========================================
    # 4. Model Architecture Verification
    # ==========================================
    print("\n[4] Verifying HybridEfficientNet Model...")

    model = HybridEfficientNet()
    model.to(Config.DEVICE)
    model.eval()

    # Create dummy batch
    dummy_batch = torch.randn(2, 3, Config.N_MELS, image.shape[2]).to(Config.DEVICE)

    with torch.no_grad():
        logits = model(dummy_batch)

    print(f"Logits Shape: {logits.shape}")

    assert logits.shape == (
        2,
        Config.NUM_CLASSES,
    ), f"Output shape mismatch. Expected (2, {Config.NUM_CLASSES}), got {logits.shape}"

    print("Model architecture verification passed.")

    # ==========================================
    # 5. Training Loop Demonstration
    # ==========================================
    print("\n[5] Running Training Loop (Demo)...")

    # Prepare DataLoaders
    val_dataset = AudioDataset(split="val", debug=Config.DEBUG)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Run Training
    # This uses fit_model from library.engine
    trained_model = fit_model(model, train_loader, val_loader)

    # Verify model file was saved
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Model checkpoint saved successfully at {Config.BEST_MODEL_PATH}")
    else:
        raise FileNotFoundError("Best model checkpoint was not created!")

    # ==========================================
    # 6. Inference Demonstration
    # ==========================================
    print("\n[6] Running Inference on Test Data...")

    # Initialize Test Dataset
    test_dataset = AudioDataset(split="test", debug=Config.DEBUG)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    trained_model.eval()
    predictions = []
    fnames = []

    print(f"Predicting on {len(test_dataset)} test samples...")

    with torch.no_grad():
        for i, (images, filenames) in enumerate(test_loader):
            images = images.to(Config.DEVICE)

            # Forward pass
            logits = trained_model(images)
            probs = torch.sigmoid(logits)

            predictions.append(probs.cpu().numpy())
            fnames.extend(filenames)

            # Just run one batch for demo purposes to save time
            if i == 0:
                print("First batch processed.")
                break

    # Format check
    batch_preds = predictions[0]
    print(f"Prediction Batch Shape: {batch_preds.shape}")
    print(f"First 5 Class Probabilities for first file: {batch_preds[0, :5]}")

    assert (
        batch_preds.min() >= 0.0 and batch_preds.max() <= 1.0
    ), "Probabilities must be in [0, 1]"

    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    run_demo()
