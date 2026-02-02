import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

# Import provided library modules
from library.utils import seed_everything, weighted_auc_score
from library.dataset import StegoDataset, get_transforms
from library.model import MonoResidualEfficientNet
from library.engine import train_one_epoch, validate, predict_tta


def run_demo():
    print("--- Starting Steganography Detection Pipeline Demo ---")

    # 1. Setup
    SEED = 42
    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Define paths
    INPUT_DIR = "./input"
    METADATA_PATH = "./metadata/train.csv"

    # 2. Data Loading Demonstration
    print("\n[1/5] Demonstrating Data Loading...")

    # Load metadata
    if not os.path.exists(METADATA_PATH):
        raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")

    full_df = pd.read_csv(METADATA_PATH)

    # OPTIMIZATION: Use a tiny subset for demonstration speed
    subset_size = 16  # Small enough for quick CPU processing if needed
    train_subset_df = full_df.iloc[:subset_size].copy()
    val_subset_df = full_df.iloc[subset_size : subset_size * 2].copy()

    print(
        f"Created subset of {len(train_subset_df)} training samples and {len(val_subset_df)} validation samples."
    )

    # Instantiate Datasets
    # We use get_transforms('train') for training to show augmentation pipeline
    train_dataset = StegoDataset(
        df=train_subset_df, input_dir=INPUT_DIR, transform=get_transforms("train")
    )

    # No transforms for validation
    val_dataset = StegoDataset(df=val_subset_df, input_dir=INPUT_DIR, transform=None)

    # Instantiate DataLoaders
    batch_size = 4
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # Verify Data Shapes
    images, labels = next(iter(train_loader))
    print(f"Batch Image Shape: {images.shape}")  # Should be (B, 1, H, W)
    print(f"Batch Label Shape: {labels.shape}")  # Should be (B,)

    # Logic Verification: Check input dimensions
    # Images are 512x512, Y-channel only -> (B, 1, 512, 512)
    assert images.dim() == 4, "Image tensor should be 4-dimensional (B, C, H, W)"
    assert images.shape[1] == 1, "Image tensor should have 1 channel (Y channel)"
    assert labels.dim() == 1, "Labels should be 1-dimensional"

    print("Data Loading verified successfully.")

    # 3. Model Initialization Demonstration
    print("\n[2/5] Demonstrating Model Initialization...")

    # Initialize model
    # Using pretrained=False for speed and offline reliability in this demo
    model = MonoResidualEfficientNet(
        model_name="efficientnet_b0", pretrained=False, num_classes=1
    )
    model.to(device)

    # Logic Verification: Check HPF layer freezing
    # The HPF layer weights should not require gradients
    hpf_requires_grad = any(param.requires_grad for param in model.hpf.parameters())
    assert (
        not hpf_requires_grad
    ), "HPF Layer parameters should be frozen (requires_grad=False)"

    # Logic Verification: Forward pass shape
    images = images.to(device)
    outputs = model(images)
    print(f"Model Output Shape: {outputs.shape}")

    assert outputs.shape == (
        batch_size,
        1,
    ), f"Expected output shape ({batch_size}, 1), got {outputs.shape}"

    print("Model initialization and forward pass verified successfully.")

    # 4. Training Loop Demonstration
    print("\n[3/5] Demonstrating Training & Validation Loop...")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    # Run one training epoch
    # We pass None for scheduler to keep it simple
    train_loss = train_one_epoch(
        model=model,
        dataloader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=None,
        label_smoothing=0.0,
    )

    print(f"Training Epoch Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss should not be NaN"

    # Run validation
    val_loss, val_score = validate(
        model=model, dataloader=val_loader, criterion=criterion, device=device
    )

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Weighted AUC: {val_score:.4f}")

    assert not np.isnan(val_loss), "Validation loss should not be NaN"
    assert 0.0 <= val_score <= 1.0, "AUC score must be between 0 and 1"

    print("Training and Validation loops verified successfully.")

    # 5. Metric Logic Verification
    print("\n[4/5] Verifying Weighted AUC Metric...")

    # Case A: Perfect prediction
    y_true_perfect = [0, 0, 1, 1]
    y_score_perfect = [0.1, 0.2, 0.9, 0.8]  # Probabilities clearly separating classes
    score_perfect = weighted_auc_score(y_true_perfect, y_score_perfect)
    print(f"Perfect Score (Expected ~1.0): {score_perfect:.4f}")
    assert score_perfect == 1.0, "Perfect predictions should yield AUC of 1.0"

    # Case B: Inverse prediction (Worst case)
    y_true_bad = [0, 0, 1, 1]
    y_score_bad = [0.9, 0.8, 0.1, 0.2]
    score_bad = weighted_auc_score(y_true_bad, y_score_bad)
    print(f"Inverse Score (Expected ~0.0): {score_bad:.4f}")
    assert score_bad == 0.0, "Inverse predictions should yield AUC of 0.0"

    print("Metric calculation verified successfully.")

    # 6. Inference (TTA) Demonstration
    print("\n[5/5] Demonstrating Test Time Augmentation (TTA)...")

    # Use the validation loader for TTA prediction demonstration
    # predict_tta expects a dataloader and returns a list of probabilities
    tta_predictions = predict_tta(model, val_loader, device)

    print(f"Number of TTA predictions: {len(tta_predictions)}")

    # Logic Verification
    assert len(tta_predictions) == len(
        val_subset_df
    ), f"Expected {len(val_subset_df)} predictions, got {len(tta_predictions)}"

    # Check value range
    min_pred, max_pred = min(tta_predictions), max(tta_predictions)
    print(f"Prediction Range: [{min_pred:.4f}, {max_pred:.4f}]")
    assert (
        0.0 <= min_pred and max_pred <= 1.0
    ), "Predictions must be probabilities in [0, 1]"

    print("TTA Inference verified successfully.")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
