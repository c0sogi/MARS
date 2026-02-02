import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from the provided library
from library import config
from library.config import seed_everything, DEVICE, WORKING_DIR
from library.data_processing import process_dataset_roi
from library.dataset import RNWIVDataset, get_transforms
from library.model import RNWIVEfficientNet
from library.train_eval import train_one_epoch, validate


def run_demo():
    print("Initializing demonstration...")

    # 1. Setup and Reproducibility
    seed_everything(42)
    print(f"Device: {DEVICE}")

    # 2. Load Metadata
    # We use the generated metadata files.
    train_meta_path = "./metadata/train_metadata.csv"
    if not os.path.exists(train_meta_path):
        raise FileNotFoundError(f"Metadata file not found at {train_meta_path}")

    df_train = pd.read_csv(train_meta_path)
    print(f"Total training samples available: {len(df_train)}")

    # Optimization: Use a small subset for demonstration speed
    subset_size = 8
    df_subset = df_train.head(subset_size).copy()
    print(f"Using subset of {subset_size} samples for demonstration.")

    # 3. Data Processing: ROI Boundary Calculation
    # The RN-WIV strategy requires knowing the start/end slices of the brain volume.
    print("\n--- Step 1: Processing ROI Boundaries ---")
    # We force calculation (load_cached_data=False) to demonstrate the logic,
    # though in production True is preferred.
    roi_df = process_dataset_roi(df_subset, load_cached_data=False)

    # Verification: Check ROI DataFrame structure
    expected_cols = [
        "BraTS21ID",
        "flair_start",
        "flair_end",
        "flair_count",
        "t1w_start",
        "t1w_end",
        "t1w_count",  # t1w is computed even if not used in dataset
        "t1wce_start",
        "t1wce_end",
        "t1wce_count",
        "t2w_start",
        "t2w_end",
        "t2w_count",
    ]
    for col in expected_cols:
        if col not in roi_df.columns:
            raise AssertionError(f"ROI DataFrame missing column: {col}")

    print("ROI boundaries computed successfully.")
    print(roi_df.head(2))

    # 4. Dataset Instantiation
    print("\n--- Step 2: Creating RN-WIV Dataset ---")
    # We use the 'train' transforms which include spatial augmentations
    train_dataset = RNWIVDataset(
        df=df_subset, roi_df=roi_df, transform=get_transforms("train"), is_train=True
    )

    # Verification: Check item shape and type
    # Expected shape: (9, 224, 224) -> 3 modalities * 3 depths
    sample_img, sample_target = train_dataset[0]
    print(f"Sample Image Shape: {sample_img.shape}")
    print(f"Sample Target: {sample_target}")

    if sample_img.shape != (9, 224, 224):
        raise AssertionError(
            f"Expected image shape (9, 224, 224), got {sample_img.shape}"
        )
    if not isinstance(sample_target, torch.Tensor):
        raise AssertionError("Target should be a torch.Tensor")

    # Create DataLoader
    batch_size = 4
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    # 5. Model Initialization
    print("\n--- Step 3: Initializing RN-WIV EfficientNet ---")
    model = RNWIVEfficientNet(pretrained=True).to(DEVICE)

    # Verification: Check first layer modification
    # The config specifies NUM_CHANNELS = 9. The conv_stem should accept 9 channels.
    first_layer = model.backbone.conv_stem
    print(f"First layer in_channels: {first_layer.in_channels}")
    if first_layer.in_channels != 9:
        raise AssertionError(
            f"Model first layer should have 9 input channels, found {first_layer.in_channels}"
        )

    # Verification: Forward pass with dummy batch
    dummy_input = torch.randn(2, 9, 224, 224).to(DEVICE)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Forward pass output shape: {output.shape}")
    if output.shape != (2, 1):
        raise AssertionError(f"Expected output shape (2, 1), got {output.shape}")

    # 6. Training Loop Demonstration
    print("\n--- Step 4: Running Training Step ---")
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    # Run one epoch (on the subset)
    train_loss, train_auc = train_one_epoch(
        model, train_loader, criterion, optimizer, scaler, DEVICE
    )
    print(f"Training Step - Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")

    if np.isnan(train_loss):
        raise AssertionError("Training loss returned NaN.")

    # 7. Validation Loop Demonstration
    print("\n--- Step 5: Running Validation Step ---")
    # Create a validation dataset/loader (using 'val' transforms)
    val_dataset = RNWIVDataset(
        df=df_subset, roi_df=roi_df, transform=get_transforms("val"), is_train=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)
    print(f"Validation Step - Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")

    # 8. Inference / Prediction
    print("\n--- Step 6: Inference Check ---")
    model.eval()
    predictions = []
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(DEVICE)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions.extend(probs)

    predictions = np.array(predictions).flatten()
    print(f"Generated {len(predictions)} predictions.")
    print(f"Predictions: {predictions}")

    if len(predictions) != subset_size:
        raise AssertionError(
            f"Expected {subset_size} predictions, got {len(predictions)}"
        )

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    run_demo()
