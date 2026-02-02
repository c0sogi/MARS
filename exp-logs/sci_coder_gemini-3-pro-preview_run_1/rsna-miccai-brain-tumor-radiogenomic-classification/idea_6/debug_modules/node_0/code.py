import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, compute_auc
from library.image_processing import ImageProcessor
from library.dataset import get_dataloader
from library.model import MultiPlanarSiameseNet
from library.trainer import train_one_epoch, validate

if __name__ == "__main__":
    print(
        "Starting demonstration of the Multi-Planar 2.5D Holographic Network pipeline..."
    )

    # ==========================================
    # 1. Setup & Configuration Overrides
    # ==========================================
    # Override Config defaults for a fast demonstration
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 4  # Process only 4 subjects
    Config.BATCH_SIZE = 2
    Config.MAX_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Use a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = Config.WORKING_DIR
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Ensure reproducibility
    seed_everything(Config.SEED)
    print(f"Configuration set: Debug={Config.DEBUG}, Batch Size={Config.BATCH_SIZE}")

    # ==========================================
    # 2. Verify Utility Functions
    # ==========================================
    print("\n[Testing] Utility Functions...")
    # Test AUC computation with known values
    y_true_test = np.array([0, 0, 1, 1])
    y_pred_perfect = np.array([0.1, 0.2, 0.8, 0.9])
    y_pred_worst = np.array([0.9, 0.8, 0.2, 0.1])

    auc_perfect = compute_auc(y_true_test, y_pred_perfect)
    auc_worst = compute_auc(y_true_test, y_pred_worst)

    assert auc_perfect == 1.0, f"Expected AUC 1.0, got {auc_perfect}"
    assert auc_worst == 0.0, f"Expected AUC 0.0, got {auc_worst}"
    print("✓ compute_auc logic verified.")

    # ==========================================
    # 3. Verify Image Processing
    # ==========================================
    print("\n[Testing] Image Processing...")
    # Load metadata
    if not os.path.exists(Config.TRAIN_METADATA_PATH):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA_PATH}")

    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    # Select a small subset for processing
    df_subset = df_train.head(Config.DEBUG_SAMPLE_SIZE).copy()

    processor = ImageProcessor(debug=Config.DEBUG)

    # Process dataset (this will load DICOMs, crop ROI, resize, and cache)
    data_dict = processor.process_dataset(
        df_subset, split_name="demo_train", load_cached_data=False
    )

    # Verify keys
    expected_keys = {"ids", "axial", "coronal", "sagittal", "targets"}
    assert expected_keys.issubset(
        data_dict.keys()
    ), f"Missing keys in data_dict. Found: {data_dict.keys()}"

    # Verify shapes
    # Expected: (N, H, W, C) where C=3 (FLAIR, T1wCE, T2w)
    n_samples = len(df_subset)
    img_size = Config.IMG_SIZE
    n_channels = len(Config.MODALITIES)

    assert data_dict["axial"].shape == (
        n_samples,
        img_size,
        img_size,
        n_channels,
    ), f"Incorrect Axial shape: {data_dict['axial'].shape}"
    assert data_dict["targets"].shape == (
        n_samples,
    ), f"Incorrect Targets shape: {data_dict['targets'].shape}"

    # Verify Normalization (0 to 1)
    assert (
        data_dict["axial"].min() >= 0.0 and data_dict["axial"].max() <= 1.0
    ), "Image data is not properly normalized to [0, 1]"

    print(f"✓ Processed {n_samples} subjects. Image shape: {data_dict['axial'].shape}")

    # ==========================================
    # 4. Verify Dataset & DataLoader
    # ==========================================
    print("\n[Testing] Dataset & DataLoader...")
    # Use the helper function to get the loader
    # Note: We pass load_cached_data=True because we just generated the cache in step 3
    train_loader = get_dataloader(
        df_subset,
        split_name="demo_train",
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        load_cached_data=True,
    )

    # Fetch one batch
    batch = next(iter(train_loader))

    # Verify Batch Structure
    assert "axial" in batch and "label" in batch and "BraTS21ID" in batch

    # Verify Tensor Shapes: (Batch, Channels, Height, Width)
    # Albumentations converts HWC to CHW
    expected_tensor_shape = (Config.BATCH_SIZE, n_channels, img_size, img_size)
    assert (
        batch["axial"].shape == expected_tensor_shape
    ), f"DataLoader tensor shape mismatch. Expected {expected_tensor_shape}, got {batch['axial'].shape}"

    assert batch["label"].shape == (
        Config.BATCH_SIZE,
    ), f"Label shape mismatch. Expected ({Config.BATCH_SIZE},), got {batch['label'].shape}"

    print(f"✓ DataLoader produced batch with shape: {batch['axial'].shape}")

    # ==========================================
    # 5. Verify Model Architecture
    # ==========================================
    print("\n[Testing] Model Architecture...")
    # Initialize model (pretrained=False for speed/offline safety)
    model = MultiPlanarSiameseNet(backbone_name="efficientnet_b0", pretrained=False)
    device = Config.DEVICE
    model.to(device)

    # Prepare inputs
    axial_in = batch["axial"].to(device)
    coronal_in = batch["coronal"].to(device)
    sagittal_in = batch["sagittal"].to(device)

    # Forward pass
    logits = model(axial_in, coronal_in, sagittal_in)

    # Verify Output
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected ({Config.BATCH_SIZE}, 1), got {logits.shape}"

    print("✓ Model forward pass successful. Logits shape correct.")

    # ==========================================
    # 6. Verify Training & Validation Steps
    # ==========================================
    print("\n[Testing] Training & Validation Steps...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Train one epoch (using the small loader)
    avg_train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"✓ Train Step Complete. Loss: {avg_train_loss:.4f}")

    # Validate (using the same loader for demo purposes)
    avg_val_loss, val_auc = validate(model, train_loader, criterion, device)
    print(f"✓ Validation Step Complete. Loss: {avg_val_loss:.4f}, AUC: {val_auc:.4f}")

    # Check that metrics are valid numbers
    assert not np.isnan(avg_train_loss), "Training loss is NaN"
    assert 0.0 <= val_auc <= 1.0, "Validation AUC is out of range [0, 1]"

    print("\n==========================================")
    print(" SUCCESS: All pipeline components verified.")
    print("==========================================")
