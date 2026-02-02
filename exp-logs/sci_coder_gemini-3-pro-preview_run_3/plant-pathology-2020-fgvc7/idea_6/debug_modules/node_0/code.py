import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, AverageMeter, get_class_weights
from library.dataset import (
    get_transforms,
    prepare_folds,
    AppleDataset,
    get_loaders,
    get_test_loader,
)
from library.models import HierarchicalEfficientNet, HierarchicalSwin
from library.training import train_fn, valid_fn, EarlyStopping
from library.inference import predict_with_tta


def run_demo():
    print("=" * 50)
    print("Starting Apple Disease Detection Library Demo")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Speed and Demo
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo...")
    # Override Config to run fast on a small subset
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Very small subset for speed
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script

    # Create a specific working directory for this demo
    demo_work_dir = "./working/demo_test"
    Config.WORK_DIR = demo_work_dir
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    print(f"Working Directory: {Config.WORK_DIR}")

    # -------------------------------------------------------------------------
    # 2. Verify Utilities
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utilities (utils.py)...")

    # Test Seeding
    seed_everything(Config.SEED)
    print("Seed set successfully.")

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(val=10, n=2)
    meter.update(val=20, n=2)
    assert meter.avg == 15.0, f"AverageMeter failed. Expected 15.0, got {meter.avg}"
    print("AverageMeter logic verified.")

    # Test Class Weights
    # We force computation to check logic, though cache might exist
    weights = get_class_weights(load_cached_data=False)
    print(f"Class Weights: {weights}")
    assert isinstance(weights, torch.Tensor), "Class weights should be a Tensor"
    assert (
        weights.shape[0] == Config.NUM_CLASSES
    ), f"Expected {Config.NUM_CLASSES} weights"
    assert (weights > 0).all(), "Class weights must be positive"
    print("Class weights calculation verified.")

    # -------------------------------------------------------------------------
    # 3. Verify Dataset and Transforms
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Dataset Pipeline (dataset.py)...")

    # Test prepare_folds
    # This reads metadata/train.csv and metadata/val.csv
    df_folds = prepare_folds(load_cached_data=False)
    assert "fold" in df_folds.columns, "df_folds missing 'fold' column"
    assert len(df_folds) > 0, "df_folds is empty"
    print(f"Folds prepared. Total samples: {len(df_folds)}")

    # Test Transforms
    transforms = get_transforms(image_size=224, phase="train")
    assert transforms is not None, "Transforms should not be None"

    # Test Dataset Instantiation
    # We use the debug flag implicitly via Config.DEBUG set earlier
    train_ds = AppleDataset(
        df_folds, transforms=transforms, output_label=True, debug=True
    )
    print(f"Debug Dataset size: {len(train_ds)}")

    # Test __getitem__
    img, label = train_ds[0]
    print(f"Sample Image Shape: {img.shape}")
    print(f"Sample Label: {label}")

    assert img.shape == (
        3,
        224,
        224,
    ), f"Image shape mismatch. Expected (3, 224, 224), got {img.shape}"
    assert label.shape == (
        4,
    ), f"Label shape mismatch. Expected (4,), got {label.shape}"
    assert isinstance(img, torch.Tensor), "Image should be a torch.Tensor"

    # Test DataLoaders
    train_loader, val_loader = get_loaders(
        fold=0, image_size=224, batch_size=Config.BATCH_SIZE, num_workers=0
    )

    # Fetch one batch
    batch_imgs, batch_labels = next(iter(train_loader))
    print(f"Batch Image Shape: {batch_imgs.shape}")
    print(f"Batch Label Shape: {batch_labels.shape}")

    assert batch_imgs.shape[0] == Config.BATCH_SIZE, "Batch size mismatch"
    assert batch_imgs.shape[1] == 3, "Channel mismatch"

    # -------------------------------------------------------------------------
    # 4. Verify Models
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Models (models.py)...")

    device = Config.DEVICE

    # Test HierarchicalEfficientNet
    # Using pretrained=False to avoid downloading weights during demo
    print("Initializing HierarchicalEfficientNet (pretrained=False)...")
    model_eff = HierarchicalEfficientNet(pretrained=False).to(device)

    dummy_input = torch.randn(2, 3, Config.IMG_SIZE_EFFNET, Config.IMG_SIZE_EFFNET).to(
        device
    )
    with torch.no_grad():
        out_eff = model_eff(dummy_input)

    print(f"EffNet Output Shape: {out_eff.shape}")
    assert out_eff.shape == (2, Config.NUM_CLASSES), "EffNet output shape mismatch"

    # Test HierarchicalSwin
    print("Initializing HierarchicalSwin (pretrained=False)...")
    model_swin = HierarchicalSwin(pretrained=False).to(device)

    dummy_input_swin = torch.randn(2, 3, Config.IMG_SIZE_SWIN, Config.IMG_SIZE_SWIN).to(
        device
    )
    with torch.no_grad():
        out_swin = model_swin(dummy_input_swin)

    print(f"Swin Output Shape: {out_swin.shape}")
    assert out_swin.shape == (2, Config.NUM_CLASSES), "Swin output shape mismatch"

    # -------------------------------------------------------------------------
    # 5. Verify Training Logic
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Training Logic (training.py)...")

    # We will use the EffNet model and the loaders created earlier
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.AdamW(model_eff.parameters(), lr=1e-4)

    print("Running 1 epoch of training (simulated)...")
    train_loss = train_fn(
        train_loader, model_eff, criterion, optimizer, device, epoch=0
    )
    print(f"Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss returned NaN"

    print("Running validation...")
    val_loss, val_auc = valid_fn(val_loader, model_eff, criterion, device)
    print(f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
    assert not np.isnan(val_loss), "Validation loss returned NaN"

    # Test EarlyStopping Logic
    es = EarlyStopping(patience=2, path=os.path.join(Config.WORK_DIR, "checkpoint.pth"))
    es(val_loss, model_eff)
    assert os.path.exists(
        os.path.join(Config.WORK_DIR, "checkpoint.pth")
    ), "Checkpoint not saved"
    print("EarlyStopping checkpoint logic verified.")

    # -------------------------------------------------------------------------
    # 6. Verify Inference Logic
    # -------------------------------------------------------------------------
    print("\n[6] Verifying Inference Logic (inference.py)...")

    # Get Test Loader
    # Note: Test set is small (183 images), so we can run on it.
    # We use a small batch size.
    test_loader = get_test_loader(
        image_size=Config.IMG_SIZE_EFFNET, batch_size=4, num_workers=0
    )

    print("Running Predict with TTA...")
    # We use the trained model_eff from the previous step
    preds, ids = predict_with_tta(model_eff, test_loader, device)

    print(f"Predictions Shape: {preds.shape}")
    print(f"Number of IDs: {len(ids)}")

    assert len(preds) == len(ids), "Mismatch between predictions and IDs"
    assert preds.shape[1] == Config.NUM_CLASSES, "Prediction classes mismatch"

    # Check if predictions are probabilities (sum to ~1, though TTA averaging might slightly drift due to float precision,
    # but softmax ensures it per prediction. Average of softmax sums to 1.)
    row_sums = preds.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), "Predictions do not sum to 1.0"

    print("Inference logic verified.")

    print("\n" + "=" * 50)
    print("All demonstrations completed successfully.")
    print("=" * 50)


if __name__ == "__main__":
    run_demo()
