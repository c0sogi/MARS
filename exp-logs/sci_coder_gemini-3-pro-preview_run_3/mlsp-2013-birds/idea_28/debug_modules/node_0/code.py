import os
import sys
import shutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, RobustMetric, SnapshotManager
from library.data_factory import (
    load_histogram_features,
    create_folds,
    get_loaders,
    BirdDataset,
)
from library.model_factory import get_cnn_model, SymbolicMLP
from library.engine import train_one_epoch, validate_one_epoch, mixup_data


def run_demo():
    print("=" * 50)
    print("STARTING LIBRARY DEMONSTRATION")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Speed
    # -------------------------------------------------------------------------
    print("\n[Step 1] Configuring environment for fast demonstration...")

    # Override Config for speed
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Small subset
    Config.BATCH_SIZE = 4
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.SNAPSHOTS_K = 2  # Keep top 2 checkpoints

    # Ensure working directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")
    print("Configuration updated for debug mode.")

    # -------------------------------------------------------------------------
    # 2. Data Factory Verification
    # -------------------------------------------------------------------------
    print("\n[Step 2] Verifying Data Factory...")

    # Load Features
    print("Loading histogram features...")
    hist_df = load_histogram_features(
        load_cached_data=False
    )  # Force reload to test parsing
    assert isinstance(
        hist_df, pd.DataFrame
    ), "load_histogram_features should return a DataFrame"
    if not hist_df.empty:
        assert "rec_id" in hist_df.columns
        assert "features" in hist_df.columns
        print(f"Features loaded: {len(hist_df)} records.")
    else:
        print("Warning: Histogram features file might be missing or empty.")

    # Create Folds
    print("Creating folds...")
    folds_df = create_folds(load_cached_data=False)  # Force reload
    assert "fold" in folds_df.columns, "folds_df must have a 'fold' column"
    assert len(folds_df) > 0, "folds_df should not be empty"
    print(f"Folds created. Total samples: {len(folds_df)}")

    # Get Loaders
    print("Initializing DataLoaders (Fold 0)...")
    train_loader, val_loader = get_loaders(
        fold_idx=0,
        folds_df=folds_df,
        feature_df=hist_df,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Verify Batch Structure
    print("Fetching one batch from Train Loader...")
    batch = next(iter(train_loader))

    images = batch["image"]
    features = batch["features"]
    targets = batch["target"]
    rec_ids = batch["rec_id"]

    print(f"Image Shape: {images.shape}")  # Expect [B, 3, 224, 224]
    print(f"Features Shape: {features.shape}")  # Expect [B, 100]
    print(f"Targets Shape: {targets.shape}")  # Expect [B, 19]

    assert images.shape == (Config.BATCH_SIZE, 3, 224, 224), "Incorrect Image Shape"
    assert features.shape == (
        Config.BATCH_SIZE,
        Config.MLP_INPUT_DIM,
    ), "Incorrect Feature Shape"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Incorrect Target Shape"

    print("Data Pipeline Verified.")

    # -------------------------------------------------------------------------
    # 3. Model Factory Verification
    # -------------------------------------------------------------------------
    print("\n[Step 3] Verifying Model Factory...")

    # CNN Model
    print("Initializing CNN (ResNet18)...")
    cnn_model = get_cnn_model("resnet18", pretrained=False)  # False for speed
    cnn_model.to(device)

    # MLP Model
    print("Initializing Symbolic MLP...")
    mlp_model = SymbolicMLP()
    mlp_model.to(device)

    # Dummy Forward Pass
    with torch.no_grad():
        cnn_out = cnn_model(images.to(device))
        mlp_out = mlp_model(features.to(device))

    assert cnn_out.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "CNN output shape mismatch"
    assert mlp_out.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "MLP output shape mismatch"

    print("Models Verified.")

    # -------------------------------------------------------------------------
    # 4. Engine & Training Loop Verification
    # -------------------------------------------------------------------------
    print("\n[Step 4] Verifying Engine (Training/Validation)...")

    criterion = nn.BCEWithLogitsLoss()

    # Test CNN Training Step
    print("Running CNN Training Step...")
    optimizer_cnn = torch.optim.Adam(cnn_model.parameters(), lr=1e-3)
    cnn_loss = train_one_epoch(
        cnn_model, train_loader, optimizer_cnn, criterion, device, input_key="image"
    )
    print(f"CNN Train Loss: {cnn_loss:.4f}")
    assert np.isfinite(cnn_loss), "CNN Training loss is not finite"

    # Test MLP Training Step
    print("Running MLP Training Step...")
    optimizer_mlp = torch.optim.Adam(mlp_model.parameters(), lr=1e-3)
    mlp_loss = train_one_epoch(
        mlp_model, train_loader, optimizer_mlp, criterion, device, input_key="features"
    )
    print(f"MLP Train Loss: {mlp_loss:.4f}")
    assert np.isfinite(mlp_loss), "MLP Training loss is not finite"

    # Test Validation Step (using MLP for speed)
    print("Running MLP Validation Step...")
    val_loss, val_auc = validate_one_epoch(
        mlp_model, val_loader, criterion, device, input_key="features"
    )
    print(f"MLP Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
    assert np.isfinite(val_loss), "Validation loss is not finite"
    assert 0.0 <= val_auc <= 1.0, "Validation AUC out of range"

    print("Engine Verified.")

    # -------------------------------------------------------------------------
    # 5. Utils Verification
    # -------------------------------------------------------------------------
    print("\n[Step 5] Verifying Utilities...")

    # Test Mixup
    print("Testing Mixup Data...")
    x_dummy = torch.randn(4, 3, 224, 224).to(device)
    y_dummy = torch.randn(4, 19).to(device)
    mixed_x, mixed_y, lam = mixup_data(x_dummy, y_dummy, alpha=0.4, device=device)

    assert mixed_x.shape == x_dummy.shape, "Mixup X shape mismatch"
    assert mixed_y.shape == y_dummy.shape, "Mixup Y shape mismatch"
    assert 0.0 <= lam <= 1.0, "Mixup lambda out of range"
    print("Mixup Verified.")

    # Test RobustMetric directly
    print("Testing RobustMetric...")
    metric = RobustMetric()
    # Simulate 2 batches
    # Batch 1: Class 0 has 0 and 1 targets
    out1 = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    tgt1 = torch.tensor([[0, 1], [1, 0]])
    metric.update(out1, tgt1)

    # Batch 2
    out2 = torch.tensor([[0.4, 0.6]])
    tgt2 = torch.tensor([[0, 1]])
    metric.update(out2, tgt2)

    score = metric.compute()
    print(f"RobustMetric Computed Score: {score:.4f}")
    assert 0.0 <= score <= 1.0
    metric.reset()
    assert len(metric.predictions) == 0
    print("RobustMetric Verified.")

    # Test SnapshotManager
    print("Testing SnapshotManager...")
    demo_ckpt_dir = os.path.join(Config.WORKING_DIR, "demo_checkpoints")
    if os.path.exists(demo_ckpt_dir):
        shutil.rmtree(demo_ckpt_dir)

    manager = SnapshotManager(checkpoint_dir=demo_ckpt_dir, k=2, maximize=True)

    # Save 3 models, expect only top 2 to remain
    # Model A: Score 0.5
    manager.save(mlp_model, score=0.5, epoch=1, fold_idx=0, model_name="test_mlp")
    # Model B: Score 0.8
    manager.save(mlp_model, score=0.8, epoch=2, fold_idx=0, model_name="test_mlp")
    # Model C: Score 0.6
    manager.save(mlp_model, score=0.6, epoch=3, fold_idx=0, model_name="test_mlp")

    saved_files = os.listdir(demo_ckpt_dir)
    print(f"Saved Checkpoints: {saved_files}")

    # We expect 0.8 and 0.6 to be kept. 0.5 should be deleted.
    assert len(saved_files) == 2, f"Expected 2 files, found {len(saved_files)}"
    scores_in_filenames = [
        float(f.split("auc")[1].replace(".pth", "")) for f in saved_files
    ]
    assert (
        0.8 in scores_in_filenames and 0.6 in scores_in_filenames
    ), "SnapshotManager failed to keep top K"
    assert (
        0.5 not in scores_in_filenames
    ), "SnapshotManager failed to delete worst model"

    print("SnapshotManager Verified.")

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED SUCCESSFULLY")
    print("=" * 50)


if __name__ == "__main__":
    run_demo()
