import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library modules
from library.config import Config
from library.utils import seed_everything, compute_robust_auc
from library.data import load_data, SpectrogramDataset, HistogramDataset, get_transforms
from library.models import BirdCNN, BirdMLP
from library.engine import train_one_epoch, validate, fit_model


def run_demo():
    print("Initializing Demo...")

    # 1. Configure for Speed and Demo constraints
    # Override Config parameters to run quickly on a small subset
    Config.EPOCHS = 2
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 32  # Small subset for speed
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.PRETRAINED = False  # Disable downloading weights for speed/offline safety
    Config.CNN_MODELS = ["resnet18"]  # Test only one backbone

    # Ensure reproducibility
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Verify Utility Functions
    print("\n--- Verifying Utils ---")
    # Test Robust AUC with synthetic data
    # Case: Perfect prediction
    y_true = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    y_pred = np.array([[0.9, 0.1, 0.9], [0.1, 0.9, 0.1], [0.8, 0.8, 0.2]])
    auc = compute_robust_auc(y_true, y_pred)
    print(f"Computed AUC (synthetic): {auc:.4f}")
    assert 0.0 <= auc <= 1.0, "AUC should be between 0 and 1"

    # Case: Missing class in batch (all 0s for class index 2)
    y_true_missing = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]])
    # compute_robust_auc handles this by skipping the column or returning 0.5 if all skipped
    auc_missing = compute_robust_auc(y_true_missing, y_pred)
    print(f"Computed AUC (missing class): {auc_missing:.4f}")
    assert isinstance(auc_missing, float)

    # 3. Verify Data Loading and Processing
    print("\n--- Verifying Data Loading ---")
    # Load metadata and features
    train_df, test_df, feature_map = load_data(load_cached_data=False)

    print(f"Train DF shape: {train_df.shape}")
    print(f"Test DF shape: {test_df.shape}")
    print(f"Feature Map size: {len(feature_map)}")

    assert not train_df.empty, "Training dataframe is empty"
    assert "fold" in train_df.columns, "Folds not created in train_df"

    # Instantiate Datasets
    train_dataset = SpectrogramDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )
    hist_dataset = HistogramDataset(train_df, feature_map, mode="train")

    # Verify Spectrogram Dataset Item
    img, lbl, idx = train_dataset[0]
    print(f"Spectrogram Image Shape: {img.shape}")
    print(f"Label Shape: {lbl.shape}")

    assert img.shape == (3, 224, 224), f"Unexpected image shape: {img.shape}"
    assert lbl.shape == (Config.NUM_CLASSES,), f"Unexpected label shape: {lbl.shape}"
    assert isinstance(img, torch.Tensor), "Image is not a tensor"

    # Verify Histogram Dataset Item
    feat, lbl_h, idx_h = hist_dataset[0]
    print(f"Histogram Feature Shape: {feat.shape}")

    assert feat.shape == (
        Config.MLP_INPUT_DIM,
    ), f"Unexpected feature shape: {feat.shape}"
    assert torch.equal(lbl, lbl_h), "Labels mismatch between datasets for same index"

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )
    # Use same dataset for val just for demo purposes
    val_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 4. Verify Models
    print("\n--- Verifying Models ---")

    # Test BirdCNN
    cnn_model = BirdCNN(backbone_name="resnet18", pretrained=False).to(device)
    dummy_img_batch = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        cnn_out = cnn_model(dummy_img_batch)

    print(f"CNN Output Shape: {cnn_out.shape}")
    assert cnn_out.shape == (2, Config.NUM_CLASSES), "CNN output shape mismatch"

    # Test BirdMLP
    mlp_model = BirdMLP().to(device)
    dummy_feat_batch = torch.randn(2, Config.MLP_INPUT_DIM).to(device)
    with torch.no_grad():
        mlp_out = mlp_model(dummy_feat_batch)

    print(f"MLP Output Shape: {mlp_out.shape}")
    assert mlp_out.shape == (2, Config.NUM_CLASSES), "MLP output shape mismatch"

    # 5. Verify Training Engine
    print("\n--- Verifying Training Engine ---")

    optimizer = torch.optim.AdamW(cnn_model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )
    criterion = nn.BCEWithLogitsLoss()

    # Run one epoch of training manually
    print("Running single epoch training step...")
    loss = train_one_epoch(cnn_model, train_loader, optimizer, device, epoch=0)
    print(f"Epoch 0 Loss: {loss:.4f}")
    assert not np.isnan(loss), "Training loss returned NaN"

    # Run validation manually
    print("Running validation step...")
    val_loss, val_auc = validate(cnn_model, val_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
    assert not np.isnan(val_loss), "Validation loss returned NaN"

    # Run full fit_model integration
    print("Running fit_model integration (2 epochs)...")
    # Using fold 0
    best_checkpoints = fit_model(
        model=cnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        fold=0,
        model_name="demo_resnet18",
    )

    print(f"Training complete. Saved {len(best_checkpoints)} checkpoints.")
    assert len(best_checkpoints) > 0, "No checkpoints were saved"

    # Verify checkpoint file existence
    best_score, best_path = best_checkpoints[0]
    assert os.path.exists(best_path), f"Checkpoint file missing at {best_path}"
    print(f"Best checkpoint found at: {best_path} with score {best_score:.4f}")

    print("\nDemo completed successfully!")


if __name__ == "__main__":
    run_demo()
