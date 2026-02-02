import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure local library imports work
sys.path.append(".")

# Import provided library components
from library.config import Config
from library.utils import seed_everything, calculate_auc, MetricMonitor
from library.dataset import PathologyDataset, get_transforms, mixup_data
from library.model import TumorClassifier, GeM
from library.engine import train_loop


def main():
    print("--- Starting Tumor Detection Pipeline Demo ---")

    # 1. Configuration & Reproducibility
    # Override Config for rapid demonstration
    Config.epochs = 2
    Config.batch_size = 4
    Config.num_workers = 0  # Disable multi-processing for simple script
    Config.debug = True

    # Create necessary directories
    Config.setup()

    # Set seed
    seed_everything(Config.seed)
    device = Config.device
    print(f"Device: {device}")
    print("Configuration optimized for speed.")

    # 2. Dataset Verification
    print("\n--- Verifying Dataset ---")

    # Load metadata
    if not os.path.exists(Config.train_metadata_path):
        raise FileNotFoundError(f"Metadata not found at {Config.train_metadata_path}")

    df_train = pd.read_csv(Config.train_metadata_path)

    # Take a small subset (20 samples) for the demo
    df_subset = df_train.iloc[:20].copy()

    # Initialize Dataset
    transforms = get_transforms(data="train")
    dataset = PathologyDataset(df_subset, transforms=transforms)

    # Verify single item retrieval
    img, label = dataset[0]
    print(f"Image Shape: {img.shape}")
    print(f"Label: {label}")

    # Assertions for Dataset
    assert img.shape == (
        3,
        Config.crop_size,
        Config.crop_size,
    ), f"Expected image shape (3, {Config.crop_size}, {Config.crop_size}), got {img.shape}"
    assert isinstance(label, torch.Tensor), "Label must be a torch Tensor"
    assert label.shape == (
        1,
    ), f"Label shape mismatch. Expected (1,), got {label.shape}"

    # Initialize DataLoader
    loader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)

    # Verify Mixup Augmentation
    print("Verifying Mixup...")
    batch_imgs, batch_lbls = next(iter(loader))
    batch_imgs, batch_lbls = batch_imgs.to(device), batch_lbls.to(device)

    mixed_x, y_a, y_b, lam = mixup_data(
        batch_imgs, batch_lbls, alpha=Config.mixup_alpha
    )

    assert mixed_x.shape == batch_imgs.shape, "Mixup output shape mismatch"
    assert y_a.shape == batch_lbls.shape, "Mixup label A shape mismatch"
    assert 0.0 <= lam <= 1.0, "Mixup lambda out of bounds"
    print("Dataset and Mixup verified.")

    # 3. Model Verification
    print("\n--- Verifying Model ---")

    # Instantiate Model
    # Using pretrained=False to ensure no download overhead for this demo
    model = TumorClassifier(pretrained=False)
    model.to(device)

    # Verify GeM Pooling existence
    if Config.use_gem_pooling:
        gem_layers = [m for m in model.modules() if isinstance(m, GeM)]
        assert (
            len(gem_layers) > 0
        ), "GeM pooling enabled in Config but not found in model"
        print("GeM Pooling layer confirmed.")

    # Verify Forward Pass
    dummy_input = torch.randn(
        Config.batch_size, 3, Config.crop_size, Config.crop_size
    ).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Output Shape: {output.shape}")
    assert output.shape == (
        Config.batch_size,
        1,
    ), f"Model output shape mismatch. Expected ({Config.batch_size}, 1), got {output.shape}"
    print("Model architecture verified.")

    # 4. Engine & Training Loop Verification
    print("\n--- Verifying Training Loop ---")

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs
    )

    # Run training loop on the subset
    # We use the subset for both train and val to ensure it runs without error
    best_auc = train_loop(
        model=model,
        train_loader=loader,
        val_loader=loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        fold=0,
        epochs=Config.epochs,
        patience=2,
    )

    print(f"Training loop completed. Best AUC: {best_auc}")

    # Verify Checkpoint Creation
    checkpoint_path = os.path.join(Config.checkpoint_dir, "best_model_fold_0.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print("Checkpoint creation verified.")

    # 5. Metric Utilities Verification
    print("\n--- Verifying Metrics ---")

    # Test AUC Calculation
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.6, 0.9])
    auc = calculate_auc(y_true, y_pred)
    assert 0.0 <= auc <= 1.0, "AUC calculation out of bounds"
    print(f"AUC Test: {auc}")

    # Test MetricMonitor
    monitor = MetricMonitor()
    monitor.update("loss", 2.0, n=1)
    monitor.update("loss", 4.0, n=1)
    assert monitor.metrics["loss"]["avg"] == 3.0, "MetricMonitor average failed"
    print("Metric utilities verified.")

    print("\n--- All Verifications Passed Successfully ---")


if __name__ == "__main__":
    main()
