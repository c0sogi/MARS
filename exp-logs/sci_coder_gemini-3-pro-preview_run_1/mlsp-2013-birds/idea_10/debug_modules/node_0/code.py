import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.utils import set_seed, calculate_roc_auc, sanitize_pseudo_labels
from library.dataset import BirdDataset, get_dataloader, MixupCollate
from library.model import BirdResNet
from library.trainer import Trainer


def run_demo():
    print("--- Starting Library Demo ---")

    # 1. Setup and Configuration Override for Speed
    # We modify the Config class attributes directly to ensure the demo runs quickly
    # and uses minimal resources.
    print("[1] Configuring environment for demo...")
    Config.IMG_HEIGHT = 64  # Reduced height
    Config.IMG_WIDTH = 128  # Reduced width
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Main process only for stability in demo
    Config.DEBUG = True  # Use debug subset
    Config.DEBUG_SUBSET_SIZE = 10  # Only use 10 samples
    Config.MIXUP_ALPHA = 1.0  # Force mixup capability check

    # Set device
    device = Config.DEVICE
    print(f"    Device: {device}")

    # Set seed for reproducibility
    set_seed(42)

    # 2. Data Loading Demo
    print("\n[2] Testing Data Loading Pipeline...")

    # Test get_dataloader factory
    # We use 'train' split which applies augmentations and Mixup
    train_loader = get_dataloader(
        split="train", batch_size=Config.BATCH_SIZE, shuffle=True, debug=True
    )

    # Fetch a single batch
    images, labels, ids = next(iter(train_loader))

    # Verify shapes
    print(f"    Batch Images Shape: {images.shape}")
    print(f"    Batch Labels Shape: {labels.shape}")
    print(f"    Batch IDs Shape: {ids.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMG_HEIGHT,
        Config.IMG_WIDTH,
    ), "Image tensor shape mismatch"
    assert labels.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Label tensor shape mismatch"
    assert ids.shape == (Config.BATCH_SIZE,), "ID tensor shape mismatch"

    # Check MixupCollate directly
    # Create dummy data for mixup test
    dummy_batch = [
        (torch.randn(3, 64, 128), torch.zeros(19), torch.tensor(1)),
        (torch.randn(3, 64, 128), torch.ones(19), torch.tensor(2)),
    ]
    mixup_fn = MixupCollate(alpha=1.0)
    m_imgs, m_lbls, m_ids = mixup_fn(dummy_batch)

    assert m_imgs.shape == (2, 3, 64, 128), "Mixup output image shape incorrect"
    # With alpha=1.0 and different labels, mixed labels should likely not be exactly 0 or 1 (unless lambda is 0 or 1)
    # We just check shape here.
    assert m_lbls.shape == (2, 19), "Mixup output label shape incorrect"
    print("    Data Loading and Mixup verified.")

    # 3. Model Initialization Demo
    print("\n[3] Testing Model Initialization...")
    # Initialize model (pretrained=False for speed in demo, avoiding download)
    model = BirdResNet(pretrained=False).to(device)

    # Forward pass verification
    with torch.no_grad():
        dummy_input = images.to(device)
        outputs = model(dummy_input)

    print(f"    Model Output Shape: {outputs.shape}")
    assert outputs.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"
    print("    Model forward pass verified.")

    # 4. Trainer and SWA Demo
    print("\n[4] Testing Trainer and SWA Loop...")

    # Setup training components
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=model, device=device, criterion=criterion, optimizer=optimizer
    )

    # Run one training epoch
    print("    Running single training epoch...")
    train_loss = trainer.train_epoch(train_loader)
    print(f"    Train Loss: {train_loss:.4f}")
    assert isinstance(train_loss, float), "Train loss should be a float"

    # Run validation
    # Using train_loader as val_loader just for demo purposes
    print("    Running validation...")
    val_loss, val_auc = trainer.validate(train_loader)
    print(f"    Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

    # Run SWA Fit
    # We simulate a very short training cycle: 2 epochs total, SWA starts at epoch 1
    save_path = os.path.join(Config.WORKING_DIR, "demo_swa_model.pth")
    print("    Running SWA fit (2 epochs)...")
    swa_model = trainer.fit_swa(
        train_loader=train_loader,
        val_loader=train_loader,
        total_epochs=2,
        swa_start_epoch=1,
        swa_lr=1e-3,
        save_path=save_path,
    )

    assert os.path.exists(save_path), "SWA model file was not saved"
    assert isinstance(
        swa_model, torch.nn.Module
    ), "Returned object is not a PyTorch module"
    print("    Trainer and SWA verified.")

    # 5. Pseudo-labeling and Utils Demo
    print("\n[5] Testing Utils and Pseudo-labeling...")

    # Test ROC AUC calculation
    y_true = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
    y_pred = np.array([[0.1, 0.9, 0.2], [0.8, 0.1, 0.1], [0.2, 0.3, 0.8]])
    auc = calculate_roc_auc(y_true, y_pred)
    print(f"    Calculated AUC (dummy data): {auc:.4f}")
    assert 0.0 <= auc <= 1.0, "AUC score out of range"

    # Test Sanitize Pseudo Labels
    raw_pseudo = np.array([[0.5, np.nan], [-0.1, 1.5]])
    clean_pseudo = sanitize_pseudo_labels(raw_pseudo)
    assert not np.isnan(clean_pseudo).any(), "NaNs found in sanitized labels"
    assert (
        clean_pseudo.min() >= 0.0 and clean_pseudo.max() <= 1.0
    ), "Values not clipped to [0, 1]"
    print("    Utils verified.")

    # Test Dataset with Pseudo-labels
    # Create a dummy pseudo-label map for the first few IDs found in the loader
    # We need actual IDs from the dataset
    dataset_ids = [
        int(train_loader.dataset.df.iloc[i]["rec_id"])
        for i in range(len(train_loader.dataset))
    ]
    target_id = dataset_ids[0]

    # Create a random probability vector
    pseudo_vec = np.random.rand(Config.NUM_CLASSES).astype(np.float32)
    pseudo_dict = {target_id: pseudo_vec}

    # Re-initialize dataset with pseudo-labels
    pl_dataset = BirdDataset(
        train_loader.dataset.df, mode="train", pseudo_labels=pseudo_dict
    )

    # Fetch the item corresponding to target_id
    # We iterate to find the index since dataset might be shuffled or subsetted
    target_idx = -1
    for idx in range(len(pl_dataset)):
        if int(pl_dataset.df.iloc[idx]["rec_id"]) == target_id:
            target_idx = idx
            break

    if target_idx != -1:
        _, label_tensor, _ = pl_dataset[target_idx]
        # Check if the label returned matches our pseudo-label
        # Note: Dataset returns torch tensor, our pseudo_vec is numpy
        assert np.allclose(
            label_tensor.numpy(), pseudo_vec, atol=1e-5
        ), "Pseudo-label injection failed"
        print("    Pseudo-label injection verified.")
    else:
        print(
            "    Skipping specific pseudo-label check (ID not found in debug subset)."
        )

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    run_demo()
