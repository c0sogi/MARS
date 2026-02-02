import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, average_checkpoints
from library.data import process_metadata, get_transforms, SkinLesionDataset
from library.model import HierarchicalEfficientNet
from library.engine import train_one_epoch, validate_one_epoch

if __name__ == "__main__":
    print("Starting demonstration of Skin Lesion Classification pipeline...")

    # 1. Configuration and Setup
    # Enable debug mode to use a smaller subset of data (1000 samples) and fewer epochs
    config = Config(debug=True, epochs=1)

    # Ensure reproducibility
    seed_everything(config.seed)
    print(f"Configuration loaded. Device: {config.device}")

    # 2. Data Processing
    print("\n[Step 1] Processing Metadata...")
    # process_metadata handles loading CSVs, imputing, scaling, and encoding
    # It returns tuples of (dataframe, meta_features, targets, diagnoses)
    (train_data, val_data, test_data, num_diag_classes) = process_metadata(
        config, load_cached_data=False
    )

    df_train, meta_train, target_train, diag_train = train_data
    df_val, meta_val, target_val, diag_val = val_data

    # Validation: Check data shapes
    print(f"Training samples: {len(df_train)}")
    print(f"Metadata feature count: {meta_train.shape[1]}")
    print(f"Diagnosis classes: {num_diag_classes}")

    assert (
        len(df_train) == len(meta_train) == len(target_train)
    ), "Train data length mismatch"
    assert meta_train.shape[1] > 0, "No metadata features generated"
    assert num_diag_classes > 1, "Diagnosis classes should be > 1"

    # 3. Dataset and Dataloader
    print("\n[Step 2] Creating Datasets and DataLoaders...")
    train_dataset = SkinLesionDataset(
        df=df_train,
        meta_features=meta_train,
        targets=target_train,
        diagnoses=diag_train,
        transforms=get_transforms(config.image_size, mode="train"),
        input_root=config.input_root,
    )

    val_dataset = SkinLesionDataset(
        df=df_val,
        meta_features=meta_val,
        targets=target_val,
        diagnoses=diag_val,
        transforms=get_transforms(config.image_size, mode="val"),
        input_root=config.input_root,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    print(f"Train Loader batches: {len(train_loader)}")
    print(f"Val Loader batches: {len(val_loader)}")

    # 4. Model Initialization
    print("\n[Step 3] Initializing HierarchicalEfficientNet...")
    model = HierarchicalEfficientNet(
        model_name=config.model_name,
        num_classes=config.num_classes,
        num_diag_classes=num_diag_classes,
        num_meta_features=meta_train.shape[1],
        pretrained=True,
    )
    model.to(config.device)

    # Validation: Check Model Forward Pass with Dummy Data
    print("Verifying model architecture with dummy batch...")
    dummy_img = torch.randn(2, 3, config.image_size, config.image_size).to(
        config.device
    )
    dummy_meta = torch.randn(2, meta_train.shape[1]).to(config.device)

    with torch.no_grad():
        p_logits, a_logits = model(dummy_img, dummy_meta)

    print(f"Primary Logits Shape: {p_logits.shape}")  # Should be (2, 1)
    print(
        f"Auxiliary Logits Shape: {a_logits.shape}"
    )  # Should be (2, num_diag_classes)

    assert p_logits.shape == (
        2,
        1,
    ), f"Expected primary logits shape (2, 1), got {p_logits.shape}"
    assert a_logits.shape == (
        2,
        num_diag_classes,
    ), f"Expected aux logits shape (2, {num_diag_classes}), got {a_logits.shape}"

    # 5. Training Loop
    print("\n[Step 4] Running Training Loop (1 Epoch)...")
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    # Simple scheduler for demonstration
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs * len(train_loader)
    )

    # Train
    train_loss = train_one_epoch(
        model, train_loader, optimizer, scheduler, config.device, config
    )
    print(f"Epoch 1 Training Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss is NaN"

    # Validate
    print("\n[Step 5] Running Validation Loop...")
    val_loss, val_auc = validate_one_epoch(model, val_loader, config.device, config)
    print(f"Epoch 1 Validation Loss: {val_loss:.4f}")
    print(f"Epoch 1 Validation AUC: {val_auc:.4f}")
    assert not np.isnan(val_loss), "Validation loss is NaN"
    assert 0.0 <= val_auc <= 1.0, "AUC score out of range"

    # 6. Checkpoint Averaging
    print("\n[Step 6] Demonstrating Checkpoint Averaging...")
    # Save current model as checkpoint A
    ckpt_path_a = os.path.join(config.working_dir, "ckpt_a.pth")
    torch.save(model.state_dict(), ckpt_path_a)

    # Modify model slightly and save as checkpoint B (to simulate a different epoch)
    with torch.no_grad():
        model.primary_head.weight += 0.01
    ckpt_path_b = os.path.join(config.working_dir, "ckpt_b.pth")
    torch.save(model.state_dict(), ckpt_path_b)

    # Average checkpoints
    print(f"Averaging checkpoints: {ckpt_path_a}, {ckpt_path_b}")
    averaged_state_dict = average_checkpoints([ckpt_path_a, ckpt_path_b])

    # Load averaged weights
    model.load_state_dict(averaged_state_dict)
    print("Averaged weights loaded successfully.")

    # Cleanup temporary checkpoints
    if os.path.exists(ckpt_path_a):
        os.remove(ckpt_path_a)
    if os.path.exists(ckpt_path_b):
        os.remove(ckpt_path_b)

    print("\nDemonstration completed successfully!")
