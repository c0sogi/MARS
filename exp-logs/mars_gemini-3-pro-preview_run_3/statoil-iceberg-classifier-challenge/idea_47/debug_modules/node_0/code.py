import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

# Import from the provided library
from library import config, utils, data, model, train


def main():
    print("=== Starting Demonstration Script ===")

    # 1. Setup
    utils.seed_everything(42)
    device = utils.get_device()
    print(f"Device selected: {device}")

    # 2. Data Processing Verification
    print("\n--- Step 1: Data Processing ---")
    # Load data (this handles caching internally)
    X_full, y_full, angle_full, ids_full, X_test, angle_test, ids_test = (
        config.process_data(load_cached_data=True)
    )

    # Assertions to verify data shapes
    # Images should be (N, 3, 75, 75)
    assert len(X_full.shape) == 4, "X_full should be 4D"
    assert X_full.shape[1] == 3, "Should have 3 channels (HH, HV, Avg)"
    assert X_full.shape[2] == 75 and X_full.shape[3] == 75, "Images should be 75x75"
    assert len(y_full) == len(X_full), "Labels length mismatch"
    assert len(angle_full) == len(X_full), "Angles length mismatch"

    print(f"Data loaded successfully.")
    print(f"Train shape: {X_full.shape}")
    print(f"Test shape: {X_test.shape}")

    # 3. Dataset & DataLoader Verification
    print("\n--- Step 2: Dataset & DataLoader ---")
    # Create a small subset for verification
    demo_ds = config.IcebergDataset(
        X_full[:32], y_full[:32], angle_full[:32], transform=None
    )
    demo_loader = DataLoader(demo_ds, batch_size=8, shuffle=False)

    # Fetch one batch
    imgs, angs, lbls = next(iter(demo_loader))

    # Verify batch shapes
    print(f"Batch images shape: {imgs.shape}")
    print(f"Batch angles shape: {angs.shape}")
    print(f"Batch labels shape: {lbls.shape}")

    assert imgs.shape == (8, 3, 75, 75), "Incorrect batch image shape"
    assert angs.shape == (8,), "Incorrect batch angle shape"
    assert lbls.shape == (8,), "Incorrect batch label shape"

    # 4. Model Verification
    print("\n--- Step 3: Model Architecture ---")
    net = model.IDPH_CNN().to(device)

    # Move batch to device
    imgs = imgs.to(device)
    angs = angs.to(device)

    # Forward pass check
    with torch.no_grad():
        out = net(imgs, angs)

    print(f"Model output shape: {out.shape}")
    assert out.shape == (8, 1), "Model output should be (Batch_Size, 1)"

    # 5. Training Demonstration (Optimized for Speed)
    print("\n--- Step 4: Training Loop (Fold 0, 2 Epochs) ---")

    # We simulate the first fold of the cross-validation
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    )
    train_idx, val_idx = next(skf.split(X_full, y_full))

    # Prepare data for this fold
    X_tr, X_val = X_full[train_idx], X_full[val_idx]
    y_tr, y_val = y_full[train_idx], y_full[val_idx]
    a_tr, a_val = angle_full[train_idx], angle_full[val_idx]

    # Create Datasets
    train_ds = config.IcebergDataset(X_tr, y_tr, a_tr, transform=None)
    val_ds = config.IcebergDataset(X_val, y_val, a_val, transform=None)

    # Create Loaders
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # Initialize Trainer
    # We use a fresh model for training
    training_model = model.IDPH_CNN().to(device)
    trainer = train.Trainer(training_model, device, fold_idx=0)

    # Run fit for only 2 epochs to demonstrate speed
    best_loss = trainer.fit(train_loader, val_loader, epochs=2)

    print(f"Training complete. Best Validation Loss: {best_loss:.4f}")

    # Verify checkpoint creation
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "model_fold_0.pth")
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    print(f"Checkpoint verified at {checkpoint_path}")

    # 6. Submission Generation
    print("\n--- Step 5: Submission Generation ---")
    # We call the library function. It will look for model_fold_0.pth.
    # It will print warnings for missing folds 1-4, which is expected and acceptable for this demo.
    train.generate_submission(load_cached_data=True)

    submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file was not created"

    # Verify submission content
    df_sub = pd.read_csv(submission_path)
    print(f"Submission loaded. Shape: {df_sub.shape}")
    print(df_sub.head())

    assert df_sub.shape[0] == len(
        ids_test
    ), f"Submission row count {df_sub.shape[0]} != test set size {len(ids_test)}"
    assert (
        "id" in df_sub.columns and "is_iceberg" in df_sub.columns
    ), "Submission columns missing"

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()
