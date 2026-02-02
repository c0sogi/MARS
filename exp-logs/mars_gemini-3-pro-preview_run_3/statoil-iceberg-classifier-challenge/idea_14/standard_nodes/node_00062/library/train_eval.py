import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from torchvision import transforms

from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    SEED,
    N_FOLDS,
    NUM_EPOCHS,
    PATIENCE,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
    NUM_WORKERS,
)
from library.utils import set_seed, save_checkpoint
from library.model import SimpleCNN
from library.data_loader import get_dataloaders, IcebergDataset


def get_transforms():
    """
    Recreates the transforms defined in data_loader.py for use in custom CV loops.
    """
    # Train: Augmentation + ToTensor
    train_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
        ]
    )

    # Val/Test: ToTensor only
    eval_transform = transforms.Compose([transforms.ToTensor()])

    return train_transform, eval_transform


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    for batch in loader:
        # Unpack batch
        # Dataset returns: img, angle, label, id
        images, angles, labels, _ = batch

        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B) -> (B, 1)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = len(loader.dataset)

    with torch.no_grad():
        for batch in loader:
            images, angles, labels, _ = batch

            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []
    ids = []

    with torch.no_grad():
        for batch in loader:
            # Test loader returns: img, angle, id (no label)
            images, angles, batch_ids = batch

            images = images.to(device)
            angles = angles.to(device)

            outputs = model(images, angles)
            # Apply Sigmoid to get probabilities
            probs = torch.sigmoid(outputs).cpu().numpy()

            preds.extend(probs.flatten())
            ids.extend(batch_ids)

    return np.array(ids), np.array(preds)


def train_kfold(debug=False):
    """
    Main driver function to perform 5-Fold Cross-Validation training and submission generation.
    """
    set_seed(SEED)
    print(f"Starting {N_FOLDS}-Fold Cross-Validation Training (Debug={debug})...")

    # 1. Retrieve Data
    # We use get_dataloaders to handle the caching and loading of raw data.
    # Since we need to perform our own K-Fold split, we will extract the underlying
    # numpy arrays from the returned loaders and merge train/val.
    base_train_loader, base_val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=debug
    )

    # Extract Train Data
    X_train = base_train_loader.dataset.X
    angles_train = base_train_loader.dataset.angles
    y_train = base_train_loader.dataset.y
    ids_train = base_train_loader.dataset.ids

    # Extract Val Data
    X_val = base_val_loader.dataset.X
    angles_val = base_val_loader.dataset.angles
    y_val = base_val_loader.dataset.y
    ids_val = base_val_loader.dataset.ids

    # Merge for CV
    X_full = np.concatenate([X_train, X_val], axis=0)
    angles_full = np.concatenate([angles_train, angles_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)
    ids_full = np.concatenate([ids_train, ids_val], axis=0)

    print(f"Total Training Samples: {len(y_full)}")

    # Get Transforms
    train_tf, eval_tf = get_transforms()

    # 2. Cross-Validation Loop
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_results = {}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

        # Create Datasets for this fold
        train_ds = IcebergDataset(
            X_full[train_idx],
            angles_full[train_idx],
            y_full[train_idx],
            ids_full[train_idx],
            transform=train_tf,
        )
        val_ds = IcebergDataset(
            X_full[val_idx],
            angles_full[val_idx],
            y_full[val_idx],
            ids_full[val_idx],
            transform=eval_tf,
        )

        # Create Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model, Criterion, Optimizer
        model = SimpleCNN().to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        # Constant Learning Rate as per strategy
        optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Training Loop with Early Stopping
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(NUM_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )
            val_loss = evaluate(model, val_loader, criterion, DEVICE)

            # Print full precision as requested
            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss}"
            )

            # Checkpoint Logic
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "best_score": best_loss,
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best=True,
                    checkpoint_dir=WORKING_DIR,
                    fold_idx=fold,
                )
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        fold_results[fold] = best_loss

    # 3. Summary
    print("\nCross-Validation Results:")
    for f, loss in fold_results.items():
        print(f"Fold {f}: {loss}")
    avg_loss = np.mean(list(fold_results.values()))
    print(f"Average Log Loss: {avg_loss}")

    # 4. Inference & Submission
    print("\nGenerating Submission via Ensembling...")
    test_preds_sum = np.zeros(len(test_loader.dataset))
    test_ids = None

    for fold in range(N_FOLDS):
        print(f"Loading model for Fold {fold + 1}...")
        model = SimpleCNN().to(DEVICE)

        # Load best model weights
        checkpoint_path = os.path.join(WORKING_DIR, f"model_best_fold_{fold}.pth")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])

        # Predict
        ids, preds = predict(model, test_loader, DEVICE)

        # Accumulate
        test_preds_sum += preds
        if test_ids is None:
            test_ids = ids

    # Average predictions
    avg_preds = test_preds_sum / N_FOLDS

    # Create DataFrame
    df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})

    # Save
    df_sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
