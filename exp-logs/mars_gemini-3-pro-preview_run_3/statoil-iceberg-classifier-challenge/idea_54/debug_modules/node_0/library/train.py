import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from library.utils import set_seed, get_device, AverageMeter, save_checkpoint
from library.dataset import IcebergDataset
from library.model import SAICNN

# Configuration
WORKING_DIR = "./working/idea_54"
CHECKPOINT_DIR = os.path.join(WORKING_DIR, "checkpoints")
FOLD_DIR = os.path.join(WORKING_DIR, "folds")
SUBMISSION_DIR = "./submission"
METADATA_DIR = "./metadata"

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 75
PATIENCE = 12
N_FOLDS = 5
SEED = 42


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for (images, angles), targets in loader:
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images, angles)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for (images, angles), targets in loader:
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))

    return losses.avg


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []
    ids = []

    with torch.no_grad():
        for (images, angles), batch_ids in loader:
            images = images.to(device)
            angles = angles.to(device)

            outputs = model(images, angles)
            probs = torch.sigmoid(outputs)

            preds.extend(probs.cpu().numpy().flatten())
            ids.extend(batch_ids)

    return np.array(ids), np.array(preds)


def prepare_folds():
    """
    Merges train and validation metadata and creates stratified folds.
    Returns a list of (train_csv_path, val_csv_path) tuples.
    """
    os.makedirs(FOLD_DIR, exist_ok=True)

    # Load existing metadata
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))

    # Combine to create full training set for CV
    full_df = pd.concat([train_meta, val_meta], ignore_index=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_files = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(full_df, full_df["is_iceberg"])
    ):
        train_fold = full_df.iloc[train_idx].copy()
        val_fold = full_df.iloc[val_idx].copy()

        train_path = os.path.join(FOLD_DIR, f"train_fold_{fold_idx}.csv")
        val_path = os.path.join(FOLD_DIR, f"val_fold_{fold_idx}.csv")

        train_fold.to_csv(train_path, index=False)
        val_fold.to_csv(val_path, index=False)

        fold_files.append((train_path, val_path))

    return fold_files


def run_training():
    """
    Orchestrates the 5-Fold CV training process.
    """
    set_seed(SEED)
    device = get_device()
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    fold_files = prepare_folds()

    for fold_idx, (train_csv, val_csv) in enumerate(fold_files):
        print(f"--- Starting Fold {fold_idx} ---")

        # Initialize Datasets & Loaders
        train_dataset = IcebergDataset(train_csv, mode="train", load_cached_data=True)
        val_dataset = IcebergDataset(val_csv, mode="val", load_cached_data=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Initialize Model
        model = SAICNN().to(device)

        # Optimizer & Loss
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss = evaluate(model, val_loader, criterion, device)

            print(
                f"Fold {fold_idx} Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
            )

            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Save checkpoint
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
                },
                is_best,
                CHECKPOINT_DIR,
                fold_idx,
            )

            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Fold {fold_idx} Best Val Loss: {best_loss:.6f}")


def run_inference():
    """
    Generates submission by averaging predictions from all fold models.
    """
    set_seed(SEED)
    device = get_device()
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Load Test Data
    test_csv = os.path.join(METADATA_DIR, "test.csv")
    test_dataset = IcebergDataset(test_csv, mode="test", load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    fold_preds = []

    # Iterate over folds
    for fold_idx in range(N_FOLDS):
        model_path = os.path.join(CHECKPOINT_DIR, f"model_best_fold_{fold_idx}.pth")
        if not os.path.exists(model_path):
            print(f"Warning: Checkpoint for fold {fold_idx} not found at {model_path}")
            continue

        print(f"Inference with model fold {fold_idx}...")
        model = SAICNN().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])

        ids, preds = predict(model, test_loader, device)
        fold_preds.append(preds)

    if not fold_preds:
        print("Error: No predictions generated.")
        return

    # Average predictions
    avg_preds = np.mean(fold_preds, axis=0)

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"id": ids, "is_iceberg": avg_preds})

    # Save
    sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    df_sub.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


def main():
    """
    Main entry point for training and inference.
    """
    print("Starting Training Process...")
    run_training()
    print("Starting Inference Process...")
    run_inference()
    print("Done.")
