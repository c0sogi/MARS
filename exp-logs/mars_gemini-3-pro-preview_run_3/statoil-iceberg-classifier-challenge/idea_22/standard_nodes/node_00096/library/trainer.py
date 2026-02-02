import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import set_seed, save_checkpoint
from library.dataset import get_dataloaders, IcebergDataset
from library.model import MASHCNN


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for batch in loader:
        images = batch["image"].to(device)
        angles = batch["angle"].to(device)
        labels = batch[1].to(device)  # dataset returns (sample_dict, label)

        # Forward pass
        outputs = model(images, angles)
        loss = criterion(outputs, labels.view(-1, 1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        count += images.size(0)

    return running_loss / count


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            angles = batch["angle"].to(device)
            labels = batch[1].to(device)

            outputs = model(images, angles)
            loss = criterion(outputs, labels.view(-1, 1))

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

    return running_loss / count


def run_fold(fold, train_loader, val_loader):
    """
    Executes the training loop for a specific fold.
    """
    print(f"\nStarting Fold {fold}...")

    device = torch.device(Config.DEVICE)
    model = MASHCNN().to(device)

    # Optimizer: Adam with constant learning rate and weight decay
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        elapsed = time.time() - start_time

        print(
            f"Fold {fold} | Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.15f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpoint and Early Stopping
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_metric": best_val_loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best=True,
                fold=fold,
            )
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Fold {fold} Best Val Loss: {best_val_loss:.15f}")
    return best_val_loss


def train():
    """
    Main training orchestration function.
    Combines pre-split train/val data and performs Stratified K-Fold CV.
    """
    set_seed(Config.SEED)

    # 1. Get initial loaders to access the full labeled dataset and transforms
    # We ignore the fixed split provided by get_dataloaders for the training process
    # and instead merge them to perform our own K-Fold split.
    train_loader_orig, val_loader_orig, _ = get_dataloaders()

    # Extract data from the fixed training set
    X_train = train_loader_orig.dataset.images
    ang_train = train_loader_orig.dataset.angles
    y_train = train_loader_orig.dataset.labels
    ids_train = train_loader_orig.dataset.ids
    # Capture the transform with augmentation
    trans_train = train_loader_orig.dataset.transform

    # Extract data from the fixed validation set
    X_val = val_loader_orig.dataset.images
    ang_val = val_loader_orig.dataset.angles
    y_val = val_loader_orig.dataset.labels
    ids_val = val_loader_orig.dataset.ids
    # Capture the transform without augmentation
    trans_val = val_loader_orig.dataset.transform

    # Concatenate to form the full dataset
    X_all = np.concatenate([X_train, X_val], axis=0)
    ang_all = np.concatenate([ang_train, ang_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    ids_all = np.concatenate([ids_train, ids_val], axis=0)

    print(f"Total labeled samples for CV: {len(y_all)}")

    # 2. Setup Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    fold_metrics = []

    # 3. CV Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        # Create datasets for this fold
        # Train part gets augmentation transform
        ds_train = IcebergDataset(
            X_all[train_idx],
            ang_all[train_idx],
            y_all[train_idx],
            ids_all[train_idx],
            transform=trans_train,
        )

        # Val part gets clean transform
        ds_val = IcebergDataset(
            X_all[val_idx],
            ang_all[val_idx],
            y_all[val_idx],
            ids_all[val_idx],
            transform=trans_val,
        )

        # Create loaders
        dl_train = DataLoader(
            ds_train,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
            drop_last=True,
        )

        dl_val = DataLoader(
            ds_val,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=(Config.DEVICE == "cuda"),
        )

        # Run training for this fold
        best_loss = run_fold(fold, dl_train, dl_val)
        fold_metrics.append(best_loss)

    print("\n" + "=" * 30)
    print("Cross-Validation Complete")
    print(
        f"Average Log Loss across {Config.NUM_FOLDS} folds: {np.mean(fold_metrics):.15f}"
    )
    print("=" * 30)
