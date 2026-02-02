import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import from library
from library.config import Config
from library.utils import seed_everything, AverageMeter
from library.data import (
    process_and_cache_data,
    IcebergDataset,
    get_transforms,
)
from library.model import IcebergResNet18


def train_one_epoch(loader, model, optimizer, device, epoch):
    model.train()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()
    smoothing = Config.LABEL_SMOOTHING

    for batch_idx, (images, angles, labels) in enumerate(loader):
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        # Apply Label Smoothing (Cite solution_lesson_node_00005)
        # y_ls = y * (1 - alpha) + 0.5 * alpha
        labels_smoothed = labels * (1.0 - smoothing) + 0.5 * smoothing

        optimizer.zero_grad()
        outputs = model(images, angles)
        loss = criterion(outputs, labels_smoothed)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate_tta(loader, model, device):
    """
    Validates using Test Time Augmentation (Original + Horizontal Flip).
    Returns Log Loss.
    """
    model.eval()
    losses = AverageMeter()
    criterion = nn.BCELoss()  # Using probabilities

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            # TTA: Original
            out1 = model(images, angles)
            prob1 = torch.sigmoid(out1)

            # TTA: Horizontal Flip (dim 3 is width)
            images_flip = torch.flip(images, [3])
            out2 = model(images_flip, angles)
            prob2 = torch.sigmoid(out2)

            # Average Probabilities
            avg_prob = (prob1 + prob2) / 2.0

            # Loss
            loss = criterion(avg_prob, labels)
            losses.update(loss.item(), images.size(0))

            all_preds.extend(avg_prob.cpu().numpy().flatten())
            all_targets.extend(labels.cpu().numpy().flatten())

    # Calculate global log loss
    eps = 1e-15
    preds = np.clip(all_preds, eps, 1 - eps)
    metric = log_loss(all_targets, preds)

    return metric


def train_single_model(train_idx, val_idx, data, output_dir):
    """
    Trains a single model on the specified split.
    Uses TTA-aware checkpointing (Cite solution_lesson_node_00025).
    """
    print(f"\n=== Starting Single Model Training ===")

    # Prepare Data
    X_train = data["train_images"][train_idx]
    ang_train = data["train_angles"][train_idx]
    y_train = data["train_labels"][train_idx]

    X_val = data["train_images"][val_idx]
    ang_val = data["train_angles"][val_idx]
    y_val = data["train_labels"][val_idx]

    # Debug Subset
    if Config.DEBUG:
        limit = min(Config.DEBUG_SUBSET_SIZE, len(X_train))
        X_train, ang_train, y_train = (
            X_train[:limit],
            ang_train[:limit],
            y_train[:limit],
        )
        X_val, ang_val, y_val = X_val[:limit], ang_val[:limit], y_val[:limit]
        print(f"DEBUG: Truncated data to {limit} samples.")

    # Datasets
    train_dataset = IcebergDataset(
        X_train, ang_train, y_train, transform=get_transforms("train")
    )
    val_dataset = IcebergDataset(
        X_val, ang_val, y_val, transform=get_transforms("valid")
    )

    # Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model Setup
    device = Config.DEVICE
    model = IcebergResNet18(
        pretrained=Config.PRETRAINED, dropout_rate=Config.DROPOUT_RATE
    )
    model = model.to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=Config.PATIENCE // 2
    )

    best_loss = float("inf")
    best_model_path = os.path.join(output_dir, "model_best.pth")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(train_loader, model, optimizer, device, epoch)

        # Validate with TTA (Cite solution_lesson_node_00025)
        val_loss = validate_tta(val_loader, model, device)
        scheduler.step(val_loss)

        # Checkpoint Best Model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"LR: {lr:.2e} | Time: {elapsed:.1f}s"
        )

    return best_model_path


def predict_test(model_paths, data, device):
    """
    Generates predictions for the test set using an ensemble of models.
    """
    print("\nGenerating Test Predictions...")
    X_test = data["test_images"]
    ang_test = data["test_angles"]
    ids_test = data["test_ids"]

    test_dataset = IcebergDataset(X_test, ang_test, transform=get_transforms("test"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Accumulate predictions
    avg_preds = np.zeros(len(X_test))

    for path in model_paths:
        print(f"Loading model: {path}")
        model = IcebergResNet18(pretrained=False)  # Weights loaded from state_dict

        # Handle SWA vs Standard state dicts
        state_dict = torch.load(path, map_location=device)
        # If SWA model, keys might start with 'module.' if not handled carefully,
        # but AveragedModel usually keeps keys clean or wraps them.
        # AveragedModel keys are usually 'module.layer...', we need to fix if necessary.
        # However, we saved swa_model.state_dict(). If swa_model was AveragedModel(model),
        # it has a 'module' attribute containing the actual model.

        # Check if keys start with 'n_averaged' (SWA metadata) and remove
        if "n_averaged" in state_dict:
            del state_dict["n_averaged"]

        # Fix keys: 'module.conv1...' -> 'conv1...'
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        model.load_state_dict(new_state_dict)
        model.to(device)
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for images, angles in test_loader:
                images = images.to(device)
                angles = angles.to(device)

                # TTA: Original + Flip
                out1 = model(images, angles)
                prob1 = torch.sigmoid(out1)

                images_flip = torch.flip(images, [3])
                out2 = model(images_flip, angles)
                prob2 = torch.sigmoid(out2)

                p = (prob1 + prob2) / 2.0
                fold_preds.extend(p.cpu().numpy().flatten())

        avg_preds += np.array(fold_preds)

    avg_preds /= len(model_paths)
    return ids_test, avg_preds


def train_loop():
    seed_everything(Config.SEED)

    # Load Data
    data = process_and_cache_data(load_cached_data=True)

    # Stratified K-Fold
    y = data["train_labels"]
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    output_dir = Config.WORKING_DIR
    os.makedirs(output_dir, exist_ok=True)

    saved_models = []

    # Train Folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        model_path = run_fold(fold, train_idx, val_idx, data, output_dir)
        saved_models.append(model_path)

    # Generate Submission
    ids, preds = predict_test(saved_models, data, Config.DEVICE)

    df_sub = pd.DataFrame({"id": ids, "is_iceberg": preds})

    save_path = Config.SUBMISSION_PATH
    df_sub.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


if __name__ == "__main__":
    train_loop()
