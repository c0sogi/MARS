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
    mixup_data,
    mixup_criterion,
)
from library.model import IcebergResNet18


def custom_update_bn(loader, model, device):
    """
    Custom update_bn to handle models with multiple inputs (image + angle).
    Standard torch.optim.swa_utils.update_bn only passes the first element of the batch.
    """
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    with torch.no_grad():
        for batch in loader:
            # Handle both labeled (img, ang, lbl) and unlabeled (img, ang) loaders
            if len(batch) == 3:
                images, angles, _ = batch
            else:
                images, angles = batch

            images = images.to(device)
            angles = angles.to(device)
            model(images, angles)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


def train_one_epoch(loader, model, optimizer, device, epoch):
    model.train()
    losses = AverageMeter()
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, angles, labels) in enumerate(loader):
        images = images.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        # Label Smoothing
        if Config.LABEL_SMOOTHING > 0:
            labels_smooth = (
                labels * (1.0 - Config.LABEL_SMOOTHING) + 0.5 * Config.LABEL_SMOOTHING
            )
        else:
            labels_smooth = labels

        outputs = model(images, angles)
        loss = criterion(outputs, labels_smooth)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate_tta(loader, model, device):
    """
    Validates using Test Time Augmentation (Original + H-Flip + V-Flip).
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

            # TTA 1: Original
            out1 = model(images, angles)
            prob1 = torch.sigmoid(out1)

            # TTA 2: Horizontal Flip (dim 3 is width)
            images_h = torch.flip(images, [3])
            out2 = model(images_h, angles)
            prob2 = torch.sigmoid(out2)

            # TTA 3: Vertical Flip (dim 2 is height)
            images_v = torch.flip(images, [2])
            out3 = model(images_v, angles)
            prob3 = torch.sigmoid(out3)

            # Average Probabilities
            avg_prob = (prob1 + prob2 + prob3) / 3.0

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


def run_fold(fold, train_idx, val_idx, data, output_dir):
    print(f"\n=== Starting Fold {fold} ===")

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
        drop_last=True,  # Important for BN stability
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

    # Schedulers
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=Config.PATIENCE
    )

    best_loss = float("inf")
    best_model_path = os.path.join(output_dir, f"model_fold_{fold}_best.pth")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(train_loader, model, optimizer, device, epoch)

        # Validate
        val_loss = validate_tta(val_loader, model, device)
        scheduler.step(val_loss)

        # Checkpoint Best Model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time
        print(
            f"Fold {fold} | Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
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

        state_dict = torch.load(path, map_location=device)
        # Clean state_dict if necessary
        if "n_averaged" in state_dict:
            del state_dict["n_averaged"]
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

                # TTA 1: Original
                out1 = model(images, angles)
                prob1 = torch.sigmoid(out1)

                # TTA 2: Horizontal Flip
                images_h = torch.flip(images, [3])
                out2 = model(images_h, angles)
                prob2 = torch.sigmoid(out2)

                # TTA 3: Vertical Flip
                images_v = torch.flip(images, [2])
                out3 = model(images_v, angles)
                prob3 = torch.sigmoid(out3)

                p = (prob1 + prob2 + prob3) / 3.0
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
