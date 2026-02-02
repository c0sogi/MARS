import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score

# Import from provided library files
from library.utils import seed_everything, get_device
from library.model import IcebergResNet
from library.data import process_split, IcebergDataset, get_transforms

# Constants
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_5/"
SUBMISSION_DIR = "./submission"
INPUT_DIR = "./input"


def apply_label_smoothing(targets, epsilon=0.05):
    """
    Applies label smoothing to binary targets.
    Formula: y_smooth = y * (1 - epsilon) + 0.5 * epsilon

    Args:
        targets (torch.Tensor): Binary targets [0, 1].
        epsilon (float): Smoothing factor.

    Returns:
        torch.Tensor: Smoothed targets.
    """
    return targets * (1 - epsilon) + 0.5 * epsilon


def train_one_epoch(model, loader, criterion, optimizer, device, epsilon=0.05):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for images, angles, labels in loader:
        images = images.to(device)
        angles = angles.to(device)
        # Ensure labels are (Batch, 1)
        labels = labels.to(device).unsqueeze(1)

        # Apply label smoothing
        smooth_labels = apply_label_smoothing(labels, epsilon)

        optimizer.zero_grad()

        logits = model(images, angles)
        loss = criterion(logits, smooth_labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate_one_epoch(model, loader, criterion, device):
    """
    Validates the model. Returns loss and metrics.
    Uses raw labels (no smoothing) for metric calculation.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, angles, labels in loader:
            images = images.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images, angles)

            # Calculate validation loss against raw labels for true performance
            loss = criterion(logits, labels)
            running_loss += loss.item() * images.size(0)

            # Store predictions for metrics
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Metrics
    # Clip predictions to avoid log(0)
    clipped_preds = np.clip(all_preds, 1e-15, 1 - 1e-15)
    val_log_loss = log_loss(all_targets, clipped_preds)

    # Accuracy (threshold 0.5)
    preds_binary = (all_preds > 0.5).astype(int)
    val_acc = accuracy_score(all_targets, preds_binary)

    return epoch_loss, val_log_loss, val_acc


def predict_test(model, loader, device):
    """
    Generates predictions for the test set using Test Time Augmentation (TTA).
    Averages predictions from: Original, Horizontal Flip, Vertical Flip.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, angles, ids in loader:
            images = images.to(device)
            angles = angles.to(device)

            # 1. Original
            logits = model(images, angles)
            probs = torch.sigmoid(logits)

            # 2. Horizontal Flip (dim 3 is width)
            images_h = torch.flip(images, [3])
            logits_h = model(images_h, angles)
            probs_h = torch.sigmoid(logits_h)

            # 3. Vertical Flip (dim 2 is height)
            images_v = torch.flip(images, [2])
            logits_v = model(images_v, angles)
            probs_v = torch.sigmoid(logits_v)

            # Average probabilities
            avg_probs = (probs + probs_h + probs_v) / 3.0

            all_preds.extend(avg_probs.cpu().numpy().flatten())
            all_ids.extend(ids)

    return np.array(all_ids), np.array(all_preds)


def run_fold_training(fold_idx, train_loader, val_loader, epochs, device, save_dir):
    """
    Executes the training loop for a single fold.
    """
    print(f"\nStarting Fold {fold_idx}...")

    model = IcebergResNet().to(device)

    # Optimizer: AdamW with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # Scheduler: ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5
    )

    # Criterion: BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epsilon=0.05
        )
        val_loss, val_log_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{epochs} - LR: {current_lr:.1e} - "
            f"Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - "
            f"Val LogLoss: {val_log_loss:.15f} - Val Acc: {val_acc:.6f}"
        )

        # Step scheduler based on validation loss
        scheduler.step(val_loss)

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

    # Save best model to disk
    save_path = os.path.join(save_dir, f"model_fold_{fold_idx}_best.pth")
    torch.save(best_model_state, save_path)
    print(
        f"Fold {fold_idx} finished. Best Val Loss: {best_val_loss:.6f}. Saved to {save_path}"
    )

    # Reload best weights
    model.load_state_dict(best_model_state)
    return model


def train_kfold_and_predict(epochs=20, batch_size=32, n_folds=5):
    """
    Main pipeline function:
    1. Loads and merges Train/Val data.
    2. Performs Stratified K-Fold Cross-Validation.
    3. Trains a model for each fold.
    4. Generates TTA predictions on the Test set for each fold.
    5. Averages predictions and saves submission.
    """
    seed_everything(42)
    device = get_device()
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # --- 1. Data Preparation ---
    print("Loading and merging data for Cross-Validation...")

    # Load original train split
    t_imgs, t_angs, t_lbls, t_ids = process_split(
        os.path.join(METADATA_DIR, "train_metadata.csv"), "train", load_cached_data=True
    )
    # Load original val split
    v_imgs, v_angs, v_lbls, v_ids = process_split(
        os.path.join(METADATA_DIR, "val_metadata.csv"), "val", load_cached_data=True
    )

    # Merge for K-Fold
    X_images = np.concatenate([t_imgs, v_imgs], axis=0)
    X_angles = np.concatenate([t_angs, v_angs], axis=0)
    y_labels = np.concatenate([t_lbls, v_lbls], axis=0)
    X_ids = np.concatenate([t_ids, v_ids], axis=0)

    print(f"Total training samples: {len(X_images)}")

    # Load Test Data
    test_imgs, test_angs, _, test_ids = process_split(
        os.path.join(METADATA_DIR, "test_metadata.csv"), "test", load_cached_data=True
    )

    test_ds = IcebergDataset(
        test_imgs, test_angs, None, test_ids, transform=get_transforms("test")
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    # --- 2. Stratified K-Fold Training ---
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_predictions = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_images, y_labels)):
        # Create Datasets for this fold
        train_ds = IcebergDataset(
            X_images[train_idx],
            X_angles[train_idx],
            y_labels[train_idx],
            X_ids[train_idx],
            transform=get_transforms("train"),
        )
        val_ds = IcebergDataset(
            X_images[val_idx],
            X_angles[val_idx],
            y_labels[val_idx],
            X_ids[val_idx],
            transform=get_transforms("val"),
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        # Train
        model = run_fold_training(
            fold_idx, train_loader, val_loader, epochs, device, WORKING_DIR
        )

        # Predict
        print(f"Generating predictions for Fold {fold_idx}...")
        ids, preds = predict_test(model, test_loader, device)
        fold_predictions.append(preds)

        # Cleanup to save memory
        del model, train_loader, val_loader, train_ds, val_ds
        torch.cuda.empty_cache()

    # --- 3. Ensemble Aggregation ---
    # Average predictions across all folds
    avg_preds = np.mean(fold_predictions, axis=0)

    # --- 4. Submission ---
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
