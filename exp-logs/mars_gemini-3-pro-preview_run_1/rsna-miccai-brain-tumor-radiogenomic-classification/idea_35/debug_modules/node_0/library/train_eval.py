import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from library.config import (
    DEVICE,
    NUM_FOLDS,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    PATIENCE,
    WORKING_DIR,
    SUBMISSION_PATH,
    NUM_WORKERS,
    SEED,
    IMG_SIZE,
)
from library.utils import AverageMeter, print_metric, set_seed
from library.model import SICAVModel
from library.dataset import SICAVDataset, get_transforms
from library.data_processing import load_data


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    losses = AverageMeter()
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)

            losses.update(loss.item(), images.size(0))
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Handle edge case where only one class is present in batch/loader
    if len(np.unique(all_targets)) < 2:
        auc = 0.5
    else:
        auc = roc_auc_score(all_targets, all_probs)

    return losses.avg, auc


def save_checkpoint(state, filename):
    """
    Saves the model state to a file.
    """
    torch.save(state, filename)


def load_checkpoint(model, filename, device):
    """
    Loads the model state from a file.
    """
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint)
        return True
    return False


def run_fold(
    fold_idx, train_ids, train_images, train_labels, val_ids, val_images, val_labels
):
    """
    Runs training for a single fold.
    """
    print(f"\nStarting Fold {fold_idx}...")

    # Create Datasets
    train_dataset = SICAVDataset(
        train_ids, train_images, train_labels, transforms=get_transforms("train")
    )
    val_dataset = SICAVDataset(
        val_ids, val_images, val_labels, transforms=get_transforms("val")
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model, Loss, Optimizer
    model = SICAVModel().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, f"best_model_fold{fold_idx}.pth")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_auc = evaluate(model, val_loader, criterion, DEVICE)

        print(
            f"Fold {fold_idx} | Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Fold {fold_idx} Finished. Best AUC: {best_auc}")
    return best_auc


def train_kfold():
    """
    Loads all training data, merges provided train/val splits, and runs 5-Fold CV.
    """
    set_seed(SEED)

    # 1. Load Data (Merge provided train and val for Cross-Validation)
    ids_train, imgs_train, lbls_train = load_data("train")
    ids_val, imgs_val, lbls_val = load_data("val")

    all_ids = np.concatenate([ids_train, ids_val])
    all_images = np.concatenate([imgs_train, imgs_val])
    all_labels = np.concatenate([lbls_train, lbls_val])

    print(f"Total training samples for CV: {len(all_ids)}")

    # 2. Stratified K-Fold
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_images, all_labels)):
        # Split data
        f_train_ids = all_ids[train_idx]
        f_train_imgs = all_images[train_idx]
        f_train_lbls = all_labels[train_idx]

        f_val_ids = all_ids[val_idx]
        f_val_imgs = all_images[val_idx]
        f_val_lbls = all_labels[val_idx]

        # Run Fold
        score = run_fold(
            fold_idx,
            f_train_ids,
            f_train_imgs,
            f_train_lbls,
            f_val_ids,
            f_val_imgs,
            f_val_lbls,
        )
        fold_scores.append(score)

    print(f"\nCV Complete. Average AUC: {np.mean(fold_scores)}")


def predict_and_submit():
    """
    Loads test data and all fold models, generates averaged predictions, and saves submission.
    """
    print("\nStarting Inference...")

    # 1. Load Test Data
    test_ids, test_images, test_labels = load_data("test")

    # Create Test Dataset/Loader
    test_dataset = SICAVDataset(
        test_ids, test_images, test_labels, transforms=get_transforms("test")
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Initialize Models
    models = []
    for fold_idx in range(NUM_FOLDS):
        model_path = os.path.join(WORKING_DIR, f"best_model_fold{fold_idx}.pth")
        if os.path.exists(model_path):
            model = SICAVModel().to(DEVICE)
            load_checkpoint(model, model_path, DEVICE)
            model.eval()
            models.append(model)
        else:
            print(f"Warning: Model for fold {fold_idx} not found at {model_path}")

    if not models:
        print("No models found for inference.")
        return

    # 3. Generate Predictions
    all_probs = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(DEVICE)

            batch_probs = []
            for model in models:
                logits = model(images)
                probs = torch.sigmoid(logits)
                batch_probs.append(probs.cpu().numpy())

            # Average predictions across models
            avg_probs = np.mean(batch_probs, axis=0)  # (B, 1)
            all_probs.extend(avg_probs.flatten())

    # 4. Save Submission
    df_sub = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": all_probs})

    df_sub.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
    print(df_sub.head())


def run_complete_pipeline():
    """
    Runs the full training and inference pipeline.
    """
    train_kfold()
    predict_and_submit()
