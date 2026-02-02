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
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    SUBMISSION_PATH,
    SEED,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    NUM_FOLDS,
    NUM_WORKERS,
    DEVICE,
    USE_AMP,
    seed_everything,
)
from library.data_processing import process_dataset_roi
from library.dataset import RNWIVDataset, get_transforms
from library.model import RNWIVEfficientNet


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

        # Collect metrics
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Performs validation inference.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = model(images)
                loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def run_training():
    """
    Orchestrates the 5-Fold Cross-Validation training loop.
    """
    seed_everything(SEED)

    # 1. Load Metadata
    if not os.path.exists(TRAIN_METADATA_PATH) or not os.path.exists(VAL_METADATA_PATH):
        print("Metadata not found. Skipping training.")
        return

    df_train_meta = pd.read_csv(TRAIN_METADATA_PATH)
    df_val_meta = pd.read_csv(VAL_METADATA_PATH)

    # Combine for Cross-Validation (StratifiedKFold on full dataset)
    df_full = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

    # 2. Process/Load ROI Boundaries (Cached)
    roi_df = process_dataset_roi(df_full, load_cached_data=True)

    # 3. Cross-Validation Setup
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    # 4. Training Loop
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_full, df_full["MGMT_value"])
    ):
        print(f"\n=== Fold {fold} ===")

        train_sub = df_full.iloc[train_idx].reset_index(drop=True)
        val_sub = df_full.iloc[val_idx].reset_index(drop=True)

        # Datasets & Loaders
        train_ds = RNWIVDataset(
            train_sub, roi_df, transform=get_transforms("train"), is_train=True
        )
        val_ds = RNWIVDataset(
            val_sub, roi_df, transform=get_transforms("val"), is_train=False
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Model, Criterion, Optimizer
        model = RNWIVEfficientNet().to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)

        best_auc = 0.0
        patience_counter = 0
        best_model_path = os.path.join(WORKING_DIR, f"best_model_fold{fold}.pth")

        for epoch in range(NUM_EPOCHS):
            train_loss, train_auc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, DEVICE
            )
            val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)

            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.8f} AUC: {train_auc:.8f} | "
                f"Val Loss: {val_loss:.8f} AUC: {val_auc:.8f}"
            )

            # Checkpoint & Early Stopping
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        # Validate best model
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

        _, fold_auc = validate(model, val_loader, criterion, DEVICE)
        print(f"Fold {fold} Best AUC: {fold_auc:.8f}")

    print("\nTraining Complete.")


def predict_and_submit():
    """
    Generates predictions for the test set using an ensemble of fold models.
    """
    print("\nStarting Inference on Test Set...")
    seed_everything(SEED)

    if not os.path.exists(TEST_METADATA_PATH):
        print("Test metadata not found.")
        return

    df_test = pd.read_csv(TEST_METADATA_PATH)

    # Process ROIs for test set
    roi_df = process_dataset_roi(df_test, load_cached_data=True)

    test_ds = RNWIVDataset(
        df_test, roi_df, transform=get_transforms("test"), is_train=False
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Load all available fold models
    models = []
    for fold in range(NUM_FOLDS):
        path = os.path.join(WORKING_DIR, f"best_model_fold{fold}.pth")
        if os.path.exists(path):
            model = RNWIVEfficientNet().to(DEVICE)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.eval()
            models.append(model)

    if not models:
        print("No trained models found. Cannot generate submission.")
        return

    avg_preds = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(DEVICE)

            batch_preds = []
            for model in models:
                with torch.amp.autocast("cuda", enabled=USE_AMP):
                    logits = model(images)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    batch_preds.append(probs)

            # Average across folds
            batch_avg = np.mean(batch_preds, axis=0)
            avg_preds.extend(batch_avg)

    # Create Submission
    submission = pd.DataFrame(
        {
            "BraTS21ID": df_test["BraTS21ID"],
            "MGMT_value": np.array(avg_preds).flatten(),
        }
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


def run():
    """
    Main entry point for the module.
    """
    run_training()
    predict_and_submit()
