import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from library import config, utils, dataset, model


def train_one_epoch(net, loader, optimizer, criterion, device, epoch):
    """
    Trains the model for one epoch.
    """
    net.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    logger = utils.get_logger(f"train_epoch_{epoch}")

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)

        optimizer.zero_grad()

        logits = net(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Store predictions for AUC
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_targets.extend(targets.detach().cpu().numpy())
        all_preds.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        # Handle edge case with single class in batch
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(net, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    net.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            targets = batch["target"].to(device).unsqueeze(1)

            logits = net(images)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(probs)

    epoch_loss = running_loss / len(loader.dataset)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def run_fold(fold_idx, train_df, val_df):
    """
    Runs training and validation for a single fold.
    """
    logger = utils.get_logger(f"fold_{fold_idx}")
    logger.info(
        f"Starting Fold {fold_idx} | Train: {len(train_df)} | Val: {len(val_df)}"
    )

    device = utils.get_device()
    utils.seed_everything(config.SEED + fold_idx)

    # 1. Prepare DataLoaders
    # We use unique phase names to ensure caching separates folds correctly
    train_loader = dataset.get_dataloader(
        train_df,
        phase=f"fold_{fold_idx}_train",
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        load_cached_data=True,
    )

    val_loader = dataset.get_dataloader(
        val_df,
        phase=f"fold_{fold_idx}_val",
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        load_cached_data=True,
    )

    # 2. Initialize Model
    net = model.EfficientNet9Channel(
        backbone_name=config.BACKBONE, pretrained=True, num_classes=1
    )
    net = net.to(device)

    # 3. Setup Optimizer, Loss, Scheduler
    optimizer = optim.AdamW(
        net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    criterion = nn.BCEWithLogitsLoss()

    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)

    # 4. Training Loop
    best_auc = 0.0
    patience = 5
    patience_counter = 0
    best_model_path = os.path.join(config.CACHE_DIR, f"best_model_fold{fold_idx}.pth")

    for epoch in range(1, config.EPOCHS + 1):
        t0 = time.time()

        train_loss, train_auc = train_one_epoch(
            net, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_auc = validate(net, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch}/{config.EPOCHS} | Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.6f} | Train AUC: {train_auc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        # Save Best Model (Monitoring AUC)
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(net.state_dict(), best_model_path)
            logger.info(f"New best AUC: {best_auc:.6f}. Model saved.")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break

    return best_auc, best_model_path


def run_training():
    """
    Orchestrates the 5-Fold Cross-Validation training process.
    """
    logger = utils.get_logger("training_runner")

    # Load Metadata
    df_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(config.VAL_METADATA_PATH)

    # Combine for Stratified Split
    full_df = pd.concat([df_train, df_val], ignore_index=True)

    # Debug Mode
    if config.DEBUG:
        full_df = full_df.sample(
            n=config.DEBUG_DATASET_SIZE, random_state=config.SEED
        ).reset_index(drop=True)
        logger.info(f"DEBUG MODE: Using subset of {len(full_df)} samples.")

    skf = StratifiedKFold(
        n_splits=config.NUM_FOLDS, shuffle=True, random_state=config.SEED
    )

    fold_scores = []

    X = full_df.drop(columns=["MGMT_value"])
    y = full_df["MGMT_value"]

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        train_fold_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df = full_df.iloc[val_idx].reset_index(drop=True)

        score, _ = run_fold(fold_idx, train_fold_df, val_fold_df)
        fold_scores.append(score)

    logger.info("=" * 40)
    logger.info(f"CV Scores: {fold_scores}")
    logger.info(f"Mean AUC: {np.mean(fold_scores):.6f}")
    logger.info("=" * 40)


def generate_submission():
    """
    Generates predictions for the test set using an ensemble of all fold models.
    """
    logger = utils.get_logger("submission_runner")
    device = utils.get_device()

    # Load Test Metadata
    df_test = pd.read_csv(config.TEST_METADATA_PATH)

    # Handle Debug Mode to ensure array shape matches DataLoader output
    if config.DEBUG:
        df_test = df_test.head(config.DEBUG_DATASET_SIZE)
        logger.info(f"DEBUG MODE: Truncated test set to {len(df_test)} samples.")

    # Prepare DataLoader
    test_loader = dataset.get_dataloader(
        df_test,
        phase="test",
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        load_cached_data=True,
    )

    # Initialize Ensemble Prediction Array
    # Shape: (Num_Samples, Num_Folds)
    ensemble_preds = np.zeros((len(df_test), config.NUM_FOLDS))

    # Iterate through folds
    for fold_idx in range(config.NUM_FOLDS):
        model_path = os.path.join(config.CACHE_DIR, f"best_model_fold{fold_idx}.pth")

        if not os.path.exists(model_path):
            logger.warning(
                f"Model for fold {fold_idx} not found at {model_path}. Skipping."
            )
            continue

        logger.info(f"Predicting with model fold {fold_idx}...")

        # Load Model
        net = model.EfficientNet9Channel(
            backbone_name=config.BACKBONE, pretrained=False, num_classes=1
        )
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.to(device)
        net.eval()

        fold_preds = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                logits = net(images)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                fold_preds.extend(probs)

        ensemble_preds[:, fold_idx] = np.array(fold_preds)

    # Average Predictions
    avg_preds = np.mean(ensemble_preds, axis=1)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"BraTS21ID": df_test["BraTS21ID"], "MGMT_value": avg_preds}
    )

    # Save
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {config.SUBMISSION_PATH}")
    logger.info(submission_df.head())
