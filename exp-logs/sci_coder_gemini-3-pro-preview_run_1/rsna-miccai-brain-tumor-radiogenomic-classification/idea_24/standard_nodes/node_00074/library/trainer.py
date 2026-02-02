import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, get_device, setup_logger
from library.data_processing import get_centroids_with_caching
from library.dataset import BraTSDataset, get_transforms
from library.model import CAWIVModel


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Store predictions for AUC
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Handle edge case where batch might have only 1 class
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    """
    Executes validation loop.
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def predict(model, loader, device):
    """
    Generates predictions for a dataset (test or validation).
    Returns IDs and probabilities.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            # Handle both (images, targets) and (images, ids) formats
            if len(batch) == 2:
                images, identifiers = batch
            else:
                raise ValueError("Unexpected batch structure in predict")

            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()

            all_preds.extend(probs.flatten())

            # If identifiers are tensors (targets), convert to numpy, else keep as is (IDs)
            if isinstance(identifiers, torch.Tensor):
                all_ids.extend(identifiers.cpu().numpy())
            else:
                all_ids.extend(identifiers)

    return all_ids, all_preds


def run_training(load_cached_data=True, debug=False, patience=3):
    """
    Main pipeline function.

    Args:
        load_cached_data (bool): Whether to load centroids from cache.
        debug (bool): If True, runs on a small subset of data.
        patience (int): Early stopping patience.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = get_device()
    logger = setup_logger(os.path.join(Config.WORKING_DIR, "training.log"))

    logger.info("Starting Training Pipeline...")
    logger.info(f"Device: {device}")

    # 2. Load Metadata
    df_train_full = pd.read_csv(Config.TRAIN_METADATA)
    df_val_full = pd.read_csv(Config.VAL_METADATA)
    df_test = pd.read_csv(Config.TEST_METADATA)

    # Combine train and val for Cross-Validation
    df_all_train = pd.concat([df_train_full, df_val_full]).reset_index(drop=True)

    if debug:
        logger.info("DEBUG MODE: Using subset of data.")
        df_all_train = df_all_train.head(20)
        df_test = df_test.head(10)

    # 3. Prepare Centroids (Cache)
    logger.info("Processing/Loading Centroids...")
    centroids_train = get_centroids_with_caching(
        df_all_train,
        Config.INPUT_DIR,
        cache_name="centroids_train_val",
        load_cached_data=load_cached_data,
    )
    centroids_test = get_centroids_with_caching(
        df_test,
        Config.INPUT_DIR,
        cache_name="centroids_test",
        load_cached_data=load_cached_data,
    )

    # 4. Cross-Validation Setup
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # Storage for OOF and Test predictions
    oof_preds = np.zeros(len(df_all_train))
    # We accumulate test predictions from each fold
    test_preds_accum = np.zeros(len(df_test))

    # Transforms
    train_transform = get_transforms(mode="train")
    val_transform = get_transforms(mode="val")

    # 5. Training Loop
    for fold, (train_idx, val_idx) in enumerate(
        skf.split(df_all_train, df_all_train["MGMT_value"])
    ):
        logger.info(f"\n{'='*20} Fold {fold} {'='*20}")

        # Split Data
        train_sub = df_all_train.iloc[train_idx].reset_index(drop=True)
        val_sub = df_all_train.iloc[val_idx].reset_index(drop=True)

        # Datasets
        train_ds = BraTSDataset(
            train_sub,
            centroids_train,
            Config.INPUT_DIR,
            transform=train_transform,
            mode="train",
        )
        val_ds = BraTSDataset(
            val_sub,
            centroids_train,
            Config.INPUT_DIR,
            transform=val_transform,
            mode="val",
        )

        # Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model, Criterion, Optimizer
        model = CAWIVModel().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.NUM_EPOCHS
        )

        # Fold tracking
        best_auc = 0.0
        best_epoch = 0
        patience_counter = 0
        best_model_path = os.path.join(Config.CACHE_DIR, f"best_model_fold{fold}.pth")

        for epoch in range(Config.NUM_EPOCHS):
            train_loss, train_auc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc = validate(model, val_loader, criterion, device)
            scheduler.step()

            logger.info(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.6f} AUC: {train_auc:.6f} | Val Loss: {val_loss:.6f} AUC: {val_auc:.6f}"
            )

            # Checkpoint
            if val_auc > best_auc:
                best_auc = val_auc
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
                logger.info(f"  -> New Best AUC: {best_auc:.6f} (Saved)")
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        logger.info(
            f"Fold {fold} Finished. Best AUC: {best_auc:.6f} at Epoch {best_epoch+1}"
        )

        # Load Best Model for Inference
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        else:
            logger.warning("Best model file not found, using last epoch weights.")

        # OOF Inference
        _, val_probs = predict(model, val_loader, device)
        oof_preds[val_idx] = val_probs

        # Test Inference
        test_ds = BraTSDataset(
            df_test,
            centroids_test,
            Config.INPUT_DIR,
            transform=val_transform,
            mode="test",
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        _, fold_test_probs = predict(model, test_loader, device)
        test_preds_accum += np.array(fold_test_probs)

    # 6. Final Evaluation
    overall_auc = roc_auc_score(df_all_train["MGMT_value"], oof_preds)
    logger.info(f"\n{'='*40}")
    logger.info(f"Overall OOF AUC: {overall_auc:.10f}")
    logger.info(f"{'='*40}")

    # 7. Generate Submission
    # Average predictions
    avg_test_preds = test_preds_accum / Config.N_FOLDS

    submission_df = pd.DataFrame(
        {"BraTS21ID": df_test["BraTS21ID"], "MGMT_value": avg_test_preds}
    )

    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
