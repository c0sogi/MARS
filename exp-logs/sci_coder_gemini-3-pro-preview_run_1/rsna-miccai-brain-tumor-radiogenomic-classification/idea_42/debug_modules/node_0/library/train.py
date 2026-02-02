import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from library import config, utils, preprocessing, dataset, model


def train_one_epoch(model_instance, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model_instance.train()
    running_loss = 0.0
    dataset_size = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device).unsqueeze(1)  # (Batch, 1)

        optimizer.zero_grad()

        logits = model_instance(images)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate_one_epoch(model_instance, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and ROC AUC score.
    """
    model_instance.eval()
    running_loss = 0.0
    dataset_size = 0

    all_targets = []
    all_probs = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model_instance(images)
            loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)

            running_loss += loss.item() * images.size(0)
            dataset_size += images.size(0)

            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    # Handle edge case where only one class is present in the batch/fold (unlikely with StratifiedKFold)
    try:
        auc_score = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc_score = 0.5

    return epoch_loss, auc_score


def generate_submission(
    test_data,
    model_paths,
    device,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
):
    """
    Generates predictions for the test set using an ensemble of fold models.
    Saves the result to submission.csv.
    """
    logger = utils.get_logger("Submission")
    logger.info("Generating submission...")

    test_images, test_ids = test_data

    # Create Test Loader
    test_dataset = dataset.MGMTDataset(
        test_images, test_ids, transform=dataset.get_transforms("test"), is_test=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Array to store sum of probabilities for averaging
    # Shape: (Num_Test_Samples, 1)
    ensemble_probs = np.zeros((len(test_ids), 1), dtype=np.float32)

    valid_models_count = 0

    for fold_idx, path in enumerate(model_paths):
        if not os.path.exists(path):
            logger.warning(f"Model path {path} does not exist. Skipping.")
            continue

        logger.info(f"Inference with model fold {fold_idx}...")

        # Initialize model
        net = model.MGMTNet(pretrained=False)  # Weights loaded from checkpoint
        net.to(device)
        net.load_state_dict(torch.load(path, map_location=device))
        net.eval()

        fold_probs = []

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                logits = net(images)
                probs = torch.sigmoid(logits)
                fold_probs.append(probs.cpu().numpy())

        fold_probs = np.concatenate(fold_probs)
        ensemble_probs += fold_probs
        valid_models_count += 1

        # Cleanup
        del net
        torch.cuda.empty_cache()

    if valid_models_count > 0:
        avg_probs = ensemble_probs / valid_models_count
    else:
        logger.error("No valid models found for inference!")
        avg_probs = np.full((len(test_ids), 1), 0.5)

    # Create DataFrame
    df_sub = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": avg_probs.flatten()})

    # Save
    save_path = config.SUBMISSION_PATH
    df_sub.to_csv(save_path, index=False)
    logger.info(f"Submission saved to {save_path}")
    logger.info(f"Head:\n{df_sub.head()}")


def run_kfold(
    num_folds=config.NUM_FOLDS,
    epochs=config.NUM_EPOCHS,
    batch_size=config.BATCH_SIZE,
    learning_rate=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY,
    patience=config.EARLY_STOPPING_PATIENCE,
):
    """
    Executes the K-Fold Cross-Validation training pipeline.
    """
    logger = utils.get_logger("Training")
    utils.seed_everything(config.SEED)
    device = torch.device(config.DEVICE)

    logger.info(f"Starting {num_folds}-Fold Cross-Validation on device: {device}")

    # 1. Load Data
    # We load cached data if available.
    (train_data_raw, val_data_raw, test_data_raw) = preprocessing.prepare_datasets(
        load_cached_data=True
    )

    # Combine Train and Val for proper K-Fold splitting
    X_train_raw, y_train_raw = train_data_raw
    X_val_raw, y_val_raw = val_data_raw

    X_full = np.concatenate([X_train_raw, X_val_raw], axis=0)
    y_full = np.concatenate([y_train_raw, y_val_raw], axis=0)

    logger.info(f"Combined Dataset Shape: {X_full.shape}")

    # 2. K-Fold Setup
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=config.SEED)

    best_model_paths = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        logger.info(f"\n{'='*20} FOLD {fold} {'='*20}")

        # Split Data
        X_train, y_train = X_full[train_idx], y_full[train_idx]
        X_val, y_val = X_full[val_idx], y_full[val_idx]

        # Create Datasets
        train_dset = dataset.MGMTDataset(
            X_train, y_train, transform=dataset.get_transforms("train")
        )
        val_dset = dataset.MGMTDataset(
            X_val, y_val, transform=dataset.get_transforms("val")
        )

        # Create Loaders
        train_loader = DataLoader(
            train_dset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        net = model.MGMTNet(pretrained=True)
        net.to(device)

        # Optimizer & Loss
        optimizer = optim.AdamW(
            net.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()

        # Tracking
        best_auc = 0.0
        patience_counter = 0
        best_model_path = os.path.join(config.WORKING_DIR, f"best_model_fold{fold}.pth")
        best_model_paths.append(best_model_path)

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                net, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc = validate_one_epoch(net, val_loader, criterion, device)

            logger.info(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc}"
            )

            # Early Stopping Check
            if val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                torch.save(net.state_dict(), best_model_path)
                logger.info(f"  -> New Best AUC! Model saved to {best_model_path}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                break

        # Cleanup for next fold
        del net, optimizer, criterion, train_loader, val_loader, train_dset, val_dset
        torch.cuda.empty_cache()
        gc.collect()

    logger.info("\nTraining Complete.")

    # 3. Generate Submission
    generate_submission(test_data_raw, best_model_paths, device)
