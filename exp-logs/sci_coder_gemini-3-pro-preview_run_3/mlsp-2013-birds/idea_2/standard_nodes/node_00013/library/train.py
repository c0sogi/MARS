import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from skmultilearn.model_selection import IterativeStratification

from library.config import Config
from library.utils import (
    seed_everything,
    calculate_roc_auc,
    save_checkpoint,
    load_checkpoint,
    format_and_save_submission,
    AverageMeter,
)
from library.dataset import BirdDataset, load_dataset_df
from library.model import BirdClassifier


def mixup_data(x, y, alpha=1.0, device="cuda"):
    """
    Applies Mixup augmentation to the batch.
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Computes the Mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        # Apply Mixup if configured
        if Config.MIXUP_ALPHA > 0:
            data, targets_a, targets_b, lam = mixup_data(
                data, target, Config.MIXUP_ALPHA, device
            )
            outputs = model(data)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else:
            outputs = model(data)
            loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), data.size(0))

    return losses.avg


def validate_one_epoch(model, loader, criterion, device):
    """
    Validates the model for one epoch.
    Returns average loss and ROC AUC score.
    """
    model.eval()
    losses = AverageMeter()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            outputs = model(data)
            loss = criterion(outputs, target)
            losses.update(loss.item(), data.size(0))

            # Apply sigmoid to get probabilities
            preds = torch.sigmoid(outputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    if len(all_preds) == 0:
        return 0.0, 0.5

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    auc_score = calculate_roc_auc(all_targets, all_preds)

    return losses.avg, auc_score


def run_fold(fold_idx, train_df, val_df, device):
    """
    Runs the training process for a single fold.
    """
    print(f"\n[Fold {fold_idx}] Starting training...")
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")

    # Initialize Datasets
    train_dataset = BirdDataset(train_df, phase="train")
    val_dataset = BirdDataset(val_df, phase="val")

    # Initialize Dataloaders
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

    # Initialize Model
    model = BirdClassifier(
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED,
        in_channels=Config.IN_CHANNELS,
    )
    model.to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler (Cosine Annealing Warm Restarts)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # Loss Function
    criterion = nn.BCEWithLogitsLoss()

    # Tracking
    best_score = -1.0
    best_epoch = 0
    patience_counter = 0
    checkpoint_path = os.path.join(Config.WORKING_DIR, f"fold_{fold_idx}_best.pth")

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, device)

        scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.8f} | "
            f"Val Loss: {val_loss:.8f} | "
            f"Val AUC: {val_auc:.10f}"
        )

        # Save Best Model
        if val_auc > best_score:
            best_score = val_auc
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, best_score, checkpoint_path)
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    print(f"[Fold {fold_idx}] Best AUC: {best_score:.10f} at Epoch {best_epoch}")

    # Load best model for return
    if os.path.exists(checkpoint_path):
        load_checkpoint(model, None, checkpoint_path, device)
    else:
        print(
            f"[Fold {fold_idx}] Warning: No best model saved. Using last epoch model."
        )
        save_checkpoint(model, optimizer, Config.EPOCHS, val_auc, checkpoint_path)

    return model, best_score


def predict_test(model, test_loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            outputs = model(data)
            preds = torch.sigmoid(outputs)
            all_preds.append(preds.cpu().numpy())

    return np.concatenate(all_preds)


def train_and_evaluate(debug=False):
    """
    Main function to run the 5-fold cross-validation and generate submission.
    """
    seed_everything(Config.SEED)
    device = Config.DEVICE

    # Load Data
    # Combine train and val to perform 5-fold CV
    df_train_meta = load_dataset_df("train")
    df_val_meta = load_dataset_df("val")
    df_full = pd.concat([df_train_meta, df_val_meta], ignore_index=True)
    df_test = load_dataset_df("test")

    if debug:
        print("Debug mode enabled: Reducing dataset size and epochs.")
        df_full = df_full.head(50)
        df_test = df_test.head(20)
        Config.EPOCHS = 2
        Config.N_FOLDS = 2

    # Prepare Test Loader
    test_dataset = BirdDataset(df_test, phase="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Prepare Stratified Folds
    # Create binary label matrix for stratification
    num_classes = Config.NUM_CLASSES
    X = df_full["rec_id"].values.reshape(-1, 1)
    y = np.zeros((len(df_full), num_classes), dtype=int)

    for idx, row in df_full.iterrows():
        label_str = str(row["labels"])
        if label_str != "?" and label_str.strip() and label_str.lower() != "nan":
            try:
                indices = [int(x) for x in label_str.split()]
                for i in indices:
                    if 0 <= i < num_classes:
                        y[idx, i] = 1
            except ValueError:
                pass

    # Use IterativeStratification for multi-label data
    k_fold = IterativeStratification(n_splits=Config.N_FOLDS, order=1)

    fold_scores = []
    test_preds_accum = np.zeros((len(df_test), Config.NUM_CLASSES))

    print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

    for fold_idx, (train_indices, val_indices) in enumerate(k_fold.split(X, y)):
        train_df = df_full.iloc[train_indices].reset_index(drop=True)
        val_df = df_full.iloc[val_indices].reset_index(drop=True)

        # Run training for this fold
        model, score = run_fold(fold_idx, train_df, val_df, device)
        fold_scores.append(score)

        # Inference on Test Set
        print(f"[Fold {fold_idx}] Generating test predictions...")
        preds = predict_test(model, test_loader, device)
        test_preds_accum += preds

        # Clean up to save memory
        del model
        torch.cuda.empty_cache()

    avg_score = np.mean(fold_scores)
    print(f"\nAverage CV AUC: {avg_score:.10f}")

    # Average predictions across folds
    avg_test_preds = test_preds_accum / Config.N_FOLDS

    # Save Submission
    rec_ids = df_test["rec_id"].values
    format_and_save_submission(rec_ids, avg_test_preds, Config.OUTPUT_FILE)
    print(f"Submission saved to {Config.OUTPUT_FILE}")
