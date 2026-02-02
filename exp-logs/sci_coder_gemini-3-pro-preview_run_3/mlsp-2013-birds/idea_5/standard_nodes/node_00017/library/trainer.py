import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from library.config import CFG
from library.utils import AverageMeter, calculate_roc_auc, seed_everything
from library.dataset import (
    BirdDataset,
    load_or_compute_spectrograms,
    mixup_data,
    mixup_criterion,
)
from library.model import BirdResNet


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, epoch):
    """
    Handles the training of one epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (images, labels, _) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Apply Mixup if enabled
        if CFG.mixup:
            images, labels_a, labels_b, lam = mixup_data(
                images, labels, CFG.mixup_alpha
            )
            logits = model(images)
            loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        loss.backward()

        if CFG.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)

        optimizer.step()

        losses.update(loss.item(), images.size(0))

    if scheduler is not None:
        scheduler.step()

    return losses.avg


def valid_one_epoch(model, loader, criterion, device):
    """
    Handles validation inference and metric calculation.
    """
    model.eval()
    losses = AverageMeter()
    preds_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            losses.update(loss.item(), images.size(0))

            # Apply sigmoid to get probabilities for AUC calculation
            probs = torch.sigmoid(logits)

            preds_list.append(probs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    preds = np.concatenate(preds_list)
    targets = np.concatenate(labels_list)

    score = calculate_roc_auc(targets, preds)
    return losses.avg, score, preds


def run_training():
    """
    Main driver function for the training pipeline.
    """
    # 1. Setup
    seed_everything(CFG.seed)
    device = torch.device(CFG.device)
    os.makedirs(CFG.output_dir, exist_ok=True)
    os.makedirs(CFG.submission_dir, exist_ok=True)

    print(f"Training on device: {device}")

    # 2. Load Metadata
    train_df = pd.read_csv(CFG.train_csv)
    val_df = pd.read_csv(CFG.val_csv)
    test_df = pd.read_csv(CFG.test_csv)

    # Combine train and val for Cross Validation
    dev_df = pd.concat([train_df, val_df], ignore_index=True)

    # Debug mode
    if CFG.debug:
        print(f"Debug mode enabled. Using {CFG.debug_sample_size} samples.")
        dev_df = dev_df.iloc[: CFG.debug_sample_size]
        test_df = test_df.iloc[: CFG.debug_sample_size]

    print(f"Development samples: {len(dev_df)}")
    print(f"Test samples: {len(test_df)}")

    # 3. Load/Compute Spectrograms
    # Pass all dataframes to ensure all necessary files are processed
    all_dfs = [dev_df, test_df]
    spec_cache = load_or_compute_spectrograms(all_dfs, load_cached_data=True)

    # 4. Prepare Test Loader
    test_dataset = BirdDataset(test_df, spec_cache, phase="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # Accumulator for test predictions
    test_preds_accum = np.zeros((len(test_df), CFG.num_classes))

    # 5. Cross Validation Loop
    kf = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dev_df)):
        print(f"\n{'='*20} Fold {fold} {'='*20}")

        # Split Data
        fold_train_df = dev_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = dev_df.iloc[val_idx].reset_index(drop=True)

        # Datasets & Loaders
        train_dataset = BirdDataset(fold_train_df, spec_cache, phase="train")
        val_dataset = BirdDataset(fold_val_df, spec_cache, phase="val")

        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.batch_size,
            shuffle=True,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )

        # Model, Criterion, Optimizer
        model = BirdResNet(pretrained=CFG.pretrained, num_classes=CFG.num_classes)
        model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CFG.epochs, eta_min=CFG.min_lr
        )

        # Training Loop
        best_score = -np.inf
        patience_counter = 0
        best_model_path = os.path.join(CFG.output_dir, f"fold_{fold}_best.pth")

        for epoch in range(CFG.epochs):
            start_time = time.time()

            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, device, epoch
            )
            val_loss, val_score, _ = valid_one_epoch(
                model, val_loader, criterion, device
            )

            elapsed = time.time() - start_time

            if (epoch + 1) % CFG.print_freq == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1}/{CFG.epochs} | Time: {elapsed:.1f}s | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"Val AUC: {val_score:.10f}"
                )

            # Save Best
            if val_score > best_score:
                best_score = val_score
                torch.save(model.state_dict(), best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= CFG.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Fold {fold} Best AUC: {best_score:.10f}")

        # Inference on Test Set with Best Model
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        fold_test_preds = []
        with torch.no_grad():
            for images, _, _ in test_loader:
                images = images.to(device)
                logits = model(images)
                probs = torch.sigmoid(logits)
                fold_test_preds.append(probs.cpu().numpy())

        fold_test_preds = np.concatenate(fold_test_preds)
        test_preds_accum += fold_test_preds

    # 6. Average Predictions
    avg_test_preds = test_preds_accum / CFG.n_folds

    # 7. Create Submission
    print("Generating submission file...")
    submission_rows = []

    # Iterate through test dataframe to map predictions to IDs
    for i, row in test_df.iterrows():
        rec_id = int(row["rec_id"])
        probs = avg_test_preds[i]

        for species_idx in range(CFG.num_classes):
            # ID format: rec_id * 100 + species_number
            row_id = rec_id * 100 + species_idx
            prob = probs[species_idx]
            submission_rows.append({"Id": row_id, "Probability": prob})

    submission_df = pd.DataFrame(submission_rows)

    # Sort by Id
    submission_df = submission_df.sort_values("Id")

    sub_path = os.path.join(CFG.submission_dir, "submission.csv")
    submission_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")
