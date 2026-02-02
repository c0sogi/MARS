import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import CFG
from library.utils import seed_everything, calculate_class_weights
from library.data import AppleDataset, get_transforms, load_full_train_data
from library.modeling import get_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        # Prepare targets for CrossEntropyLoss
        # If labels are one-hot/probabilistic (N, C), convert to indices (N,)
        if labels.ndim == 2:
            targets = torch.argmax(labels, dim=1)
        else:
            targets = labels.long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def valid_one_epoch(model, loader, criterion, device):
    """
    Validates the model for one epoch.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            batch_size = images.size(0)

            if labels.ndim == 2:
                targets = torch.argmax(labels, dim=1)
            else:
                targets = labels.long()

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply softmax to get probabilities for ROC AUC
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    epoch_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Calculate Mean Column-wise ROC AUC
    try:
        score = roc_auc_score(all_labels, all_preds, average="macro", multi_class="ovr")
    except Exception:
        # Fallback for edge cases (e.g., only one class present in validation set)
        score = 0.0

    return epoch_loss, score


def train_models():
    """
    Main training loop implementing Heterogeneous K-Fold Ensemble.
    """
    seed_everything(CFG.seed)
    device = torch.device(CFG.device)

    print(f"Using device: {device}")

    # 1. Load Data
    df = load_full_train_data()

    # Ensure stratify label exists
    if "stratify_label" not in df.columns:
        df["stratify_label"] = df[CFG.target_cols].idxmax(axis=1)

    # 2. Stratified K-Fold Split
    skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)

    # 3. Iterate Architectures
    for arch in CFG.model_architectures:

        # 4. Iterate Folds
        for fold, (train_idx, val_idx) in enumerate(
            skf.split(df, df["stratify_label"])
        ):
            print(f"\n{'='*20}")
            print(f"Architecture: {arch} | Fold: {fold}")
            print(f"{'='*20}")

            # Prepare Fold Data
            train_df = df.iloc[train_idx].reset_index(drop=True)
            val_df = df.iloc[val_idx].reset_index(drop=True)

            train_dataset = AppleDataset(train_df, transform=get_transforms("train"))
            val_dataset = AppleDataset(val_df, transform=get_transforms("valid"))

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

            # Prepare Model
            model = get_model(arch, CFG.num_classes, pretrained=True)
            model.to(device)

            # Prepare Loss (Weighted Cross Entropy)
            weights = calculate_class_weights(
                train_df, CFG.target_cols, device=CFG.device
            )
            criterion = nn.CrossEntropyLoss(weight=weights)

            # Prepare Optimizer
            optimizer = torch.optim.Adam(
                model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay
            )

            # Prepare Scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=CFG.T_0, T_mult=CFG.T_mult, eta_min=CFG.min_lr
            )

            # Training Loop
            best_auc = 0.0
            patience = 10
            patience_counter = 0

            for epoch in range(CFG.epochs):
                train_loss = train_one_epoch(
                    model, train_loader, criterion, optimizer, device
                )
                val_loss, val_auc = valid_one_epoch(
                    model, val_loader, criterion, device
                )

                # Step scheduler at epoch end
                scheduler.step()

                # Print full precision as requested
                print(
                    f"Epoch {epoch+1} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val AUC: {val_auc}"
                )

                # Save Best Model
                if val_auc > best_auc:
                    best_auc = val_auc
                    patience_counter = 0

                    save_name = f"{arch}_fold_{fold}.pth"
                    save_path = os.path.join(CFG.models_dir, save_name)
                    torch.save(model.state_dict(), save_path)
                else:
                    patience_counter += 1

                # Early Stopping
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

            # Cleanup to free memory for next fold/arch
            del (
                model,
                optimizer,
                scheduler,
                train_loader,
                val_loader,
                criterion,
                weights,
            )
            gc.collect()
            torch.cuda.empty_cache()
