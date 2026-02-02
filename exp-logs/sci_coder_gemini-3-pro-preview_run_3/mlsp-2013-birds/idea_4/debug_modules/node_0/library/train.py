import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Try importing IterativeStratification for multi-label stratification
try:
    from skmultilearn.model_selection import IterativeStratification

    HAS_ITERATIVE = True
except ImportError:
    from sklearn.model_selection import KFold

    HAS_ITERATIVE = False

from library.config import Config
from library.utils import set_seed, calculate_roc_auc, save_checkpoint, load_checkpoint
from library.dataset import get_dataloaders, mixup_data, BirdDataset
from library.model import BirdResNet


def get_binary_labels(df):
    """Helper to convert dataframe labels to binary matrix for stratification."""
    num_classes = Config.NUM_CLASSES
    y = np.zeros((len(df), num_classes), dtype=int)
    for idx, row in df.iterrows():
        label_str = row["labels"]
        if pd.notna(label_str) and label_str != "?" and str(label_str).strip():
            try:
                indices = [int(x) for x in str(label_str).split()]
                for i in indices:
                    if 0 <= i < num_classes:
                        y[idx, i] = 1
            except ValueError:
                pass
    return y


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)
        batch_size = images.size(0)

        # Apply Mixup
        images, targets_a, targets_b, lam = mixup_data(
            images, targets, alpha=1.0, device=device
        )

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, targets_a) * lam + criterion(outputs, targets_b) * (
            1 - lam
        )
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    return running_loss / dataset_size


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply sigmoid for AUC calculation
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / dataset_size
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    epoch_auc = calculate_roc_auc(all_targets, all_preds)

    return epoch_loss, epoch_auc


def run_fold(fold_idx, train_df, val_df, load_cached_data=True):
    print(f"\n{'='*20} Fold {fold_idx} {'='*20}")
    print(f"Train Size: {len(train_df)}, Val Size: {len(val_df)}")

    # Get DataLoaders
    train_loader, val_loader, _ = get_dataloaders(
        train_df, val_df, pd.DataFrame(), load_cached_data=load_cached_data
    )

    # Initialize Model, Criterion, Optimizer
    device = Config.DEVICE
    model = BirdResNet(pretrained=Config.PRETRAINED, num_classes=Config.NUM_CLASSES)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # Training Loop
    best_auc = 0.0
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.10f} | "
            f"Time: {time.time() - start_time:.2f}s"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_checkpoint(
                {"state_dict": model.state_dict(), "auc": best_auc, "epoch": epoch},
                f"fold_{fold_idx}_best.pth",
            )
            print(f"--> Best AUC Updated: {best_auc:.10f}")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    return best_auc


def generate_submission(test_df, load_cached_data=True):
    print(f"\n{'='*20} Generating Submission {'='*20}")

    # Setup Test Loader
    # We pass empty train/val dfs to get_dataloaders just to get the test loader
    _, _, test_loader = get_dataloaders(
        pd.DataFrame(), pd.DataFrame(), test_df, load_cached_data=load_cached_data
    )

    device = Config.DEVICE
    num_folds = Config.NUM_FOLDS

    # Storage for ensemble predictions
    ensemble_preds = np.zeros((len(test_df), Config.NUM_CLASSES))

    # Iterate over folds
    models_found = 0
    for fold_idx in range(num_folds):
        checkpoint_name = f"fold_{fold_idx}_best.pth"
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, checkpoint_name)

        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint for fold {fold_idx} not found. Skipping.")
            continue

        print(f"Loading model for Fold {fold_idx}...")
        model = BirdResNet(pretrained=False, num_classes=Config.NUM_CLASSES)
        model.to(device)
        load_checkpoint(checkpoint_name, model, device=device)
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for images, _, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs)
                fold_preds.append(probs.cpu().numpy())

        fold_preds = np.concatenate(fold_preds, axis=0)
        ensemble_preds += fold_preds
        models_found += 1

    if models_found > 0:
        ensemble_preds /= models_found
    else:
        print("Error: No models found for inference.")
        return

    # Format Submission
    # Format: Id,Probability
    # Id = rec_id * 100 + species_id
    submission_rows = []

    # We need to map predictions back to rec_ids
    # The test_loader preserves order of test_df
    rec_ids = test_df["rec_id"].values

    for i, rec_id in enumerate(rec_ids):
        probs = ensemble_preds[i]
        for species_id in range(Config.NUM_CLASSES):
            row_id = int(rec_id * 100 + species_id)
            prob = probs[species_id]
            submission_rows.append({"Id": row_id, "Probability": prob})

    submission_df = pd.DataFrame(submission_rows)

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training(load_cached_data=True):
    set_seed(Config.SEED)

    # 1. Load Data
    train_orig = pd.read_csv(Config.TRAIN_CSV)
    val_orig = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Combine for CV
    dev_df = pd.concat([train_orig, val_orig], ignore_index=True)

    # 2. Stratified Split
    X = dev_df["rec_id"].values.reshape(-1, 1)
    y = get_binary_labels(dev_df)

    folds = []
    if HAS_ITERATIVE:
        print("Using IterativeStratification for splitting.")
        k_fold = IterativeStratification(n_splits=Config.NUM_FOLDS, order=1)
        for train_idx, val_idx in k_fold.split(X, y):
            folds.append((train_idx, val_idx))
    else:
        print("Using KFold for splitting.")
        k_fold = KFold(
            n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
        )
        for train_idx, val_idx in k_fold.split(X):
            folds.append((train_idx, val_idx))

    # 3. Train Folds
    fold_scores = []
    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        train_fold_df = dev_df.iloc[train_indices]
        val_fold_df = dev_df.iloc[val_indices]

        best_auc = run_fold(
            fold_idx, train_fold_df, val_fold_df, load_cached_data=load_cached_data
        )
        fold_scores.append(best_auc)

    print(f"\nCV Average AUC: {np.mean(fold_scores):.6f}")

    # 4. Inference
    generate_submission(test_df, load_cached_data=load_cached_data)
