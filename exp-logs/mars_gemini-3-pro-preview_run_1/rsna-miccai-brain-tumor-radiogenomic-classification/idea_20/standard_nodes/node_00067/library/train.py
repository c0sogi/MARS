import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, get_device
from library.data import get_datasets, WIVSDataset, get_transforms
from library.model import WIVSNet


def train_one_epoch(model, loader, optimizer, criterion, device):
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

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Store for metrics
        all_targets.append(targets.detach().cpu().numpy())
        all_preds.append(torch.sigmoid(outputs).detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def validate_epoch(model, loader, criterion, device):
    """
    Performs validation.
    """
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(torch.sigmoid(outputs).cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    all_targets = np.concatenate(all_targets)
    all_preds = np.concatenate(all_preds)

    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5

    return epoch_loss, epoch_auc


def run_fold(fold_idx, train_ds, val_ds, device):
    """
    Runs training for a single fold.
    """
    print(f"\nStarting Fold {fold_idx}...")

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    model = WIVSNet(pretrained=True)
    model.to(device)

    # Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = nn.BCEWithLogitsLoss()

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # Tracking
    best_auc = 0.0
    best_model_path = os.path.join(Config.MODEL_DIR, f"wivsnet_fold{fold_idx}.pth")
    patience = 5
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        train_loss, train_auc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_auc = validate_epoch(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.4f} AUC: {train_auc:.6f} | "
            f"Val Loss: {val_loss:.4f} AUC: {val_auc:.6f}"
        )

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            print(f"  >>> New Best AUC! Model saved to {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"  >>> Early stopping triggered after {epoch+1} epochs.")
            break

    # Cleanup
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return best_auc


def predict_test_set(test_dataset, num_folds, device):
    """
    Generates predictions for the test set by averaging outputs from all fold models.
    """
    print("\nStarting Inference on Test Set...")

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Array to store accumulated probabilities: (N_samples, 1)
    avg_preds = np.zeros((len(test_dataset), 1), dtype=np.float32)

    for fold_idx in range(num_folds):
        model_path = os.path.join(Config.MODEL_DIR, f"wivsnet_fold{fold_idx}.pth")
        if not os.path.exists(model_path):
            print(
                f"Warning: Model for fold {fold_idx} not found at {model_path}. Skipping."
            )
            continue

        print(f"Loading model for Fold {fold_idx}...")
        model = WIVSNet(pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        fold_preds = []

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                probs = torch.sigmoid(outputs).cpu().numpy()
                fold_preds.append(probs)

        fold_preds = np.concatenate(fold_preds, axis=0)
        avg_preds += fold_preds

        del model
        torch.cuda.empty_cache()
        gc.collect()

    # Average
    avg_preds /= num_folds
    return avg_preds.flatten()


def run_training():
    """
    Main execution function.
    """
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Using device: {device}")

    # 1. Load Data
    # library.data returns datasets based on metadata split.
    # To do 5-Fold CV properly, we merge them and re-split.
    print("Loading datasets...")
    train_ds_raw, val_ds_raw, test_ds = get_datasets(load_cached_data=True)

    # Merge train and val for Cross Validation
    all_images = np.concatenate([train_ds_raw.images, val_ds_raw.images], axis=0)
    all_labels = np.concatenate([train_ds_raw.labels, val_ds_raw.labels], axis=0)
    all_ids = np.concatenate([train_ds_raw.ids, val_ds_raw.ids], axis=0)

    print(f"Total labeled samples: {len(all_labels)}")
    print(f"Test samples: {len(test_ds)}")

    # 2. Cross Validation
    skf = StratifiedKFold(
        n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
    )

    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_images, all_labels)):
        # Create datasets for this fold
        # Note: We re-instantiate WIVSDataset to apply correct transforms (train vs valid)

        X_train, y_train, ids_train = (
            all_images[train_idx],
            all_labels[train_idx],
            all_ids[train_idx],
        )
        X_val, y_val, ids_val = (
            all_images[val_idx],
            all_labels[val_idx],
            all_ids[val_idx],
        )

        train_fold_ds = WIVSDataset(
            X_train, y_train, ids_train, transform=get_transforms("train")
        )
        val_fold_ds = WIVSDataset(
            X_val, y_val, ids_val, transform=get_transforms("valid")
        )

        # Run Fold
        best_auc = run_fold(fold_idx, train_fold_ds, val_fold_ds, device)
        fold_scores.append(best_auc)

    print("\n==================================")
    print("Cross-Validation Complete")
    print(f"Fold AUCs: {fold_scores}")
    print(f"Mean AUC: {np.mean(fold_scores):.6f}")
    print("==================================")

    # 3. Inference & Submission
    final_preds = predict_test_set(test_ds, Config.NUM_FOLDS, device)

    # Create submission DataFrame
    submission_df = pd.DataFrame({"BraTS21ID": test_ds.ids, "MGMT_value": final_preds})

    # Save
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    # This block is not required by the prompt instructions but facilitates local testing if run directly.
    # The prompt asks to implement the module functions.
    run_training()
