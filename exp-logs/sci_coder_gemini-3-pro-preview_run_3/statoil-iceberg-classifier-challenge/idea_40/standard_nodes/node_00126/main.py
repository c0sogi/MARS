import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import from provided library files
from library.config import Config
from library.utils import set_seed, compute_metrics
from library.data import IcebergDataset, _load_and_process
from library.model import IcebergModel


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    # Linear DropBlock Schedule
    progress = epoch / total_epochs
    current_drop_prob = Config.DROPBLOCK_START_PROB + progress * (
        Config.DROPBLOCK_MAX_PROB - Config.DROPBLOCK_START_PROB
    )
    current_drop_prob = min(current_drop_prob, Config.DROPBLOCK_MAX_PROB)
    model.set_dropblock_prob(current_drop_prob)

    for images, angles, targets in loader:
        images = images.to(device)
        angles = angles.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images, angles)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_targets, all_preds)
    return epoch_loss, metrics


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, angles, targets in loader:
            images = images.to(device)
            angles = angles.to(device)
            targets = targets.to(device)

            logits = model(images, angles)
            loss = criterion(logits, targets)

            running_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_targets, all_preds)
    return epoch_loss, metrics, np.array(all_preds), np.array(all_targets)


def predict_test(model, loader, device):
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for images, angles, ids in loader:
            images = images.to(device)
            angles = angles.to(device)

            logits = model(images, angles)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_preds.extend(probs)
            all_ids.extend(ids)

    return np.array(all_preds), all_ids


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Override epochs for fast baseline execution
    EPOCHS = 30
    PATIENCE = 8

    # 2. Data Loading & Preparation
    # We need to reconstruct the full training set for CV
    df_train_meta = pd.read_csv(Config.TRAIN_META)
    angle_impute_val = df_train_meta["inc_angle"].median()

    # Load Train part
    X_train_part, angles_train_part, y_train_part = _load_and_process(
        Config.TRAIN_META,
        Config.TRAIN_JSON,
        "train",
        angle_impute_val,
        load_cached_data=True,
    )
    # Load Val part
    X_val_part, angles_val_part, y_val_part = _load_and_process(
        Config.VAL_META,
        Config.TRAIN_JSON,
        "val",
        angle_impute_val,
        load_cached_data=True,
    )

    # Concatenate
    X_full = np.concatenate([X_train_part, X_val_part], axis=0)
    angles_full = np.concatenate([angles_train_part, angles_val_part], axis=0)
    y_full = np.concatenate([y_train_part, y_val_part], axis=0)

    print(f"Full dataset shape: {X_full.shape}")

    # Load Test Data
    X_test, angles_test, ids_test = _load_and_process(
        Config.TEST_META,
        Config.TEST_JSON,
        "test",
        angle_impute_val,
        load_cached_data=True,
    )
    test_dataset = IcebergDataset(
        X_test, angles_test, ids_test, transform=None, is_test=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Cross-Validation Loop
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)

    oof_preds = np.zeros(len(y_full))
    test_preds_accum = np.zeros(len(ids_test))

    # Store indices to map OOF back correctly
    # Since we concatenated, indices 0..len(train) are train, rest are val.
    # But we want to map based on the concatenated array indices.

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n--- Fold {fold + 1}/5 ---")

        # Split Data
        X_tr, X_val = X_full[train_idx], X_full[val_idx]
        a_tr, a_val = angles_full[train_idx], angles_full[val_idx]
        y_tr, y_val = y_full[train_idx], y_full[val_idx]

        # Create Datasets
        # Augmentation for train
        train_transform = torch.nn.Sequential(
            torch.nn.Identity()  # Placeholder, actual transform applied in Dataset via torchvision
        )
        # We use the transform logic from library.data manually or pass the transform object
        # library.data uses torchvision.transforms.Compose.
        from torchvision import transforms

        tf_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )

        train_ds = IcebergDataset(X_tr, a_tr, y_tr, transform=tf_train, is_test=False)
        val_ds = IcebergDataset(X_val, a_val, y_val, transform=None, is_test=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        # Model & Optimizer
        model = IcebergModel().to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(EPOCHS):
            train_loss, train_met = train_one_epoch(
                model, train_loader, optimizer, criterion, device, epoch, EPOCHS
            )
            val_loss, val_met, _, _ = validate(model, val_loader, criterion, device)

            # Simple logging
            # print(f"Ep {epoch+1} | T_Loss: {train_loss:.4f} | V_Loss: {val_loss:.4f} | V_Acc: {val_met['accuracy']:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        model.load_state_dict(best_model_state)

        # Generate OOF Preds
        _, _, val_probs, _ = validate(model, val_loader, criterion, device)
        oof_preds[val_idx] = val_probs

        # Generate Test Preds
        fold_test_preds, _ = predict_test(model, test_loader, device)
        test_preds_accum += fold_test_preds

        fold_metrics.append(best_loss)
        print(f"Fold {fold+1} Best Val Loss: {best_loss:.6f}")

    # 4. Evaluation & Failure Analysis
    final_metric = log_loss(y_full, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(y_full - oof_preds)

    # Compute simple features for correlation
    # Mean of Band 1 (channel 0) and Band 2 (channel 1)
    # X_full shape: (N, 3, 75, 75)
    b1_mean = np.mean(X_full[:, 0, :, :], axis=(1, 2))
    b2_mean = np.mean(X_full[:, 1, :, :], axis=(1, 2))
    b1_std = np.std(X_full[:, 0, :, :], axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": angles_full,
            "band_1_mean": b1_mean,
            "band_2_mean": b2_mean,
            "band_1_std": b1_std,
        }
    )

    correlations = analysis_df.corr()["error"].sort_values(ascending=False)
    print("Correlation with Error Magnitude:")
    print(correlations)

    # 5. Submission
    THRESHOLD = 0.1806015565870406

    if final_metric < THRESHOLD:
        print(f"\nMetric {final_metric} < {THRESHOLD}. Generating submission...")
        avg_test_preds = test_preds_accum / 5.0

        sub_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_test_preds})

        sub_path = "./submission/submission.csv"
        sub_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()
