import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
import glob

# Import from provided library
from library.config import (
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    WORKING_DIR,
    SUBMISSION_PATH,
    SEED,
    BATCH_SIZE,
    NUM_WORKERS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    INPUT_DROPOUT_PROB,
    IMG_SIZE,
    INPUT_DIR,
)
from library.utils import get_device, seed_everything
from library.data_processing import process_dataset
from library.dataset import BraTSDataset, get_transforms
from library.model import RNVSNetwork
from library.trainer import train_epoch, validate

# Constants for this run
NUM_FOLDS = 5
EPOCHS = 8  # Fast baseline
THRESHOLD = 0.6705454545454544


def get_filesystem_features(df):
    """
    Extracts simple filesystem metadata (file counts) to correlate with error.
    """
    features = []
    modalities = ["flair", "t1w", "t1wce", "t2w"]

    for _, row in df.iterrows():
        feat_row = {}
        for mod in modalities:
            # Metadata contains relative paths like 'train/00000/FLAIR'
            rel_path = row[f"{mod}_path"]
            full_path = os.path.join(INPUT_DIR, rel_path)
            if os.path.exists(full_path):
                # Fast count of files
                count = len(
                    [name for name in os.listdir(full_path) if name.endswith(".dcm")]
                )
                feat_row[f"{mod}_count"] = count
            else:
                feat_row[f"{mod}_count"] = 0
        features.append(feat_row)
    return pd.DataFrame(features)


def inference(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            preds.extend(probs.cpu().numpy().flatten())
    return np.array(preds)


def main():
    # 1. Setup
    seed_everything(SEED)
    device = get_device()
    os.makedirs(WORKING_DIR, exist_ok=True)

    print(f"Running on device: {device}")

    # 2. Data Loading & Preparation
    print("Loading metadata...")
    df_train_meta = pd.read_csv(TRAIN_METADATA_PATH)
    df_val_meta = pd.read_csv(VAL_METADATA_PATH)

    # Combine for CV
    df_full = pd.concat([df_train_meta, df_val_meta], ignore_index=True)

    # Process Dataset (Caching to working dir)
    cache_ids = os.path.join(WORKING_DIR, "full_train_ids.npy")
    cache_imgs = os.path.join(WORKING_DIR, "full_train_images.npy")
    cache_lbls = os.path.join(WORKING_DIR, "full_train_labels.npy")

    ids, images, labels = process_dataset(
        df_full, cache_ids, cache_imgs, cache_lbls, load_cached_data=True
    )

    # 3. K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    oof_preds = np.zeros(len(labels))
    fold_aucs = []

    print(f"Starting {NUM_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(images, labels)):
        print(f"\n=== Fold {fold} ===")

        # Split Data
        X_train, X_val = images[train_idx], images[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        ids_train, ids_val = ids[train_idx], ids[val_idx]

        # Create Datasets
        train_ds = BraTSDataset(
            X_train,
            y_train,
            ids_train,
            transform=get_transforms("train"),
            input_dropout_prob=INPUT_DROPOUT_PROB,
        )
        val_ds = BraTSDataset(
            X_val,
            y_val,
            ids_val,
            transform=get_transforms("val"),
            input_dropout_prob=0.0,
        )

        # Create Loaders
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

        # Model & Optimizer
        model = RNVSNetwork().to(device)
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_fold_auc = 0.0
        best_model_path = os.path.join(WORKING_DIR, f"best_model_fold{fold}.pth")

        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_auc = validate(model, val_loader, criterion, device)

            # Save Best
            if val_auc > best_fold_auc:
                best_fold_auc = val_auc
                torch.save(model.state_dict(), best_model_path)

        print(f"Fold {fold} Best AUC: {best_fold_auc:.6f}")
        fold_aucs.append(best_fold_auc)

        # Generate OOF Preds using best model
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        fold_preds = inference(model, val_loader, device)
        oof_preds[val_idx] = fold_preds

    # 4. Overall Evaluation
    final_auc = roc_auc_score(labels, oof_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate Error
    errors = np.abs(labels - oof_preds)

    # Extract Features (File counts)
    print("Extracting metadata features for correlation analysis...")
    df_features = get_filesystem_features(df_full)

    # Add error and target to dataframe
    df_features["error"] = errors
    df_features["target"] = labels

    # Correlation
    correlations = df_features.corr()["error"].sort_values(ascending=False)
    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # 6. Submission
    if final_auc > THRESHOLD:
        print(f"\nMetric {final_auc} > {THRESHOLD}. Generating submission...")

        # Load Test Data
        df_test_meta = pd.read_csv(TEST_METADATA_PATH)
        cache_test_ids = os.path.join(WORKING_DIR, "test_ids.npy")
        cache_test_imgs = os.path.join(WORKING_DIR, "test_images.npy")

        test_ids, test_images, _ = process_dataset(
            df_test_meta, cache_test_ids, cache_test_imgs, None, load_cached_data=True
        )

        test_ds = BraTSDataset(
            test_images,
            labels=None,
            ids=test_ids,
            transform=get_transforms("test"),
            input_dropout_prob=0.0,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Ensemble Inference
        avg_preds = np.zeros(len(test_ids))

        for fold in range(NUM_FOLDS):
            model_path = os.path.join(WORKING_DIR, f"best_model_fold{fold}.pth")
            model = RNVSNetwork().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))

            fold_test_preds = inference(model, test_loader, device)
            avg_preds += fold_test_preds

        avg_preds /= NUM_FOLDS

        # Save
        submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": avg_preds})
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")

    else:
        print(f"\nMetric {final_auc} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()
