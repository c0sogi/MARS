import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from skmultilearn.model_selection import IterativeStratification

# Import from provided library
from library.config import Config
from library.utils import set_seed, calculate_metric
from library.dataset import BirdDataset, prepare_data, get_transforms
from library.model import BirdResNet
from library.trainer import train_one_epoch, validate_one_epoch


def parse_label_matrix(df, num_classes):
    """
    Converts the 'labels' column (string of space-separated ints) into a binary matrix.
    """
    matrix = np.zeros((len(df), num_classes), dtype=int)
    for idx, row in df.iterrows():
        lbl_str = str(row["labels"])
        if lbl_str != "?" and lbl_str.strip():
            try:
                indices = [int(x) for x in lbl_str.split()]
                indices = [i for i in indices if 0 <= i < num_classes]
                matrix[idx, indices] = 1
            except ValueError:
                pass
    return matrix


def get_signal_stats(rec_id):
    """
    Computes signal energy from the cached spectrogram.
    """
    cache_path = os.path.join(Config.CACHE_DIR, f"{rec_id}.npy")
    if os.path.exists(cache_path):
        spec = np.load(cache_path)
        # spec is (n_mels, time), values are log-mel db (roughly)
        # We use mean value as a proxy for energy/loudness in this representation
        return np.mean(spec)
    return 0.0


def run():
    # 1. Setup
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Combine train and val for Cross-Validation
    dev_df = pd.concat([train_df, val_df], ignore_index=True)

    print(f"Development set size: {len(dev_df)}")
    print(f"Test set size: {len(test_df)}")

    # 3. Prepare Data (Cache Spectrograms)
    # We process unique rec_ids from both sets
    all_df = pd.concat([dev_df, test_df], ignore_index=True).drop_duplicates(
        subset=["rec_id"]
    )
    prepare_data(all_df, load_cached_data=True)

    # 4. Cross-Validation Setup
    n_folds = Config.N_FOLDS
    X = dev_df["rec_id"].values.reshape(-1, 1)
    y = parse_label_matrix(dev_df, Config.NUM_CLASSES)

    # Use IterativeStratification for multi-label data
    stratifier = IterativeStratification(n_splits=n_folds, order=1)

    # Store OOF predictions and Test predictions
    oof_preds = np.zeros((len(dev_df), Config.NUM_CLASSES))
    oof_targets = np.zeros((len(dev_df), Config.NUM_CLASSES))
    test_preds_accumulator = np.zeros((len(test_df), Config.NUM_CLASSES))

    # Map original indices to OOF array
    # Since we iterate, we need to track which sample corresponds to which row in dev_df
    # IterativeStratification returns indices relative to the input arrays

    fold = 0
    for train_idx, val_idx in stratifier.split(X, y):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")

        # Split Data
        fold_train_df = dev_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = dev_df.iloc[val_idx].reset_index(drop=True)

        # Create Datasets
        train_ds = BirdDataset(
            fold_train_df, mode="train", transform=get_transforms("train")
        )
        val_ds = BirdDataset(fold_val_df, mode="val", transform=get_transforms("val"))

        # Create Loaders
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

        # Initialize Model
        model = BirdResNet(pretrained=Config.PRETRAINED, num_classes=Config.NUM_CLASSES)
        model = model.to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
        )

        # Training Loop
        best_fold_auc = 0.0
        best_model_state = None

        for epoch in range(Config.EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc = validate_one_epoch(model, val_loader, criterion, device)

            scheduler.step()

            if val_auc > best_fold_auc:
                best_fold_auc = val_auc
                best_model_state = model.state_dict()
                # print(f"  Epoch {epoch+1}: New Best AUC {val_auc:.4f}")

        print(f"  Best Fold AUC: {best_fold_auc:.6f}")

        # Load Best Model for Inference
        model.load_state_dict(best_model_state)
        model.eval()

        # Generate OOF Predictions for this fold
        with torch.no_grad():
            fold_val_preds = []
            fold_val_targets = []
            for images, labels in val_loader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.sigmoid(outputs).cpu().numpy()
                fold_val_preds.append(preds)
                fold_val_targets.append(labels.numpy())

            fold_val_preds = np.concatenate(fold_val_preds, axis=0)
            fold_val_targets = np.concatenate(fold_val_targets, axis=0)

            # Store in global OOF arrays
            oof_preds[val_idx] = fold_val_preds
            oof_targets[val_idx] = fold_val_targets

        # Generate Test Predictions for this fold
        test_ds = BirdDataset(test_df, mode="test", transform=get_transforms("test"))
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        with torch.no_grad():
            fold_test_preds = []
            # We need to ensure order is preserved. DataLoader is sequential.
            for images, _, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                preds = torch.sigmoid(outputs).cpu().numpy()
                fold_test_preds.append(preds)

            fold_test_preds = np.concatenate(fold_test_preds, axis=0)
            test_preds_accumulator += fold_test_preds

        fold += 1

    # 5. Final Evaluation
    final_val_auc = calculate_metric(oof_targets, oof_preds)
    print(f"Final Validation Metric: {final_val_auc}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Compute per-sample error (Mean Squared Error)
    mse_per_sample = np.mean((oof_targets - oof_preds) ** 2, axis=1)

    # Feature 1: Number of labels
    num_labels = np.sum(oof_targets, axis=1)

    # Feature 2: Signal Energy (Mean Spectrogram Value)
    signal_energies = np.array([get_signal_stats(rid) for rid in dev_df["rec_id"]])

    # Correlations
    corr_labels = np.corrcoef(mse_per_sample, num_labels)[0, 1]
    corr_energy = np.corrcoef(mse_per_sample, signal_energies)[0, 1]

    print(f"Correlation (Error vs Num Labels): {corr_labels:.4f}")
    print(f"Correlation (Error vs Signal Energy): {corr_energy:.4f}")

    # 7. Submission
    threshold = 0.9072993371210134
    if final_val_auc > threshold:
        print(
            f"\nValidation metric ({final_val_auc}) > threshold ({threshold}). Generating submission..."
        )

        # Average test predictions
        avg_test_preds = test_preds_accumulator / n_folds

        # Format submission
        submission_rows = []
        rec_ids = test_df["rec_id"].values

        for i, rid in enumerate(rec_ids):
            probs = avg_test_preds[i]
            for species_idx, prob in enumerate(probs):
                row_id = int(rid * 100 + species_idx)
                submission_rows.append({"Id": row_id, "Probability": prob})

        sub_df = pd.DataFrame(submission_rows)
        sub_df = sub_df.sort_values("Id")

        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric ({final_val_auc}) <= threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    run()
