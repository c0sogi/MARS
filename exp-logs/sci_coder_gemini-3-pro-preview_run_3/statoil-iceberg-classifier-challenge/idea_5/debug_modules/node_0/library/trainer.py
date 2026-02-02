import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

from library.utils import seed_everything, get_device
from library.dataset import (
    process_subset,
    IcebergDataset,
    CACHE_DIR,
    INPUT_DIR,
    METADATA_DIR,
)
from library.model import MicroResNet, train_model


def load_data_for_cv(load_cached_data=True):
    """
    Loads and concatenates train and validation subsets to form the full training dataset.
    Handles caching and raw data loading if necessary.
    """
    # Load Metadata
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))

    # Calculate Angle Mean from Train Set (for imputation)
    angle_mean = train_meta["inc_angle"].mean()
    if np.isnan(angle_mean):
        angle_mean = 0.0

    # Check if we need to load raw JSONs
    modes = ["train", "val"]
    need_raw = False

    if not load_cached_data:
        need_raw = True
    else:
        for m in modes:
            if not (
                os.path.exists(os.path.join(CACHE_DIR, f"X_{m}.npy"))
                and os.path.exists(os.path.join(CACHE_DIR, f"angle_{m}.npy"))
                and os.path.exists(os.path.join(CACHE_DIR, f"ids_{m}.npy"))
                and os.path.exists(os.path.join(CACHE_DIR, f"y_{m}.npy"))
            ):
                need_raw = True
                break

    raw_data_dict = {}
    if need_raw:
        print("Loading raw JSON files for processing...")
        with open(os.path.join(INPUT_DIR, "train.json"), "r") as f:
            raw_data_dict["train.json"] = json.load(f)

    # Process Subsets
    X_train, ang_train, y_train, ids_train = process_subset(
        "train", train_meta, raw_data_dict, CACHE_DIR, load_cached_data, angle_mean
    )

    X_val, ang_val, y_val, ids_val = process_subset(
        "val", val_meta, raw_data_dict, CACHE_DIR, load_cached_data, angle_mean
    )

    # Concatenate to form full dataset for CV
    X_full = np.concatenate([X_train, X_val], axis=0)
    ang_full = np.concatenate([ang_train, ang_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)
    ids_full = np.concatenate([ids_train, ids_val], axis=0)

    return X_full, ang_full, y_full, ids_full, angle_mean


def load_test_data(load_cached_data=True, angle_mean=0.0):
    """
    Loads test data handling caching logic.
    """
    test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    need_raw = False
    if not load_cached_data:
        need_raw = True
    else:
        if not (
            os.path.exists(os.path.join(CACHE_DIR, "X_test.npy"))
            and os.path.exists(os.path.join(CACHE_DIR, "angle_test.npy"))
            and os.path.exists(os.path.join(CACHE_DIR, "ids_test.npy"))
        ):
            need_raw = True

    raw_data_dict = {}
    if need_raw:
        print("Loading raw test JSON...")
        with open(os.path.join(INPUT_DIR, "test.json"), "r") as f:
            raw_data_dict["test.json"] = json.load(f)

    X_test, ang_test, y_test, ids_test = process_subset(
        "test", test_meta, raw_data_dict, CACHE_DIR, load_cached_data, angle_mean
    )

    return X_test, ang_test, ids_test


def run_cross_validation(
    n_splits=5,
    epochs=50,
    batch_size=32,
    lr=1e-3,
    patience=10,
    num_workers=2,
    load_cached_data=True,
    seed=42,
):
    """
    Orchestrates 5-Fold Cross-Validation training.
    """
    seed_everything(seed)

    # 1. Load Full Data
    X, angles, y, ids, angle_mean = load_data_for_cv(load_cached_data)

    # 2. Setup K-Fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Transforms
    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
    )
    val_transform = None

    fold_results = []

    # 3. Loop Folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*20} Fold {fold+1}/{n_splits} {'='*20}")

        # Split Data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        ang_train_fold, ang_val_fold = angles[train_idx], angles[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        ids_train_fold, ids_val_fold = ids[train_idx], ids[val_idx]

        # Create Datasets
        train_dataset = IcebergDataset(
            X_train_fold,
            ang_train_fold,
            y_train_fold,
            ids_train_fold,
            transform=train_transform,
        )
        val_dataset = IcebergDataset(
            X_val_fold, ang_val_fold, y_val_fold, ids_val_fold, transform=val_transform
        )

        # Create Loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

        # Init Model
        model = MicroResNet()

        # Setup Save Path
        fold_dir = os.path.join(CACHE_DIR, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)
        save_path = os.path.join(fold_dir, "model_best.pth")

        # Train
        model, history = train_model(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            lr=lr,
            patience=patience,
            save_path=save_path,
        )

        # Store best val loss
        best_loss = min(history["val_loss"])
        fold_results.append(best_loss)
        print(f"Fold {fold+1} Best Val Loss: {best_loss}")

    print(f"\nCV Complete. Average Val Loss: {np.mean(fold_results)}")
    return angle_mean


def predict_and_submit(
    n_splits=5,
    batch_size=32,
    num_workers=2,
    load_cached_data=True,
    angle_mean=0.0,
    output_path="./submission/submission.csv",
):
    """
    Generates predictions using the ensemble of trained models with TTA.
    """
    device = get_device()
    seed_everything(42)

    # Load Test Data
    X_test, ang_test, ids_test = load_test_data(load_cached_data, angle_mean)

    test_dataset = IcebergDataset(X_test, ang_test, None, ids_test, transform=None)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Accumulate predictions
    ensemble_preds = np.zeros(len(ids_test))

    print("\nStarting Ensemble Prediction...")

    for fold in range(n_splits):
        print(f"Predicting with Fold {fold+1} model...")

        # Load Model
        model = MicroResNet().to(device)
        fold_dir = os.path.join(CACHE_DIR, f"fold_{fold}")
        model_path = os.path.join(fold_dir, "model_best.pth")

        if not os.path.exists(model_path):
            print(
                f"Warning: Model for fold {fold} not found at {model_path}. Skipping."
            )
            continue

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        fold_preds = []

        # TTA Prediction Loop
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                angles = batch["angle"].to(device)

                # TTA 1: Original
                pred_orig = model(images, angles)

                # TTA 2: Horizontal Flip
                images_h = torch.flip(images, [3])
                pred_h = model(images_h, angles)

                # TTA 3: Vertical Flip
                images_v = torch.flip(images, [2])
                pred_v = model(images_v, angles)

                # Average TTA predictions
                pred_avg = (pred_orig + pred_h + pred_v) / 3.0
                fold_preds.extend(pred_avg.cpu().numpy())

        ensemble_preds += np.array(fold_preds)

    # Average over folds
    ensemble_preds /= n_splits

    # Save Submission
    df = pd.DataFrame({"id": ids_test, "is_iceberg": ensemble_preds})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
