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
from library.inference import predict_with_tta


def load_train_data(load_cached_data=True):
    """
    Loads only the training subset for Cross-Validation.
    """
    # Load Metadata
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))

    # Calculate Angle Mean from Train Set (for imputation)
    angle_mean = train_meta["inc_angle"].mean()
    if np.isnan(angle_mean):
        angle_mean = 0.0

    # Check if we need to load raw JSONs
    mode = "train"
    need_raw = False

    if not load_cached_data:
        need_raw = True
    else:
        if not (
            os.path.exists(os.path.join(CACHE_DIR, f"X_{mode}.npy"))
            and os.path.exists(os.path.join(CACHE_DIR, f"angle_{mode}.npy"))
            and os.path.exists(os.path.join(CACHE_DIR, f"ids_{mode}.npy"))
            and os.path.exists(os.path.join(CACHE_DIR, f"y_{mode}.npy"))
        ):
            need_raw = True

    raw_data_dict = {}
    if need_raw:
        print("Loading raw JSON files for processing...")
        with open(os.path.join(INPUT_DIR, "train.json"), "r") as f:
            raw_data_dict["train.json"] = json.load(f)

    # Process Subset
    X_train, ang_train, y_train, ids_train = process_subset(
        "train", train_meta, raw_data_dict, CACHE_DIR, load_cached_data, angle_mean
    )

    return X_train, ang_train, y_train, ids_train, angle_mean


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
    num_workers=4,
    load_cached_data=True,
    seed=42,
    model_dir="./working/models",
):
    """
    Orchestrates 5-Fold Cross-Validation training on the training set.
    """
    seed_everything(seed)

    # 1. Load Train Data
    X, angles, y, ids, angle_mean = load_train_data(load_cached_data)

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
        fold_dir = os.path.join(model_dir, f"fold_{fold}")
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

    print(f"\nCV Complete. Average OOF Val Loss: {np.mean(fold_results)}")
    return angle_mean


def predict_ensemble(loader, n_splits, model_dir, device):
    """
    Generates predictions using the ensemble of trained models with TTA.
    """
    ensemble_preds = []
    models_found = 0

    print(f"Starting Ensemble Prediction using models from {model_dir}...")

    for fold in range(n_splits):
        fold_dir = os.path.join(model_dir, f"fold_{fold}")
        model_path = os.path.join(fold_dir, "model_best.pth")

        if not os.path.exists(model_path):
            print(f"Warning: Model for fold {fold} not found. Skipping.")
            continue

        # Load Model
        model = MicroResNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        # Predict
        preds = predict_with_tta(model, loader, device)
        ensemble_preds.append(preds)
        models_found += 1

    if models_found == 0:
        raise RuntimeError("No models found for ensemble prediction.")

    # Average predictions
    avg_preds = np.mean(ensemble_preds, axis=0)
    return avg_preds
