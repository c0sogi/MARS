import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Import from provided library
from library.config import Config
from library.utils import seed_everything
from library.dataset import load_or_generate_data, BraTSDataset, get_transforms
from library.model import LMSPEfficientNet
from library.trainer import train_one_epoch


def inference(model, loader, device):
    """
    Runs inference on a DataLoader using the provided model.
    Returns a numpy array of probabilities.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            # Unpack batch: (images, targets) or (images, ids)
            images = batch[0]
            images = images.to(device)

            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds.extend(probs)
    return np.array(preds)


def run():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Running on device: {device}")

    # 2. Data Loading
    # Load Metadata DataFrames
    train_df_full = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df_holdout = pd.read_csv(Config.VAL_METADATA_PATH)
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # Load/Process Data (Cached)
    # We load the full training set which we will split for CV
    print("Loading Training Data...")
    X_train_full, ids_train_full, y_train_full = load_or_generate_data(
        train_df_full, "train", load_cached_data=True
    )

    # Load Holdout Validation data
    print("Loading Validation Data...")
    X_val_holdout, ids_val_holdout, y_val_holdout = load_or_generate_data(
        val_df_holdout, "val", load_cached_data=True
    )

    # Load Test data
    print("Loading Test Data...")
    X_test, ids_test, _ = load_or_generate_data(test_df, "test", load_cached_data=True)

    # 3. Stratified K-Fold Cross Validation
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    fold_models = []

    print(f"\nStarting {Config.N_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full)):
        print(f"\n=== Fold {fold+1}/{Config.N_FOLDS} ===")

        # Split Data for this fold
        X_tr, y_tr = X_train_full[train_idx], y_train_full[train_idx]
        X_va, y_va = X_train_full[val_idx], y_train_full[val_idx]
        ids_tr, ids_va = ids_train_full[train_idx], ids_train_full[val_idx]

        # Create Datasets and Loaders
        train_ds = BraTSDataset(X_tr, ids_tr, y_tr, transform=get_transforms("train"))
        val_ds = BraTSDataset(X_va, ids_va, y_va, transform=get_transforms("val"))

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Initialize Model
        model = LMSPEfficientNet()
        model.to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_fold_auc = 0.0
        best_model_state = None

        for epoch in range(Config.EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )

            # Validation on fold-val set
            model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for imgs, tgts in val_loader:
                    imgs = imgs.to(device)
                    outputs = model(imgs)
                    probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                    val_preds.extend(probs)
                    val_targets.extend(tgts.numpy())

            try:
                fold_auc = roc_auc_score(val_targets, val_preds)
            except ValueError:
                fold_auc = 0.5

            # Save best state
            if fold_auc > best_fold_auc:
                best_fold_auc = fold_auc
                best_model_state = model.state_dict()

        # Persist best model for this fold
        save_path = os.path.join(Config.WORKING_DIR, f"model_fold_{fold}.pth")
        if best_model_state is not None:
            torch.save(best_model_state, save_path)
            model.load_state_dict(best_model_state)
        else:
            torch.save(model.state_dict(), save_path)

        fold_models.append(model)
        print(f"Fold {fold+1} Best AUC: {best_fold_auc:.6f}")

    # 4. Hold-out Validation (Ensemble)
    print("\nEvaluating Ensemble on Hold-out Validation Set...")
    val_holdout_ds = BraTSDataset(
        X_val_holdout, ids_val_holdout, y_val_holdout, transform=get_transforms("val")
    )
    val_holdout_loader = DataLoader(
        val_holdout_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Average predictions from all fold models
    ensemble_preds = np.zeros(len(y_val_holdout))

    for i, model in enumerate(fold_models):
        preds = inference(model, val_holdout_loader, device)
        ensemble_preds += preds

    ensemble_preds /= Config.N_FOLDS

    final_auc = roc_auc_score(y_val_holdout, ensemble_preds)
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(y_val_holdout - ensemble_preds)

    # Calculate simple input features: Mean Intensity, Std Intensity per sample
    # X_val_holdout shape: (N, H, W, C)
    mean_intensities = np.mean(X_val_holdout, axis=(1, 2, 3))
    std_intensities = np.std(X_val_holdout, axis=(1, 2, 3))

    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "mean_intensity": mean_intensities,
            "std_intensity": std_intensities,
            "target": y_val_holdout,
        }
    )

    corr_mean = df_analysis["error"].corr(df_analysis["mean_intensity"])
    corr_std = df_analysis["error"].corr(df_analysis["std_intensity"])
    corr_target = df_analysis["error"].corr(df_analysis["target"])

    print(f"Correlation (Error vs Mean Intensity): {corr_mean:.4f}")
    print(f"Correlation (Error vs Std Intensity): {corr_std:.4f}")
    print(f"Correlation (Error vs Target Class): {corr_target:.4f}")

    # 6. Submission
    THRESHOLD = 0.6705454545454544
    if final_auc > THRESHOLD:
        print(
            f"\nMetric ({final_auc:.6f}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_ds = BraTSDataset(
            X_test, ids_test, labels=None, transform=get_transforms("test")
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_ensemble_preds = np.zeros(len(ids_test))

        for model in fold_models:
            preds = inference(model, test_loader, device)
            test_ensemble_preds += preds

        test_ensemble_preds /= Config.N_FOLDS

        submission_df = pd.DataFrame(
            {"BraTS21ID": ids_test, "MGMT_value": test_ensemble_preds}
        )

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_auc:.6f}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()
