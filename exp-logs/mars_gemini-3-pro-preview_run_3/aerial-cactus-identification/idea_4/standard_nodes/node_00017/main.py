import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Import provided library functions
from library.utils import seed_everything, save_state_dict
from library.models import ModifiedWideSEResNet, ModifiedDenseNet
from library.data import load_and_cache_data, get_transforms, CactusDataset
from library.trainer import train_one_epoch, validate, predict_tta

# --- Configuration ---
SEED = 42
BATCH_SIZE = 128
EPOCHS = 35
LR = 1e-3
N_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working"
SUBMISSION_DIR = "./submission"

# Ensure directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


def main():
    # 1. Reproducibility
    seed_everything(SEED)

    print(f"Using device: {DEVICE}")

    # 2. Load Data
    # Load Train Data (for Cross-Validation)
    print("Loading Training Data...")
    train_imgs, train_ids, train_lbls = load_and_cache_data(
        os.path.join(METADATA_DIR, "train_metadata.csv"),
        INPUT_DIR,
        "train_cv",
        load_cached_data=True,
    )

    # Load Hold-out Validation Data (for Final Metric)
    print("Loading Hold-out Validation Data...")
    val_imgs, val_ids, val_lbls = load_and_cache_data(
        os.path.join(METADATA_DIR, "val_metadata.csv"),
        INPUT_DIR,
        "holdout_val",
        load_cached_data=True,
    )

    # Load Test Data
    print("Loading Test Data...")
    test_imgs, test_ids, _ = load_and_cache_data(
        os.path.join(METADATA_DIR, "test_metadata.csv"),
        INPUT_DIR,
        "test",
        load_cached_data=True,
    )

    # 3. Prepare Storage for Stacking
    # Map IDs to indices for correct OOF placement
    id_to_idx = {id_: i for i, id_ in enumerate(train_ids)}

    # OOF Predictions: (N_train, 2_models)
    oof_preds = np.zeros((len(train_imgs), 2))

    # Hold-out Predictions: (N_val, N_folds, 2_models)
    val_preds_storage = np.zeros((len(val_imgs), N_FOLDS, 2))

    # Test Predictions: (N_test, N_folds, 2_models)
    test_preds_storage = np.zeros((len(test_imgs), N_FOLDS, 2))

    # Transforms
    train_transform = get_transforms("train")
    valid_transform = get_transforms("valid")  # Used for validation and test

    # 4. Stratified K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (train_idx, valid_idx) in enumerate(skf.split(train_imgs, train_lbls)):
        print(f"\n{'='*20} Fold {fold+1}/{N_FOLDS} {'='*20}")

        # Prepare Fold Data
        X_train, y_train = train_imgs[train_idx], train_lbls[train_idx]
        X_valid, y_valid = train_imgs[valid_idx], train_lbls[valid_idx]

        # Create Datasets
        train_ds = CactusDataset(
            X_train, y_train, train_ids[train_idx], transform=train_transform
        )
        valid_ds = CactusDataset(
            X_valid, y_valid, train_ids[valid_idx], transform=valid_transform
        )

        # Create Loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # --- Architecture 1: Modified Wide SE-ResNet ---
        print(f"[Fold {fold+1}] Training Modified Wide SE-ResNet...")
        model_res = ModifiedWideSEResNet(num_classes=1).to(DEVICE)
        optimizer = optim.AdamW(model_res.parameters(), lr=LR, weight_decay=1e-2)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_auc_res = 0
        res_path = os.path.join(WORKING_DIR, f"resnet_fold{fold}.pth")

        for epoch in range(EPOCHS):
            _ = train_one_epoch(
                model_res, train_loader, criterion, optimizer, DEVICE, alpha=1.0
            )
            _, val_auc = validate(model_res, valid_loader, criterion, DEVICE)
            scheduler.step()

            if val_auc > best_auc_res:
                best_auc_res = val_auc
                save_state_dict(model_res, res_path)

        # Load best model for inference
        model_res.load_state_dict(torch.load(res_path))

        # Inference: OOF (Fold Validation)
        oof_dict = predict_tta(model_res, valid_loader, DEVICE)
        for id_, prob in oof_dict.items():
            oof_preds[id_to_idx[id_], 0] = prob

        # Inference: Hold-out Validation
        holdout_ds = CactusDataset(
            val_imgs, val_lbls, val_ids, transform=valid_transform
        )
        holdout_loader = DataLoader(
            holdout_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        holdout_dict = predict_tta(model_res, holdout_loader, DEVICE)
        for i, id_ in enumerate(val_ids):
            val_preds_storage[i, fold, 0] = holdout_dict[id_]

        # Inference: Test
        test_ds = CactusDataset(
            test_imgs, np.zeros(len(test_imgs)), test_ids, transform=valid_transform
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        test_dict = predict_tta(model_res, test_loader, DEVICE)
        for i, id_ in enumerate(test_ids):
            test_preds_storage[i, fold, 0] = test_dict[id_]

        # --- Architecture 2: Modified DenseNet ---
        print(f"[Fold {fold+1}] Training Modified DenseNet...")
        model_dense = ModifiedDenseNet(num_classes=1).to(DEVICE)
        optimizer = optim.AdamW(model_dense.parameters(), lr=LR, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_auc_dense = 0
        dense_path = os.path.join(WORKING_DIR, f"densenet_fold{fold}.pth")

        for epoch in range(EPOCHS):
            _ = train_one_epoch(
                model_dense, train_loader, criterion, optimizer, DEVICE, alpha=1.0
            )
            _, val_auc = validate(model_dense, valid_loader, criterion, DEVICE)
            scheduler.step()

            if val_auc > best_auc_dense:
                best_auc_dense = val_auc
                save_state_dict(model_dense, dense_path)

        # Load best model for inference
        model_dense.load_state_dict(torch.load(dense_path))

        # Inference: OOF
        oof_dict = predict_tta(model_dense, valid_loader, DEVICE)
        for id_, prob in oof_dict.items():
            oof_preds[id_to_idx[id_], 1] = prob

        # Inference: Hold-out
        holdout_dict = predict_tta(model_dense, holdout_loader, DEVICE)
        for i, id_ in enumerate(val_ids):
            val_preds_storage[i, fold, 1] = holdout_dict[id_]

        # Inference: Test
        test_dict = predict_tta(model_dense, test_loader, DEVICE)
        for i, id_ in enumerate(test_ids):
            test_preds_storage[i, fold, 1] = test_dict[id_]

    # 5. Train Meta-Learner (Stacking)
    print("\nTraining Meta-Learner (Logistic Regression)...")
    meta_model = LogisticRegression()
    # OOF preds are features, train labels are targets
    meta_model.fit(oof_preds, train_lbls)

    # 6. Evaluate on Hold-out Set
    # Average predictions across folds for each architecture
    val_preds_avg = val_preds_storage.mean(axis=1)  # Shape: (N_val, 2)

    # Meta-learner prediction
    final_val_probs = meta_model.predict_proba(val_preds_avg)[:, 1]

    # Calculate Metric
    final_metric = roc_auc_score(val_lbls, final_val_probs)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    # Calculate absolute error
    errors = np.abs(final_val_probs - val_lbls)

    # Compute meta-features for hold-out images
    # val_imgs is (N, 32, 32, 3) RGB
    brightness = val_imgs.mean(axis=(1, 2, 3))
    contrast = val_imgs.std(axis=(1, 2, 3))

    corr_brightness = np.corrcoef(errors, brightness)[0, 1]
    corr_contrast = np.corrcoef(errors, contrast)[0, 1]

    print(f"Correlation between Error and Brightness: {corr_brightness:.10f}")
    print(f"Correlation between Error and Contrast: {corr_contrast:.10f}")

    # 8. Generate Submission
    THRESHOLD = 0.9999953560392056

    if final_metric > THRESHOLD:
        print("\nValidation metric met threshold. Generating submission...")

        # Average test predictions across folds
        test_preds_avg = test_preds_storage.mean(axis=1)  # Shape: (N_test, 2)

        # Meta-learner prediction
        final_test_probs = meta_model.predict_proba(test_preds_avg)[:, 1]

        submission_df = pd.DataFrame({"id": test_ids, "has_cactus": final_test_probs})

        sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        submission_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(
            f"\nValidation metric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
