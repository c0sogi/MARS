import os
import shutil
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from library.config import Config
from library.utils import (
    seed_everything,
    metric_mae,
    save_checkpoint,
    load_checkpoint,
    inverse_transform_target,
    transform_target,
    save_npy,
)
from library.data_loader import load_tabular_dataset, get_data_loaders
from library.models_vision import ScalarConditionedEfficientNet
from library.models_tabular import LightGBMTrainer

# Ensure reproducibility
seed_everything(Config.SEED)


def prepare_unified_spectrograms():
    """
    Consolidates train and val spectrograms into a single directory
    to support K-Fold Cross Validation across the original split boundaries.
    """
    unified_dir = os.path.join(Config.IDEA_DIR, "spectrograms_all")
    os.makedirs(unified_dir, exist_ok=True)

    # Source directories
    dirs_to_copy = [Config.CACHE_SPECTROGRAMS_TRAIN, Config.CACHE_SPECTROGRAMS_VAL]

    # We use symlinks to save space and time, or copy if symlinks fail
    # Given the environment, copy is safer if symlinks are restricted, but let's try copy for robustness
    # Optimization: Check if dir is already populated to avoid re-copying
    existing_files = len(glob.glob(os.path.join(unified_dir, "*.npy")))
    # Approx check: if we have enough files, skip.
    # Exact count might vary, but > 3000 implies it's likely done.
    if existing_files > 3000:
        return unified_dir

    print("Preparing unified spectrogram directory for K-Fold CV...")
    for src_dir in dirs_to_copy:
        if os.path.exists(src_dir):
            for file_name in os.listdir(src_dir):
                if file_name.endswith(".npy"):
                    src_file = os.path.join(src_dir, file_name)
                    dst_file = os.path.join(unified_dir, file_name)
                    if not os.path.exists(dst_file):
                        shutil.copy2(src_file, dst_file)

    return unified_dir


def train_vision_fold(train_loader, val_loader, fold_id, device):
    """
    Trains the ScalarConditionedEfficientNet for a single fold.
    """
    print(f"\n--- Vision Branch: Training Fold {fold_id} ---")

    model = ScalarConditionedEfficientNet(pretrained=True)
    model.to(device)

    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    criterion = nn.L1Loss()

    best_mae = float("inf")
    best_epoch = 0
    patience_counter = 0

    # Placeholder for OOF predictions
    # We can't easily align loader order with dataframe index unless we track indices.
    # However, our data loaders are sequential for Val (shuffle=False).
    # We will collect predictions and return them.

    for epoch in range(1, Config.NUM_EPOCHS + 1):
        # Training
        model.train()
        train_loss_sum = 0

        for batch_idx, (specs, scalars, targets) in enumerate(train_loader):
            specs, scalars, targets = (
                specs.to(device),
                scalars.to(device),
                targets.to(device),
            )

            optimizer.zero_grad()
            outputs = model(specs, scalars)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

        avg_train_loss = train_loss_sum / len(train_loader)
        scheduler.step()

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for specs, scalars, targets in val_loader:
                specs, scalars, targets = (
                    specs.to(device),
                    scalars.to(device),
                    targets.to(device),
                )
                outputs = model(specs, scalars)

                # Inverse transform for metric calculation
                preds_inv = inverse_transform_target(outputs.cpu().numpy())
                targets_inv = inverse_transform_target(targets.cpu().numpy())

                val_preds.append(preds_inv)
                val_targets.append(targets_inv)

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)

        current_mae = metric_mae(val_targets, val_preds)

        print(
            f"Epoch {epoch}/{Config.NUM_EPOCHS} | Train Loss (Log): {avg_train_loss:.6f} | Val MAE: {current_mae:.10f}"
        )

        if current_mae < best_mae:
            best_mae = current_mae
            best_epoch = epoch
            patience_counter = 0

            # Save Checkpoint
            ckpt_path = os.path.join(Config.IDEA_DIR, f"cnn_fold_{fold_id}.pth")
            save_checkpoint(model, optimizer, epoch, current_mae, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    print(f"Fold {fold_id} Best Vision MAE: {best_mae:.10f} at Epoch {best_epoch}")

    # Load best model for final OOF generation
    ckpt_path = os.path.join(Config.IDEA_DIR, f"cnn_fold_{fold_id}.pth")
    _, _ = load_checkpoint(ckpt_path, model, device=device)
    model.eval()

    final_oof_preds = []
    with torch.no_grad():
        for specs, scalars, targets in val_loader:
            specs, scalars = specs.to(device), scalars.to(device)
            outputs = model(specs, scalars)
            preds_inv = inverse_transform_target(outputs.cpu().numpy())
            final_oof_preds.append(preds_inv)

    return model, np.concatenate(final_oof_preds).flatten()


def predict_vision(model, loader, device):
    """
    Generates predictions using the Vision model.
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for specs, scalars in loader:
            specs, scalars = specs.to(device), scalars.to(device)
            outputs = model(specs, scalars)
            preds_inv = inverse_transform_target(outputs.cpu().numpy())
            preds.append(preds_inv)
    return np.concatenate(preds).flatten()


def train_meta_learner(X, y):
    """
    Trains a Ridge Regression meta-learner.
    """
    print("\n--- Training Meta-Learner ---")
    meta_model = Ridge(alpha=Config.META_ALPHA, fit_intercept=True)
    meta_model.fit(X, y)

    print(
        f"Meta-Learner Coefficients: Tabular={meta_model.coef_[0]:.4f}, Vision={meta_model.coef_[1]:.4f}"
    )
    print(f"Meta-Learner Intercept: {meta_model.intercept_:.4f}")

    return meta_model


def run_training():
    # 1. Load Data
    # This triggers feature engineering if cache is missing
    df_train_raw, df_val_raw, df_test = load_tabular_dataset(load_cached=True)

    # Combine Train and Val for 5-Fold CV
    df_full = pd.concat([df_train_raw, df_val_raw], ignore_index=True)

    # Prepare Unified Spectrogram Directory for Vision Model
    unified_spec_dir = prepare_unified_spectrograms()

    # Note: We must temporarily override the cache paths in Config or pass the new dir explicitly.
    # The VolcanoDataset takes `spectrogram_dir` as init arg, so we are good.

    # 2. Setup Cross-Validation
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    # Storage for OOF and Test Predictions
    oof_preds_tabular = np.zeros(len(df_full))
    oof_preds_vision = np.zeros(len(df_full))
    y_true = df_full["time_to_eruption"].values

    test_preds_tabular = np.zeros((len(df_test), Config.N_FOLDS))
    test_preds_vision = np.zeros((len(df_test), Config.N_FOLDS))

    # 3. Iterate Folds
    for fold_id, (train_idx, val_idx) in enumerate(kf.split(df_full)):
        print(f"\n========================= FOLD {fold_id} =========================")

        # Split Data
        df_fold_train = df_full.iloc[train_idx].copy()
        df_fold_val = df_full.iloc[val_idx].copy()

        # --- Branch A: Tabular (LightGBM) ---
        lgbm_trainer = LightGBMTrainer()
        lgbm_model = lgbm_trainer.train(df_fold_train, df_fold_val, fold_id=fold_id)

        # Predict OOF
        oof_preds_tabular[val_idx] = lgbm_trainer.predict(df_fold_val, model=lgbm_model)
        # Predict Test
        test_preds_tabular[:, fold_id] = lgbm_trainer.predict(df_test, model=lgbm_model)

        # --- Branch B: Vision (EfficientNet) ---
        # Create DataLoaders using the unified directory
        # We need to manually instantiate datasets because get_data_loaders uses Config paths
        from library.data_loader import VolcanoDataset, DataLoader

        train_ds = VolcanoDataset(df_fold_train, unified_spec_dir, mode="train")
        val_ds = VolcanoDataset(df_fold_val, unified_spec_dir, mode="val")
        test_ds = VolcanoDataset(df_test, Config.CACHE_SPECTROGRAMS_TEST, mode="test")

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
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        vision_model, val_preds_vision = train_vision_fold(
            train_loader, val_loader, fold_id, Config.DEVICE
        )

        # Store OOF
        # Note: val_preds_vision is aligned with val_loader iteration order.
        # Since val_loader is shuffle=False and built from df_fold_val, order matches df_fold_val.
        oof_preds_vision[val_idx] = val_preds_vision.flatten()

        # Predict Test
        test_preds_vision[:, fold_id] = predict_vision(
            vision_model, test_loader, Config.DEVICE
        ).flatten()

    # 4. Evaluate Ensemble
    print("\n========================= ENSEMBLE EVALUATION =========================")
    mae_tabular = metric_mae(y_true, oof_preds_tabular)
    mae_vision = metric_mae(y_true, oof_preds_vision)

    print(f"Overall OOF MAE - Tabular: {mae_tabular:.10f}")
    print(f"Overall OOF MAE - Vision:  {mae_vision:.10f}")

    # Train Meta-Learner
    # Stack features: (N_samples, 2)
    X_meta = np.column_stack((oof_preds_tabular, oof_preds_vision))
    meta_model = train_meta_learner(X_meta, y_true)

    # Evaluate Meta-Learner on OOF
    oof_preds_meta = meta_model.predict(X_meta)
    mae_meta = metric_mae(y_true, oof_preds_meta)
    print(f"Overall OOF MAE - Ensemble: {mae_meta:.10f}")

    # 5. Generate Submission
    print("\nGenerating Submission...")

    # Average predictions across folds
    avg_test_tabular = np.mean(test_preds_tabular, axis=1)
    avg_test_vision = np.mean(test_preds_vision, axis=1)

    # Apply Meta-Learner
    X_test_meta = np.column_stack((avg_test_tabular, avg_test_vision))
    final_predictions = meta_model.predict(X_test_meta)

    # Ensure non-negative predictions (physics constraint)
    final_predictions = np.maximum(final_predictions, 0)

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"segment_id": df_test["segment_id"], "time_to_eruption": final_predictions}
    )

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(submission_df.head())


if __name__ == "__main__":
    # This block is here for local testing only, the orchestrator will import the module.
    pass
