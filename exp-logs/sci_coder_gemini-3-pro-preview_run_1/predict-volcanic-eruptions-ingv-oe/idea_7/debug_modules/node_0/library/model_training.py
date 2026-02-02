import os
import gc
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from library.config import (
    SEED,
    LGB_PARAMS,
    LGB_ROUNDS,
    LGB_EARLY_STOPPING_ROUNDS,
    CNN_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
    NUM_WORKERS,
    NUM_FOLDS,
    RIDGE_ALPHA,
    CACHE_DIR,
    PATIENCE,
)
from library.utils import seed_everything
from library.data_loader import get_data_loaders
from library.cnn_architecture import VolcanoEfficientNet

# Ensure reproducible behavior
seed_everything(SEED)


def prepare_data_for_cv(
    train_meta, val_meta, train_feats, val_feats, train_vision, val_vision
):
    """
    Combines pre-split train/val data into a single dataset for 5-Fold CV.
    Ensures alignment between tabular data, vision data, and targets based on segment_id.
    """
    # 1. Combine Metadata
    full_meta = pd.concat([train_meta, val_meta], axis=0, ignore_index=True)

    # 2. Combine Tabular Features
    full_feats = pd.concat([train_feats, val_feats], axis=0, ignore_index=True)

    # 3. Combine Vision Data
    # train_vision is (X, y, ids)
    X_vision = np.concatenate([train_vision[0], val_vision[0]], axis=0)
    y_vision = np.concatenate([train_vision[1], val_vision[1]], axis=0)
    ids_vision = np.concatenate([train_vision[2], val_vision[2]], axis=0)

    # 4. Align Everything by segment_id
    # We sort everything by segment_id to ensure indices match across modalities

    # Sort Metadata
    full_meta = full_meta.sort_values("segment_id").reset_index(drop=True)
    target_ids = full_meta["segment_id"].values
    y_target = full_meta["time_to_eruption"].values

    # Sort Tabular
    full_feats = full_feats.sort_values("segment_id").reset_index(drop=True)

    # Sort Vision
    # Get sort indices for vision IDs
    sort_idx = np.argsort(ids_vision)
    X_vision_sorted = X_vision[sort_idx]
    y_vision_sorted = y_vision[sort_idx]  # Should match y_target
    ids_vision_sorted = ids_vision[sort_idx]

    # Verify alignment
    assert np.array_equal(
        target_ids, full_feats["segment_id"].values
    ), "Tabular IDs do not match Metadata IDs"
    assert np.array_equal(
        target_ids, ids_vision_sorted
    ), "Vision IDs do not match Metadata IDs"
    assert np.allclose(
        y_target, y_vision_sorted
    ), "Vision targets do not match Metadata targets"

    print(f"Combined Data prepared: {len(full_meta)} samples.")

    return full_feats, X_vision_sorted, y_target


def train_lgbm_fold(X_train, y_train, X_val, y_val, fold_idx):
    """
    Trains a LightGBM model for a single fold.
    """
    # Create datasets
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    callbacks = [
        lgb.early_stopping(stopping_rounds=LGB_EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(period=0),  # Silent
    ]

    model = lgb.train(
        LGB_PARAMS,
        lgb_train,
        num_boost_round=LGB_ROUNDS,
        valid_sets=[lgb_train, lgb_val],
        callbacks=callbacks,
    )

    # Predict
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)
    score = mean_absolute_error(y_val, val_preds)
    print(f"[LGBM Fold {fold_idx}] MAE: {score:.10f}")

    # Save model
    model_path = os.path.join(CACHE_DIR, f"lgb_model_fold_{fold_idx}.txt")
    model.save_model(model_path)

    return model, val_preds


def train_cnn_fold(X_train, y_train, X_val, y_val, fold_idx):
    """
    Trains the VolcanoEfficientNet for a single fold.
    """
    # Create DataLoaders
    train_loader, val_loader = get_data_loaders(
        X_train, y_train, X_val, y_val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # Initialize Model
    model = VolcanoEfficientNet(pretrained=True).to(DEVICE)

    # Loss: L1Loss on Log-Scaled Targets (Dataset returns log1p targets)
    criterion = nn.L1Loss()

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=CNN_EPOCHS, eta_min=1e-6)

    best_mae = float("inf")
    best_model_path = os.path.join(CACHE_DIR, f"cnn_fold_{fold_idx}.pth")

    # Training Loop
    for epoch in range(CNN_EPOCHS):
        model.train()
        train_loss = 0.0

        for imgs, targets in train_loader:
            imgs, targets = imgs.to(DEVICE), targets.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds_log = []
        val_targets_log = []

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(DEVICE), targets.to(DEVICE).unsqueeze(1)
                outputs = model(imgs)

                loss = criterion(outputs, targets)
                val_loss += loss.item() * imgs.size(0)

                val_preds_log.append(outputs.cpu().numpy())
                val_targets_log.append(targets.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        scheduler.step()

        # Calculate MAE in original scale
        val_preds_log = np.concatenate(val_preds_log)
        val_targets_log = np.concatenate(val_targets_log)

        val_preds_orig = np.expm1(val_preds_log)
        val_targets_orig = np.expm1(val_targets_log)

        current_mae = mean_absolute_error(val_targets_orig, val_preds_orig)

        if current_mae < best_mae:
            best_mae = current_mae
            torch.save(model.state_dict(), best_model_path)

    # Load best model for final prediction
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    model.eval()

    final_preds_log = []
    with torch.no_grad():
        for imgs, _ in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            final_preds_log.append(outputs.cpu().numpy())

    final_preds_log = np.concatenate(final_preds_log)
    final_preds_orig = np.expm1(final_preds_log).flatten()

    print(f"[CNN Fold {fold_idx}] Best MAE: {best_mae:.10f}")

    # Clean up
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return best_model_path, final_preds_orig


def train_meta_learner(oof_df):
    """
    Trains a Ridge Regression meta-learner on OOF predictions.
    """
    X_meta = oof_df[["pred_lgb", "pred_cnn"]].values
    y_meta = oof_df["target"].values

    meta_model = Ridge(alpha=RIDGE_ALPHA, random_state=SEED)
    meta_model.fit(X_meta, y_meta)

    # Evaluate
    preds = meta_model.predict(X_meta)
    mae = mean_absolute_error(y_meta, preds)
    print(f"[Meta-Learner] OOF MAE: {mae:.10f}")
    print(
        f"[Meta-Learner] Coefficients: LGB={meta_model.coef_[0]:.4f}, CNN={meta_model.coef_[1]:.4f}"
    )

    return meta_model


def run_cross_validation(full_tabular, full_vision_X, full_targets):
    """
    Orchestrates the 5-Fold Cross Validation.
    """
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    oof_preds_lgb = np.zeros(len(full_targets))
    oof_preds_cnn = np.zeros(len(full_targets))

    lgb_models = []
    cnn_model_paths = []

    # Drop non-feature columns for LGBM
    drop_cols = ["segment_id", "file_path", "time_to_eruption", "virtual_source"]
    feature_cols = [c for c in full_tabular.columns if c not in drop_cols]

    print(f"Starting {NUM_FOLDS}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_tabular, full_targets)):
        print(f"\n--- Fold {fold} ---")

        # Split Data
        # Tabular
        X_tab_train = full_tabular.iloc[train_idx][feature_cols]
        y_tab_train = full_targets[train_idx]
        X_tab_val = full_tabular.iloc[val_idx][feature_cols]
        y_tab_val = full_targets[val_idx]

        # Vision
        X_vis_train = full_vision_X[train_idx]
        y_vis_train = full_targets[train_idx]
        X_vis_val = full_vision_X[val_idx]
        y_vis_val = full_targets[val_idx]

        # 1. Train LightGBM
        lgb_model, lgb_pred = train_lgbm_fold(
            X_tab_train, y_tab_train, X_tab_val, y_tab_val, fold
        )
        lgb_models.append(lgb_model)
        oof_preds_lgb[val_idx] = lgb_pred

        # 2. Train CNN
        cnn_path, cnn_pred = train_cnn_fold(
            X_vis_train, y_vis_train, X_vis_val, y_vis_val, fold
        )
        cnn_model_paths.append(cnn_path)
        oof_preds_cnn[val_idx] = cnn_pred

    # Create OOF DataFrame
    oof_df = pd.DataFrame(
        {"target": full_targets, "pred_lgb": oof_preds_lgb, "pred_cnn": oof_preds_cnn}
    )

    # 3. Train Meta-Learner
    meta_model = train_meta_learner(oof_df)

    return lgb_models, cnn_model_paths, meta_model, oof_df
