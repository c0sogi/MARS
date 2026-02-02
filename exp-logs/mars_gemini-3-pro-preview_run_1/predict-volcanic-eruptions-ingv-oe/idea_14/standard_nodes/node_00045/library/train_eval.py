import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from library.config import Config
from library.utils import seed_everything, log1p_target, expm1_target
from library.feature_engineering import FeatureEngineer
from library.dataset import SeismicDataset
from library.models import ScalarFusedEfficientNet, LightGBMWrapper


def train_lgbm_fold(X_train, y_train, X_val, y_val, fold_idx):
    """
    Trains a LightGBM model for a single fold and returns the model and validation predictions.
    """
    print(f"\n--- Training LightGBM Fold {fold_idx} ---")
    model = LightGBMWrapper()
    model.fit(X_train, y_train, X_val, y_val)

    # Save model for persistence
    model_path = os.path.join(Config.WORKING_DIR, f"lgb_model_fold_{fold_idx}.txt")
    model.save_model(model_path)

    # Generate validation predictions
    val_preds = model.predict(X_val)
    return model, val_preds


def train_cnn_fold(train_meta, val_meta, fold_idx):
    """
    Trains the ScalarFusedEfficientNet for a single fold.
    Handles data loading, training loop, validation monitoring, and early stopping.
    Returns the OOF predictions for the validation set.
    """
    print(f"\n--- Training CNN Fold {fold_idx} ---")
    device = Config.DEVICE

    # Initialize Datasets
    # Note: We use SPECTROGRAM_TRAIN_DIR for both because we are cross-validating the training set
    train_dataset = SeismicDataset(
        train_meta, Config.SPECTROGRAM_TRAIN_DIR, mode="train"
    )
    val_dataset = SeismicDataset(val_meta, Config.SPECTROGRAM_TRAIN_DIR, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model, Optimizer, Scheduler, Loss
    model = ScalarFusedEfficientNet().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    criterion = nn.L1Loss()

    best_mae = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, f"cnn_fold_{fold_idx}.pth")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0

        for images, scalars, targets in train_loader:
            images = images.to(device)
            scalars = scalars.to(device)
            # Targets are log-scaled by Dataset if mode='train'
            targets = targets.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images, scalars)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)

        # Validation Loop
        model.eval()
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for images, scalars, targets in val_loader:
                images = images.to(device)
                scalars = scalars.to(device)

                # Convert targets back to original scale for MAE calculation
                targets_log = targets.numpy()
                targets_raw = expm1_target(targets_log)

                outputs = model(images, scalars)
                preds_log = outputs.cpu().numpy().flatten()
                preds_raw = expm1_target(preds_log)

                val_preds_list.extend(preds_raw)
                val_targets_list.extend(targets_raw)

        val_mae = mean_absolute_error(val_targets_list, val_preds_list)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss (Log MAE): {train_loss:.5f} | Val MAE: {val_mae:.5f}"
        )

        scheduler.step()

        # Early Stopping & Checkpointing
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

    # Load best model to generate final OOF predictions
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    oof_preds = []
    with torch.no_grad():
        for images, scalars, _ in val_loader:
            images = images.to(device)
            scalars = scalars.to(device)
            outputs = model(images, scalars)
            preds_log = outputs.cpu().numpy().flatten()
            preds_raw = expm1_target(preds_log)
            oof_preds.extend(preds_raw)

    return np.array(oof_preds)


def predict_cnn_test(test_meta, fold_idx):
    """
    Generates predictions for the test set using a specific fold's trained CNN.
    """
    device = Config.DEVICE
    model_path = os.path.join(Config.WORKING_DIR, f"cnn_fold_{fold_idx}.pth")

    test_dataset = SeismicDataset(test_meta, Config.SPECTROGRAM_TEST_DIR, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    model = ScalarFusedEfficientNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds_list = []
    with torch.no_grad():
        for images, scalars, _ in test_loader:
            images = images.to(device)
            scalars = scalars.to(device)
            outputs = model(images, scalars)
            preds_log = outputs.cpu().numpy().flatten()
            # Inverse transform to get time_to_eruption
            preds_raw = expm1_target(preds_log)
            preds_list.extend(preds_raw)

    return np.array(preds_list)


def run_training():
    """
    Main execution pipeline:
    1. Feature Engineering (Tabular + Vision)
    2. 5-Fold Cross-Validation (LGBM + CNN)
    3. Meta-Learner Stacking
    4. Submission Generation
    """
    seed_everything(Config.SEED)

    # ==========================================
    # 1. Feature Engineering
    # ==========================================
    print("Step 1: Feature Engineering")
    fe = FeatureEngineer()

    # Generate/Load Tabular Features
    df_train_features = fe.process_tabular("train", load_cached_data=True)
    df_test_features = fe.process_tabular("test", load_cached_data=True)

    # Generate/Load Vision Features
    fe.process_vision("train", load_cached_data=True)
    fe.process_vision("test", load_cached_data=True)

    # ==========================================
    # 2. Data Preparation
    # ==========================================
    print("Step 2: Preparing Data for Cross-Validation")
    df_train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Handle Debug Mode
    if Config.DEBUG:
        print(f"DEBUG MODE: Truncating data to {Config.DEBUG_SAMPLE_SIZE} samples.")
        df_train_meta = df_train_meta.head(Config.DEBUG_SAMPLE_SIZE)

    # Align Tabular Features with Metadata
    # Ensure rows match exactly by joining on segment_id
    df_train_features = (
        df_train_features.set_index("segment_id")
        .reindex(df_train_meta["segment_id"])
        .reset_index()
    )
    df_test_features = (
        df_test_features.set_index("segment_id")
        .reindex(df_test_meta["segment_id"])
        .reset_index()
    )

    # Prepare Tabular Inputs
    drop_cols = ["segment_id", "time_to_eruption", "file_path"]
    feature_cols = [c for c in df_train_features.columns if c not in drop_cols]

    X_full = df_train_features[feature_cols]
    y_full = df_train_features["time_to_eruption"]
    X_test = df_test_features[feature_cols]

    # Initialize Arrays for Stacking
    oof_lgbm = np.zeros(len(df_train_meta))
    test_preds_lgbm = np.zeros(len(df_test_meta))

    oof_cnn = np.zeros(len(df_train_meta))
    test_preds_cnn = np.zeros(len(df_test_meta))

    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    # ==========================================
    # 3. Cross-Validation Loop
    # ==========================================
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_full, y_full)):
        print(f"\n==================== FOLD {fold} ====================")

        # --- Branch A: LightGBM ---
        X_tr_lgb, y_tr_lgb = X_full.iloc[train_idx], y_full.iloc[train_idx]
        X_val_lgb, y_val_lgb = X_full.iloc[val_idx], y_full.iloc[val_idx]

        lgbm_model, val_preds_lgbm = train_lgbm_fold(
            X_tr_lgb, y_tr_lgb, X_val_lgb, y_val_lgb, fold
        )
        oof_lgbm[val_idx] = val_preds_lgbm
        test_preds_lgbm += lgbm_model.predict(X_test) / Config.N_FOLDS

        # --- Branch B: CNN ---
        # Subset metadata for Dataset creation
        train_meta_fold = df_train_meta.iloc[train_idx].copy()
        val_meta_fold = df_train_meta.iloc[val_idx].copy()

        val_preds_cnn = train_cnn_fold(train_meta_fold, val_meta_fold, fold)
        oof_cnn[val_idx] = val_preds_cnn
        test_preds_cnn += predict_cnn_test(df_test_meta, fold) / Config.N_FOLDS

    # ==========================================
    # 4. Evaluation & Stacking
    # ==========================================
    mae_lgbm = mean_absolute_error(y_full, oof_lgbm)
    mae_cnn = mean_absolute_error(y_full, oof_cnn)
    print(f"\n--- Ensemble Performance ---")
    print(f"LGBM OOF MAE: {mae_lgbm:.5f}")
    print(f"CNN OOF MAE:  {mae_cnn:.5f}")

    print("\nStep 5: Meta-Learner Training")
    # Stack OOF predictions
    X_meta = np.column_stack([oof_lgbm, oof_cnn])
    X_test_meta = np.column_stack([test_preds_lgbm, test_preds_cnn])

    # Train Ridge Regression
    meta_model = Ridge(alpha=Config.META_MODEL_ALPHA, random_state=Config.SEED)
    meta_model.fit(X_meta, y_full)

    print(
        f"Meta-Learner Coefficients: LGBM={meta_model.coef_[0]:.4f}, CNN={meta_model.coef_[1]:.4f}"
    )
    print(f"Meta-Learner Intercept:    {meta_model.intercept_:.4f}")

    # Evaluate Stacked OOF
    final_oof = meta_model.predict(X_meta)
    final_oof = np.maximum(0, final_oof)  # Clip negative predictions
    mae_meta = mean_absolute_error(y_full, final_oof)
    print(f"Stacked OOF MAE: {mae_meta:.5f}")

    # ==========================================
    # 5. Submission
    # ==========================================
    print("\nStep 6: Generating Submission")
    final_test_preds = meta_model.predict(X_test_meta)
    final_test_preds = np.maximum(0, final_test_preds)

    submission_df = pd.DataFrame(
        {
            "segment_id": df_test_meta["segment_id"],
            "time_to_eruption": final_test_preds,
        }
    )

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
