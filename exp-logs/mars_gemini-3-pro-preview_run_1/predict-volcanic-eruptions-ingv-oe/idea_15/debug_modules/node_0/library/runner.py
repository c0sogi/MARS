import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

from library.config import Config
from library.utils import (
    seed_everything,
    mae_score,
    get_device,
    save_pickle,
    load_pickle,
)
from library.feature_engineering import FeatureEngineer
from library.dataset import VolcanoDataset
from library.model_vision import EfficientNetFiLM
from library.model_tabular import train_lgbm_fold, predict_lgbm


def prepare_data(load_cached_data: bool = True):
    """
    Orchestrates data processing using FeatureEngineer with caching.
    """
    fe = FeatureEngineer()

    # 1. Process Training Data
    print("Preparing Training Data...")
    train_meta_path = os.path.join(Config.METADATA_DIR, "train.csv")
    # Output dir: ./working/idea_15/train_data
    df_train = fe.process_dataset(
        train_meta_path, output_dir_name="train_data", load_cached_data=load_cached_data
    )
    train_spec_dir = os.path.join(Config.WORKING_DIR, "train_data", "spectrograms")

    # 2. Process Test Data
    print("Preparing Test Data...")
    test_meta_path = os.path.join(Config.METADATA_DIR, "test.csv")
    # Output dir: ./working/idea_15/test_data
    df_test = fe.process_dataset(
        test_meta_path, output_dir_name="test_data", load_cached_data=load_cached_data
    )
    test_spec_dir = os.path.join(Config.WORKING_DIR, "test_data", "spectrograms")

    return df_train, train_spec_dir, df_test, test_spec_dir


def run_tabular_cv(df: pd.DataFrame, n_folds: int = Config.N_FOLDS):
    """
    Executes 5-Fold CV for the Tabular Branch (LightGBM).
    """
    # Identify feature columns: all numeric columns except metadata and targets
    # We include scalar_ columns as they are valid statistical features
    exclude_cols = ["segment_id", "time_to_eruption", "file_path"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Ensure features are numeric
    feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

    oof_preds = np.zeros(len(df))
    models = []

    print(f"Starting Tabular CV with {len(feature_cols)} features...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"--- Tabular Fold {fold+1}/{n_folds} ---")
        train_sub = df.iloc[train_idx]
        val_sub = df.iloc[val_idx]

        model, val_pred = train_lgbm_fold(train_sub, val_sub, feature_cols)

        oof_preds[val_idx] = val_pred
        models.append(model)

    score = mae_score(df["time_to_eruption"].values, oof_preds)
    print(f"Tabular Branch Overall CV MAE: {score}")

    return oof_preds, models, feature_cols


def run_vision_cv(
    df: pd.DataFrame, spectrogram_dir: str, n_folds: int = Config.N_FOLDS
):
    """
    Executes 5-Fold CV for the Vision Branch (EfficientNetFiLM).
    """
    device = get_device()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)

    oof_preds = np.zeros(len(df))
    model_paths = []
    scalar_stats_list = []

    print(f"Starting Vision CV on device: {device}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"--- Vision Fold {fold+1}/{n_folds} ---")

        # Split Data
        train_sub = df.iloc[train_idx].reset_index(drop=True)
        val_sub = df.iloc[val_idx].reset_index(drop=True)

        # Create Datasets
        # Mode='train' computes scalar stats from train_sub
        train_ds = VolcanoDataset(train_sub, spectrogram_dir, mode="train")
        stats = train_ds.get_scalar_stats()
        scalar_stats_list.append(stats)

        # Mode='val' uses stats from train_sub to prevent leakage
        val_ds = VolcanoDataset(
            val_sub, spectrogram_dir, mode="val", scalar_stats=stats
        )

        # DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Initialize Model
        scalar_dim = len(train_ds.scalar_cols)
        model = EfficientNetFiLM(scalar_input_dim=scalar_dim).to(device)

        # Optimization
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
        criterion = nn.L1Loss()

        # Training Loop
        best_mae = float("inf")
        patience_counter = 0
        model_save_path = os.path.join(
            Config.WORKING_DIR, f"vision_model_fold_{fold}.pth"
        )

        for epoch in range(Config.EPOCHS):
            model.train()
            train_loss = 0.0

            for spec, scalar, target in train_loader:
                spec = spec.to(device)
                scalar = scalar.to(device)
                target = target.to(device)

                optimizer.zero_grad()
                output = model(spec, scalar).squeeze(1)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * spec.size(0)

            scheduler.step()
            train_loss /= len(train_ds)

            # Validation
            model.eval()
            val_preds_log = []
            val_targets_log = []

            with torch.no_grad():
                for spec, scalar, target in val_loader:
                    spec = spec.to(device)
                    scalar = scalar.to(device)
                    target = target.to(device)

                    output = model(spec, scalar).squeeze(1)

                    val_preds_log.append(output.cpu().numpy())
                    val_targets_log.append(target.cpu().numpy())

            val_preds_log = np.concatenate(val_preds_log)
            val_targets_log = np.concatenate(val_targets_log)

            # Convert back to original scale for MAE calculation
            val_preds_orig = np.expm1(val_preds_log)
            val_targets_orig = np.expm1(val_targets_log)

            current_mae = mae_score(val_targets_orig, val_preds_orig)

            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MAE: {current_mae}"
            )

            # Early Stopping & Checkpointing
            if current_mae < best_mae:
                best_mae = current_mae
                patience_counter = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Load best model for OOF generation
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        model.eval()

        fold_preds_log = []
        with torch.no_grad():
            for spec, scalar, _ in val_loader:
                spec = spec.to(device)
                scalar = scalar.to(device)
                output = model(spec, scalar).squeeze(1)
                fold_preds_log.append(output.cpu().numpy())

        fold_preds_log = np.concatenate(fold_preds_log)
        oof_preds[val_idx] = np.expm1(fold_preds_log)
        model_paths.append(model_save_path)

        # Cleanup
        del model, optimizer, scheduler, train_loader, val_loader, train_ds, val_ds
        torch.cuda.empty_cache()
        gc.collect()

    score = mae_score(df["time_to_eruption"].values, oof_preds)
    print(f"Vision Branch Overall CV MAE: {score}")

    return oof_preds, model_paths, scalar_stats_list


def train_meta_learner(oof_tab, oof_vis, y_true):
    """
    Trains a Ridge Regression Meta-Learner on OOF predictions.
    """
    print("Training Meta-Learner...")

    # Feature Stacking
    X_meta = np.column_stack([oof_tab, oof_vis])

    meta_model = Ridge(alpha=Config.META_ALPHA, random_state=Config.SEED)
    meta_model.fit(X_meta, y_true)

    preds = meta_model.predict(X_meta)
    score = mae_score(y_true, preds)

    print(f"Ensemble OOF MAE: {score}")
    print(
        f"Coefficients -> Tabular: {meta_model.coef_[0]:.4f}, Vision: {meta_model.coef_[1]:.4f}"
    )

    return meta_model


def generate_submission(
    df_test: pd.DataFrame,
    test_spec_dir: str,
    tab_models: list,
    tab_feats: list,
    vis_model_paths: list,
    vis_scalar_stats: list,
    meta_model,
):
    """
    Generates final predictions for the test set.
    """
    print("Generating Test Predictions...")

    # 1. Tabular Predictions
    print("Predicting with Tabular Models...")
    tab_preds = np.zeros(len(df_test))
    for model in tab_models:
        tab_preds += predict_lgbm(model, df_test, tab_feats)
    tab_preds /= len(tab_models)

    # 2. Vision Predictions
    print("Predicting with Vision Models...")
    device = get_device()
    vis_preds = np.zeros(len(df_test))

    # Iterate over each fold's model and its specific scalar stats
    for i, (path, stats) in enumerate(zip(vis_model_paths, vis_scalar_stats)):
        # Create test dataset using the training stats of this fold
        ds = VolcanoDataset(df_test, test_spec_dir, mode="test", scalar_stats=stats)
        loader = DataLoader(
            ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
        )

        scalar_dim = len(ds.scalar_cols)
        model = EfficientNetFiLM(scalar_input_dim=scalar_dim).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        fold_preds_log = []
        with torch.no_grad():
            for spec, scalar, _ in loader:
                spec = spec.to(device)
                scalar = scalar.to(device)
                output = model(spec, scalar).squeeze(1)
                fold_preds_log.append(output.cpu().numpy())

        fold_preds_log = np.concatenate(fold_preds_log)
        # Inverse transform
        vis_preds += np.expm1(fold_preds_log)

        del model, ds, loader
        torch.cuda.empty_cache()

    vis_preds /= len(vis_model_paths)

    # 3. Ensemble Prediction
    print("Ensembling...")
    X_meta_test = np.column_stack([tab_preds, vis_preds])
    final_preds = meta_model.predict(X_meta_test)

    # Ensure non-negative
    final_preds = np.maximum(final_preds, 0)

    # 4. Save Submission
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    submission = pd.DataFrame(
        {"segment_id": df_test["segment_id"], "time_to_eruption": final_preds}
    )
    submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


def run():
    """
    Master execution function.
    """
    seed_everything(Config.SEED)

    # 1. Data Prep
    df_train, train_spec_dir, df_test, test_spec_dir = prepare_data(
        load_cached_data=True
    )

    # 2. Tabular Branch
    oof_tab, tab_models, tab_feats = run_tabular_cv(df_train)

    # 3. Vision Branch
    oof_vis, vis_model_paths, vis_scalar_stats = run_vision_cv(df_train, train_spec_dir)

    # 4. Meta Learner
    y_true = df_train["time_to_eruption"].values
    meta_model = train_meta_learner(oof_tab, oof_vis, y_true)

    # 5. Submission
    generate_submission(
        df_test,
        test_spec_dir,
        tab_models,
        tab_feats,
        vis_model_paths,
        vis_scalar_stats,
        meta_model,
    )
