import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# Suppress warnings
warnings.filterwarnings("ignore")

# Import from provided library
from library.config import Config
from library.utils import seed_everything, metric_mae
from library.data_loader import load_tabular_dataset, VolcanoDataset, DataLoader
from library.models_tabular import LightGBMTrainer
from library.training import (
    prepare_unified_spectrograms,
    train_vision_fold,
    predict_vision,
    train_meta_learner,
)


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # 2. Load Data
    # This loads features from cache if available, or generates them.
    # It returns raw train/val splits based on metadata, but we will merge them for CV.
    df_train_raw, df_val_raw, df_test = load_tabular_dataset(load_cached=True)

    # Combine for 5-Fold CV
    df_full = pd.concat([df_train_raw, df_val_raw], ignore_index=True)
    y_true = df_full["time_to_eruption"].values

    # Prepare unified spectrogram directory (handles copying if needed)
    unified_spec_dir = prepare_unified_spectrograms()

    # 3. Cross-Validation Loop
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED)

    # Storage
    oof_preds_tabular = np.zeros(len(df_full))
    oof_preds_vision = np.zeros(len(df_full))

    test_preds_tabular = np.zeros((len(df_test), Config.N_FOLDS))
    test_preds_vision = np.zeros((len(df_test), Config.N_FOLDS))

    print(f"Starting {Config.N_FOLDS}-Fold Cross-Validation...")

    for fold_id, (train_idx, val_idx) in enumerate(kf.split(df_full)):
        # Split Data
        df_fold_train = df_full.iloc[train_idx].copy()
        df_fold_val = df_full.iloc[val_idx].copy()

        # --- Branch A: Tabular (LightGBM) ---
        lgbm_trainer = LightGBMTrainer()
        # Suppress LightGBM output further if needed, though config has verbosity=-1
        lgbm_model = lgbm_trainer.train(df_fold_train, df_fold_val, fold_id=fold_id)

        # Predict
        oof_preds_tabular[val_idx] = lgbm_trainer.predict(df_fold_val, model=lgbm_model)
        test_preds_tabular[:, fold_id] = lgbm_trainer.predict(df_test, model=lgbm_model)

        # --- Branch B: Vision (EfficientNet) ---
        # Setup DataLoaders
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

        # Train Vision Model
        vision_model, val_preds_vision_fold = train_vision_fold(
            train_loader, val_loader, fold_id, Config.DEVICE
        )

        # Store OOF (val_preds_vision_fold is already inverse transformed and flattened)
        oof_preds_vision[val_idx] = val_preds_vision_fold

        # Predict Test
        test_preds_vision[:, fold_id] = predict_vision(
            vision_model, test_loader, Config.DEVICE
        )

    # 4. Ensemble & Evaluation
    X_meta = np.column_stack((oof_preds_tabular, oof_preds_vision))
    meta_model = train_meta_learner(X_meta, y_true)

    oof_preds_meta = meta_model.predict(X_meta)

    # Ensure non-negative predictions (physics constraint)
    oof_preds_meta = np.maximum(oof_preds_meta, 0)

    final_mae = metric_mae(y_true, oof_preds_meta)

    print(f"Final Validation Metric: {final_mae}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute errors
    errors = np.abs(y_true - oof_preds_meta)

    # Create analysis dataframe
    df_analysis = df_full.copy()

    # Select numerical feature columns for correlation
    # Exclude ID, target, and file_path
    feature_cols = [
        c
        for c in df_analysis.columns
        if c not in ["segment_id", "time_to_eruption", "file_path"]
    ]
    # Ensure we only check numeric columns
    feature_cols = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(df_analysis[c])
    ]

    # Compute correlations
    correlations = {}
    for col in feature_cols:
        # Handle potential NaNs in features (though pipeline fills them)
        if df_analysis[col].isnull().any():
            continue
        try:
            corr = np.corrcoef(df_analysis[col], errors)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr
        except:
            continue

    # Sort by absolute correlation
    sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features Correlated with Error Magnitude:")
    for feat, corr_val in sorted_corrs[:5]:
        print(f"{feat}: {corr_val:.4f}")

    # 6. Submission
    THRESHOLD = 1920624.12
    if final_mae < THRESHOLD:
        print("\nValidation metric meets threshold. Generating submission...")

        # Average test predictions across folds
        avg_test_tabular = np.mean(test_preds_tabular, axis=1)
        avg_test_vision = np.mean(test_preds_vision, axis=1)

        # Stack for meta-learner
        X_test_meta = np.column_stack((avg_test_tabular, avg_test_vision))

        # Predict
        final_test_preds = meta_model.predict(X_test_meta)
        final_test_preds = np.maximum(final_test_preds, 0)

        # Create submission file
        submission_df = pd.DataFrame(
            {"segment_id": df_test["segment_id"], "time_to_eruption": final_test_preds}
        )

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric {final_mae} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()
