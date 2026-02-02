import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error

from library import config
from library.data_processor import get_train_val_datasets, get_test_dataset
from library.utils import save_submission, seed_everything


def train_lgbm(X_train, y_train, X_val, y_val):
    """
    Trains a LightGBM regressor with early stopping.
    """
    model = lgb.LGBMRegressor(**config.LGBM_PARAMS)

    # Configure callbacks for early stopping and silence
    callbacks = [
        lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=callbacks,
    )
    return model


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    Trains an XGBoost regressor with early stopping.
    """
    # Cite debug_lesson_3: Align Dependencies with Environment Capabilities
    # XGBoost 3.x requires early_stopping_rounds in the constructor, not fit()
    model = xgb.XGBRegressor(
        **config.XGB_PARAMS, early_stopping_rounds=config.EARLY_STOPPING_ROUNDS
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def run_stratified_ensemble(load_cached_data=True):
    """
    Executes the Stratified K-Fold Cross-Validation Ensemble.
    Trains LightGBM and XGBoost on each fold, averages their predictions,
    and generates the final submission file.
    """
    seed_everything(config.SEED)

    print("Loading datasets...")
    # Load initial splits from data processor
    X_train_split, y_train_split, X_val_split, y_val_split = get_train_val_datasets(
        load_cached_data=load_cached_data
    )

    # Load test dataset
    X_test, test_ids = get_test_dataset(load_cached_data=load_cached_data)

    # Combine train and val for K-Fold CV
    print("Merging train and validation sets for Stratified K-Fold CV...")
    X = pd.concat([X_train_split, X_val_split], axis=0).reset_index(drop=True)
    y = pd.concat([y_train_split, y_val_split], axis=0).reset_index(drop=True)

    # Create bins for stratification of continuous target
    # Using 15 bins to ensure sufficient granularity
    num_bins = 15
    y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED
    )

    # Arrays to store predictions
    oof_preds = np.zeros(len(y))
    test_preds_accumulator = np.zeros(len(X_test))

    fold_scores = []

    print(f"Starting {config.N_FOLDS}-Fold Stratified Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
        print(f"\n--- Fold {fold + 1}/{config.N_FOLDS} ---")

        # Split data
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]

        # 1. Train LightGBM
        print("Training LightGBM...")
        lgbm_model = train_lgbm(X_tr, y_tr, X_va, y_va)
        p_lgbm_val = lgbm_model.predict(X_va)
        p_lgbm_test = lgbm_model.predict(X_test)

        # 2. Train XGBoost
        print("Training XGBoost...")
        xgb_model = train_xgboost(X_tr, y_tr, X_va, y_va)
        p_xgb_val = xgb_model.predict(X_va)
        p_xgb_test = xgb_model.predict(X_test)

        # 3. Ensemble (Simple Average)
        p_val_ensemble = (p_lgbm_val + p_xgb_val) / 2.0
        p_test_ensemble = (p_lgbm_test + p_xgb_test) / 2.0

        # Store OOF predictions
        oof_preds[val_idx] = p_val_ensemble

        # Accumulate Test predictions
        test_preds_accumulator += p_test_ensemble

        # Compute and print Fold MAE
        fold_mae = mean_absolute_error(y_va, p_val_ensemble)
        fold_scores.append(fold_mae)
        print(f"Fold {fold + 1} MAE: {fold_mae}")

    # Compute Global Metrics
    global_mae = mean_absolute_error(y, oof_preds)
    avg_cv_mae = np.mean(fold_scores)

    print("\n" + "=" * 30)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 30)
    print(f"Fold MAEs: {fold_scores}")
    print(f"Average CV MAE: {avg_cv_mae}")
    print(f"Global OOF MAE: {global_mae}")

    # Finalize Test Predictions
    final_test_preds = test_preds_accumulator / config.N_FOLDS

    # Save Submission
    print("\nSaving submission...")
    save_submission(final_test_preds, test_ids)
