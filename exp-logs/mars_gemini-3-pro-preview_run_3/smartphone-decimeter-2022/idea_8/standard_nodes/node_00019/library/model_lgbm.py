import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from library.config import Config
import os


def train_residual_model(
    df, feature_cols, target_col, group_col="drive_id", params=None, n_splits=5
):
    """
    Trains LightGBM models using GroupKFold cross-validation to predict residuals.

    Args:
        df (pd.DataFrame): Training data containing features, target, and group column.
        feature_cols (list): List of feature column names.
        target_col (str): Name of the target column (e.g., 'target_E' or 'target_N').
        group_col (str): Name of the column to use for grouping (default: 'drive_id').
        params (dict): LightGBM parameters. If None, uses Config.LGBM_PARAMS.
        n_splits (int): Number of CV splits.

    Returns:
        list: List of trained LightGBM models (one per fold).
        np.array: Out-of-fold (OOF) predictions for the entire dataset.
        float: Overall Mean Absolute Error (MAE) score.
    """
    if params is None:
        params = Config.LGBM_PARAMS.copy()

    # Ensure deterministic behavior
    params["random_state"] = Config.SEED

    X = df[feature_cols]
    y = df[target_col]
    groups = df[group_col]

    gkf = GroupKFold(n_splits=n_splits)
    models = []
    oof_preds = np.zeros(len(df))

    print(f"Training LightGBM for target: {target_col}")
    print(f"Input shape: {X.shape}, Groups: {groups.nunique()}")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Create LightGBM datasets
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

        callbacks = [
            lgb.early_stopping(
                stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=False
            ),
            lgb.log_evaluation(period=Config.VERBOSE_EVAL),
        ]

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=Config.N_ESTIMATORS,
            valid_sets=[lgb_train, lgb_val],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = val_preds

        fold_score = mean_absolute_error(y_val, val_preds)
        print(f"Fold {fold+1} MAE: {fold_score}")

        models.append(model)

    overall_score = mean_absolute_error(y, oof_preds)
    print(f"Overall CV MAE for {target_col}: {overall_score}")

    return models, oof_preds, overall_score


def predict_residuals(df, feature_cols, models):
    """
    Predicts residuals using a list of trained models (ensemble averaging).

    Args:
        df (pd.DataFrame): Data to predict on (Test or Validation set).
        feature_cols (list): List of feature column names.
        models (list): List of trained LightGBM models.

    Returns:
        np.array: Averaged predictions from all models.
    """
    X = df[feature_cols]
    preds = np.zeros(len(X))

    if not models:
        print("Warning: No models provided for prediction. Returning zeros.")
        return preds

    for model in models:
        preds += model.predict(X, num_iteration=model.best_iteration)

    preds /= len(models)
    return preds
