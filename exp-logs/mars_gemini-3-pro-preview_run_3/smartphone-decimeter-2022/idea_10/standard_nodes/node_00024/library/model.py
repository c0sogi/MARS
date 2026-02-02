import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from library.config import LGBM_PARAMS, SEED, WORKING_DIR


class ResidualRegressor:
    """
    Wraps two LightGBM regressors to predict East and North residuals independently.
    """

    def __init__(self, params=None):
        self.params = params if params else LGBM_PARAMS
        self.model_e = lgb.LGBMRegressor(**self.params)
        self.model_n = lgb.LGBMRegressor(**self.params)
        self.feature_names = None

    def fit(self, X, y_e, y_n, X_val=None, y_e_val=None, y_n_val=None):
        """
        Trains the East and North models.
        """
        self.feature_names = X.columns.tolist()

        # Callbacks for early stopping and logging
        callbacks_e = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        callbacks_n = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        # Train East Model
        eval_set_e = [(X_val, y_e_val)] if X_val is not None else None
        self.model_e.fit(
            X, y_e, eval_set=eval_set_e, eval_metric="mae", callbacks=callbacks_e
        )

        # Train North Model
        eval_set_n = [(X_val, y_n_val)] if X_val is not None else None
        self.model_n.fit(
            X, y_n, eval_set=eval_set_n, eval_metric="mae", callbacks=callbacks_n
        )

    def predict(self, X):
        """
        Predicts residuals.
        """
        pred_e = self.model_e.predict(X)
        pred_n = self.model_n.predict(X)
        return pred_e, pred_n


def get_feature_columns(df):
    """
    Identifies numerical feature columns, excluding metadata and IDs.
    """
    exclude_cols = [
        "tripId",
        "drive_id",
        "phone_name",
        "UnixTimeMillis",
        "gnss_path",
        "imu_path",
        "gt_path",
        "LatitudeDegrees",
        "LongitudeDegrees",
        "AltitudeMeters",
    ]
    # Also exclude any object columns just in case
    feats = [c for c in df.columns if c not in exclude_cols and df[c].dtype != "object"]
    return feats


def train_model(X_full, y_full, groups):
    """
    Performs GroupKFold cross-validation to train an ensemble of residual regressors.

    Args:
        X_full (pd.DataFrame): Feature dataframe.
        y_full (pd.DataFrame): Target dataframe (must contain 'target_e', 'target_n').
        groups (pd.Series): Group labels for CV (drive_id).

    Returns:
        list: A list of trained ResidualRegressor objects (one per fold).
    """
    feature_cols = get_feature_columns(X_full)
    print(f"Training with {len(feature_cols)} features: {feature_cols[:5]}...")

    X = X_full[feature_cols]
    y_e = y_full["target_e"]
    y_n = y_full["target_n"]

    gkf = GroupKFold(n_splits=5)
    models = []

    oof_preds_e = np.zeros(len(X))
    oof_preds_n = np.zeros(len(X))

    fold_scores = []

    print(f"Starting GroupKFold training on {len(X)} samples...")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_e, groups)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_e_train, y_e_val = y_e.iloc[train_idx], y_e.iloc[val_idx]
        y_n_train, y_n_val = y_n.iloc[train_idx], y_n.iloc[val_idx]

        model = ResidualRegressor(LGBM_PARAMS)
        model.fit(X_train, y_e_train, y_n_train, X_val, y_e_val, y_n_val)

        val_pred_e, val_pred_n = model.predict(X_val)

        oof_preds_e[val_idx] = val_pred_e
        oof_preds_n[val_idx] = val_pred_n

        # Calculate fold MAE (mean of East MAE and North MAE)
        mae_e = np.mean(np.abs(y_e_val - val_pred_e))
        mae_n = np.mean(np.abs(y_n_val - val_pred_n))
        avg_mae = (mae_e + mae_n) / 2

        fold_scores.append(avg_mae)
        print(f"Fold {fold+1} MAE: East={mae_e}, North={mae_n}, Avg={avg_mae}")

        models.append(model)

    # Overall metrics
    total_mae_e = np.mean(np.abs(y_e - oof_preds_e))
    total_mae_n = np.mean(np.abs(y_n - oof_preds_n))
    total_avg_mae = (total_mae_e + total_mae_n) / 2

    print(
        f"Overall CV MAE: East={total_mae_e}, North={total_mae_n}, Avg={total_avg_mae}"
    )

    return models


def predict_residuals(models, X_test):
    """
    Generates predictions using an ensemble of trained models.

    Args:
        models (list): List of trained ResidualRegressor objects.
        X_test (pd.DataFrame): Test features.

    Returns:
        pd.DataFrame: DataFrame with 'pred_e' and 'pred_n' columns.
    """
    feature_cols = get_feature_columns(X_test)
    X = X_test[feature_cols]

    pred_e_sum = np.zeros(len(X))
    pred_n_sum = np.zeros(len(X))

    for model in models:
        pe, pn = model.predict(X)
        pred_e_sum += pe
        pred_n_sum += pn

    avg_pred_e = pred_e_sum / len(models)
    avg_pred_n = pred_n_sum / len(models)

    return pd.DataFrame({"pred_e": avg_pred_e, "pred_n": avg_pred_n})
