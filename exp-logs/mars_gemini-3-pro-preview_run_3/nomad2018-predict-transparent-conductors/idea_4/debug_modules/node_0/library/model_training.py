import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from library.config import Config
from library.utils import log_transform, inverse_log_transform, rmsle_score


def train_xgboost_model(
    X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=100, verbose=False
):
    """
    Trains an XGBoost regressor on the provided features and target.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training target values (original scale).
        X_val (pd.DataFrame or np.ndarray, optional): Validation features.
        y_val (pd.Series or np.ndarray, optional): Validation target values (original scale).
        early_stopping_rounds (int): Rounds for early stopping.
        verbose (bool): Whether to print training progress.

    Returns:
        xgb.XGBRegressor: The trained model.
    """
    # Apply log transformation to targets to match RMSLE objective
    y_train_log = log_transform(y_train)

    eval_set = [(X_train, y_train_log)]

    if X_val is not None and y_val is not None:
        y_val_log = log_transform(y_val)
        eval_set.append((X_val, y_val_log))

    # Initialize model with parameters from Config
    model = xgb.XGBRegressor(**Config.XGB_PARAMS)

    # Fit the model
    # Note: early_stopping_rounds is passed to fit() for compatibility with sklearn wrapper
    model.fit(
        X_train,
        y_train_log,
        eval_set=eval_set,
        early_stopping_rounds=early_stopping_rounds,
        verbose=verbose,
    )

    return model


def cross_validate_model(X, y, n_splits=5, random_state=42):
    """
    Performs K-Fold cross-validation for an XGBoost model on a single target.

    Args:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target vector.
        n_splits (int): Number of folds.
        random_state (int): Seed for reproducibility.

    Returns:
        list: List of RMSLE scores for each fold.
    """
    # Ensure inputs are numpy arrays for consistent indexing
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = []

    print(f"Starting {n_splits}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Train model with early stopping based on the validation fold
        model = train_xgboost_model(
            X_train_fold,
            y_train_fold,
            X_val_fold,
            y_val_fold,
            early_stopping_rounds=50,
            verbose=False,
        )

        # Predict on validation set (predictions are in log scale)
        y_pred_log = model.predict(X_val_fold)

        # Inverse transform to get predictions in original scale (eV)
        y_pred = inverse_log_transform(y_pred_log)

        # Calculate RMSLE
        # Ensure predictions are non-negative (though expm1 is > -1, energy should be > 0)
        y_pred = np.maximum(y_pred, 0)
        score = rmsle_score(y_val_fold, y_pred)
        scores.append(score)

        print(f"Fold {fold+1} RMSLE: {score}")

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    print(f"CV Results: Mean RMSLE = {mean_score}, Std = {std_score}")

    return scores
