import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import library.config as config


def run_cross_validation(
    X,
    y,
    params=None,
    num_folds=None,
    num_boost_round=None,
    early_stopping_rounds=None,
    verbose_eval=None,
    max_samples=None,
):
    """
    Executes K-Fold cross-validation using LightGBM.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        params (dict, optional): LightGBM hyperparameters. Defaults to config.LGBM_PARAMS.
        num_folds (int, optional): Number of CV folds. Defaults to config.NUM_FOLDS.
        num_boost_round (int, optional): Max boosting iterations. Defaults to config.NUM_BOOST_ROUND.
        early_stopping_rounds (int, optional): Rounds for early stopping. Defaults to config.EARLY_STOPPING_ROUNDS.
        verbose_eval (int, optional): Frequency of metric logging. Defaults to config.VERBOSE_EVAL.
        max_samples (int, optional): If set, subsamples the dataset for debugging.

    Returns:
        list: A list of trained LightGBM Booster objects.
    """
    # Set defaults from config if not provided
    if params is None:
        params = config.LGBM_PARAMS.copy()
    else:
        params = params.copy()

    num_folds = num_folds or config.NUM_FOLDS
    num_boost_round = num_boost_round or config.NUM_BOOST_ROUND
    early_stopping_rounds = early_stopping_rounds or config.EARLY_STOPPING_ROUNDS
    verbose_eval = verbose_eval or config.VERBOSE_EVAL

    # Subsample for debugging if requested
    if max_samples is not None and max_samples < len(X):
        print(f"Subsampling dataset to {max_samples} samples for debugging...")
        X = X.sample(n=max_samples, random_state=config.RANDOM_SEED)
        y = y.loc[X.index]

    # Initialize KFold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=config.RANDOM_SEED)

    models = []
    scores = []

    print(f"Starting {num_folds}-Fold Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        # Split data
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        # Create LightGBM Datasets
        dtrain = lgb.Dataset(X_train_fold, label=y_train_fold)
        dval = lgb.Dataset(X_val_fold, label=y_val_fold, reference=dtrain)

        # Define callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=verbose_eval),
        ]

        # Train model
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Predict and Evaluate
        val_preds = model.predict(X_val_fold, num_iteration=model.best_iteration)
        mae = mean_absolute_error(y_val_fold, val_preds)

        print(f"Fold {fold} MAE: {mae}")

        scores.append(mae)
        models.append(model)

    # Report Average Score
    avg_mae = np.mean(scores)
    print(f"Average MAE across {num_folds} folds: {avg_mae}")

    return models


def predict(models, X):
    """
    Generates predictions using an ensemble of trained models.

    Args:
        models (list): List of trained LightGBM Booster objects.
        X (pd.DataFrame): Feature matrix for inference.

    Returns:
        np.ndarray: Averaged predictions.
    """
    if not models:
        return np.zeros(len(X))

    # Generate predictions for each model
    preds = np.zeros((len(X), len(models)))

    for i, model in enumerate(models):
        preds[:, i] = model.predict(X, num_iteration=model.best_iteration)

    # Average predictions
    avg_preds = np.mean(preds, axis=1)

    return avg_preds
