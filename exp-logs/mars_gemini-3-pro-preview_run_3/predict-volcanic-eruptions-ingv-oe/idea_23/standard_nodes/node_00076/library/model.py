import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error
from library.config import Config


def train_lgbm_fold(X_train, y_train, X_val, y_val, params=None):
    """
    Trains a single LightGBM model for a specific fold with early stopping.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.
        params (dict, optional): LightGBM parameters. Defaults to Config.MODEL_PARAMS.

    Returns:
        tuple: (trained_model, validation_mae)
    """
    if params is None:
        params = Config.MODEL_PARAMS.copy()
    else:
        params = params.copy()

    # Extract early stopping rounds to use in callback, removing from params to avoid warnings
    es_rounds = params.pop("early_stopping_rounds", 100)

    # Ensure verbosity is set to silent for the engine (we handle logging via callback)
    params["verbosity"] = -1

    # Create LightGBM datasets
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    # Configure callbacks
    callbacks = [
        lgb.early_stopping(stopping_rounds=es_rounds, verbose=False),
        lgb.log_evaluation(period=1000),
    ]

    # Train the model
    model = lgb.train(
        params, train_set, valid_sets=[train_set, val_set], callbacks=callbacks
    )

    # Generate predictions on validation set using the best iteration
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)

    # Calculate MAE with full precision
    score = mean_absolute_error(y_val, val_pred)

    return model, score


def predict_ensemble(models, X_test):
    """
    Generates predictions using the ensemble of models and returns the average.

    Args:
        models (list): List of trained LightGBM models.
        X_test (pd.DataFrame): Test features.

    Returns:
        np.ndarray: Averaged predictions.
    """
    if not models:
        return np.zeros(len(X_test))

    # Initialize predictions array
    preds = np.zeros(len(X_test))

    # Accumulate predictions from all models
    for model in models:
        preds += model.predict(X_test, num_iteration=model.best_iteration)

    # Compute arithmetic mean
    preds /= len(models)

    return preds


def run_cross_validation(train_df):
    """
    Orchestrates the Stratified K-Fold Cross-Validation training process.

    Args:
        train_df (pd.DataFrame): The processed training dataframe containing features and target.

    Returns:
        list: A list of trained LightGBM models.
    """
    # Separate features and target
    # Exclude metadata columns
    feature_cols = [
        c for c in train_df.columns if c not in ["segment_id", "time_to_eruption"]
    ]
    X = train_df[feature_cols]
    y = train_df["time_to_eruption"]

    # Create bins for stratification based on target quantiles
    # This ensures each fold has a representative distribution of eruption times
    num_bins = 10
    y_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

    # Initialize Stratified K-Fold
    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    models = []
    scores = []

    print(f"Starting Cross-Validation with {Config.N_FOLDS} folds...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins)):
        # Split data
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        # Train model for this fold
        model, score = train_lgbm_fold(X_train, y_train, X_val, y_val)

        # Print full precision score
        print(f"Fold {fold+1} MAE: {score}")

        models.append(model)
        scores.append(score)

    print(f"Average CV MAE: {np.mean(scores)}")
    return models
