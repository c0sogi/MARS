import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from library.config import (
    NUM_LEAVES,
    LEARNING_RATE,
    N_ESTIMATORS,
    EARLY_STOPPING_ROUNDS,
    OBJECTIVE,
    METRIC,
    VERBOSITY,
    SEED,
)
from library.utils import calculate_mae


def train_lgbm_fold(X_train, y_train, X_val, y_val):
    """
    Trains a single LightGBM regressor with early stopping.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target.

    Returns:
        model: The trained LightGBM model.
        float: The best validation MAE score.
    """
    model = lgb.LGBMRegressor(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        objective=OBJECTIVE,
        metric=METRIC,
        verbosity=VERBOSITY,
        random_state=SEED,
        n_jobs=-1,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
        lgb.log_evaluation(period=0),  # Suppress logging as requested
    ]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=METRIC,
        callbacks=callbacks,
    )

    # Best score is typically accessed via best_score_ attribute for the valid set
    # metric name is 'l1' for 'mae' in lightgbm structure usually, or 'mae' if specified
    best_score = (
        model.best_score_["valid_0"][METRIC] if "valid_0" in model.best_score_ else 0.0
    )

    return model, best_score


def run_cross_validation(df, n_splits=5):
    """
    Executes Stratified K-Fold Cross-Validation.

    Args:
        df (pd.DataFrame): The dataset containing features, 'segment_id', and 'time_to_eruption'.
        n_splits (int): Number of folds.

    Returns:
        list: A list of trained LightGBM models.
        pd.DataFrame: Out-of-Fold predictions with columns ['segment_id', 'pred', 'target'].
        dict: Dictionary containing fold scores and overall MAE.
    """
    # Prepare features and target
    feature_cols = [
        c for c in df.columns if c not in ["segment_id", "time_to_eruption"]
    ]
    X = df[feature_cols]
    y = df["time_to_eruption"]
    ids = df["segment_id"].values

    # Create bins for stratification (Regression -> Stratified Split)
    # We use 10 bins as done in the metadata generation script
    num_bins = 10
    # Handle edge case where dataset is small
    if len(df) < num_bins:
        num_bins = len(df) // 2

    target_bins = pd.qcut(y, q=num_bins, labels=False, duplicates="drop")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    models = []
    oof_preds = np.zeros(len(df))
    fold_scores = []

    print(f"Starting {n_splits}-Fold Stratified Cross-Validation...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, target_bins)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model, score = train_lgbm_fold(X_train, y_train, X_val, y_val)

        # Predict on validation set
        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds

        # Calculate explicit MAE for this fold to be sure
        fold_mae = calculate_mae(y_val, val_preds)
        fold_scores.append(fold_mae)
        models.append(model)

        print(f"Fold {fold + 1}/{n_splits} MAE: {fold_mae}")

    # Calculate global OOF MAE
    overall_mae = calculate_mae(y, oof_preds)
    print(f"Overall OOF MAE: {overall_mae}")

    oof_df = pd.DataFrame({"segment_id": ids, "pred": oof_preds, "target": y.values})

    metrics = {"fold_scores": fold_scores, "overall_mae": overall_mae}

    return models, oof_df, metrics


def generate_predictions(models, test_df):
    """
    Generates predictions for the test set by averaging predictions from all provided models.

    Args:
        models (list): List of trained LightGBM models.
        test_df (pd.DataFrame): Test dataset containing features and 'segment_id'.

    Returns:
        pd.DataFrame: DataFrame with 'segment_id' and 'time_to_eruption' (predicted).
    """
    feature_cols = [
        c for c in test_df.columns if c not in ["segment_id", "time_to_eruption"]
    ]
    X_test = test_df[feature_cols]

    # Accumulate predictions
    avg_preds = np.zeros(len(test_df))

    for model in models:
        preds = model.predict(X_test)
        avg_preds += preds

    # Average
    avg_preds /= len(models)

    submission = pd.DataFrame(
        {"segment_id": test_df["segment_id"], "time_to_eruption": avg_preds}
    )

    return submission
