import lightgbm as lgb
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, mae_score


def train_lgbm_fold(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list):
    """
    Trains a LightGBM model for a single fold using the Energy-Partitioned Tabular Features.

    Args:
        train_df (pd.DataFrame): Training data for this fold.
        val_df (pd.DataFrame): Validation data for this fold.
        feature_cols (list): List of feature column names to use for training.

    Returns:
        tuple: (trained_booster, val_predictions)
    """
    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Prepare Parameters
    params = Config.LGB_PARAMS.copy()

    # Extract training control parameters that shouldn't be in the params dict for lgb.train
    n_estimators = params.pop("n_estimators", 5000)
    early_stopping_rounds = params.pop("early_stopping_rounds", 100)

    # Target column
    target_col = "time_to_eruption"

    # Create LightGBM Datasets
    # Reference to data is kept to avoid memory overhead, but free_raw_data=False ensures safety
    dtrain = lgb.Dataset(
        train_df[feature_cols], label=train_df[target_col], free_raw_data=False
    )
    dval = lgb.Dataset(
        val_df[feature_cols],
        label=val_df[target_col],
        reference=dtrain,
        free_raw_data=False,
    )

    # Callbacks for early stopping and logging
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
        lgb.log_evaluation(period=0),  # Disable default logging to keep output clean
    ]

    # Train the model
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    # Generate Validation Predictions
    val_preds = model.predict(val_df[feature_cols], num_iteration=model.best_iteration)

    # Calculate and Print Metric
    # "When printing validation metrics, please print the full precision without any rounding or formatting."
    score = mae_score(val_df[target_col].values, val_preds)
    print(f"Validation MAE: {score}")

    return model, val_preds


def predict_lgbm(model, test_df: pd.DataFrame, feature_cols: list) -> np.ndarray:
    """
    Generates predictions for the test set using a trained LightGBM model.

    Args:
        model: Trained LightGBM booster.
        test_df (pd.DataFrame): Test dataframe containing features.
        feature_cols (list): List of feature columns used during training.

    Returns:
        np.ndarray: Predictions.
    """
    preds = model.predict(test_df[feature_cols], num_iteration=model.best_iteration)
    return preds
