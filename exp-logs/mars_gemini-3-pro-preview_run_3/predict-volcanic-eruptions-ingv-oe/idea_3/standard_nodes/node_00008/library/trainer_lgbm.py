import lightgbm as lgb
import pandas as pd
import numpy as np
from library.config import LGBM_PARAMS, LGBM_EARLY_STOPPING_ROUNDS, SEED
from library.utils import seed_everything, calculate_mae


def train_lgbm_model(train_df, val_df):
    """
    Trains a LightGBM Regressor using the provided training and validation dataframes.

    Args:
        train_df (pd.DataFrame): Training data containing features and 'time_to_eruption'.
        val_df (pd.DataFrame): Validation data containing features and 'time_to_eruption'.

    Returns:
        model (lgb.Booster): The trained LightGBM model.
        val_preds (np.ndarray): Predictions on the validation set.
    """
    seed_everything(SEED)

    # Identify feature columns (exclude metadata and target)
    exclude_cols = ["segment_id", "time_to_eruption"]
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    print(f"Training LightGBM with {len(feature_cols)} features...")

    # Prepare X and y
    X_train = train_df[feature_cols]
    y_train = train_df["time_to_eruption"]

    X_val = val_df[feature_cols]
    y_val = val_df["time_to_eruption"]

    # Create LightGBM Datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    # Prepare parameters
    # Make a copy to avoid modifying the global config dict
    params = LGBM_PARAMS.copy()

    # Extract n_estimators to use as num_boost_round
    num_boost_round = params.pop("n_estimators", 10000)

    # Setup callbacks
    callbacks = [
        lgb.early_stopping(stopping_rounds=LGBM_EARLY_STOPPING_ROUNDS, verbose=True),
        lgb.log_evaluation(period=100),
    ]

    # Train the model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    # Generate validation predictions
    val_preds = model.predict(X_val, num_iteration=model.best_iteration)

    # Calculate and print final metric
    mae = calculate_mae(y_val, val_preds)
    print(f"Final LightGBM Validation MAE: {mae}")

    return model, val_preds


def predict_lgbm(model, test_df):
    """
    Generates predictions for the test set using the trained LightGBM model.

    Args:
        model (lgb.Booster): Trained LightGBM model.
        test_df (pd.DataFrame): Test data containing features.

    Returns:
        np.ndarray: Predicted time_to_eruption values.
    """
    # Identify feature columns
    # We assume test_df has the same feature columns as train_df
    # We just need to drop metadata columns if they exist
    exclude_cols = ["segment_id", "time_to_eruption"]
    feature_cols = [c for c in test_df.columns if c not in exclude_cols]

    X_test = test_df[feature_cols]

    # Predict using the best iteration found during training
    preds = model.predict(X_test, num_iteration=model.best_iteration)

    return preds
