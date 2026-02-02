import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library.config import XGB_PARAMS, TRAIN_CONFIG
from library.utils import log_transform, inverse_log_transform


def train_model(train_df, val_df):
    """
    Trains XGBoost regressors for each target variable using the provided training and validation data.

    Args:
        train_df (pd.DataFrame): Training data containing features and targets.
        val_df (pd.DataFrame): Validation data containing features and targets.

    Returns:
        dict: A dictionary where keys are target names and values are trained XGBRegressor objects.
    """
    targets = TRAIN_CONFIG["target_cols"]

    # Identify feature columns: drop ID, file paths, and targets
    # We assume any column not in the exclude list is a feature
    exclude_cols = ["id", "file_path"] + targets
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]

    print(f"Training with {len(feature_cols)} features.")

    trained_models = {}

    for target in targets:
        print(f"\n--- Training for target: {target} ---")

        # Prepare X and y
        X_train = train_df[feature_cols]
        y_train = train_df[target]

        X_val = val_df[feature_cols]
        y_val = val_df[target]

        # Apply Log Transformation to Targets
        y_train_log = log_transform(y_train)
        y_val_log = log_transform(y_val)

        # Initialize Model
        # We pass early_stopping_rounds to the constructor if supported, or fit.
        # Common practice for sklearn API is passing it to fit, but recent versions allow constructor.
        # We will pass it to fit for broad compatibility.
        model = xgb.XGBRegressor(**XGB_PARAMS)

        # Train
        model.fit(
            X_train,
            y_train_log,
            eval_set=[(X_train, y_train_log), (X_val, y_val_log)],
            early_stopping_rounds=TRAIN_CONFIG["early_stopping_rounds"],
            verbose=TRAIN_CONFIG["verbose_eval"],
        )

        # Evaluate
        # Predict on validation set (log scale)
        preds_log = model.predict(X_val)
        rmse_log = np.sqrt(mean_squared_error(y_val_log, preds_log))

        # Predict on validation set (original scale)
        preds_original = inverse_log_transform(preds_log)
        rmse_original = np.sqrt(mean_squared_error(y_val, preds_original))

        print(f"Validation RMSE (Log Scale): {rmse_log}")
        print(f"Validation RMSE (Original Scale): {rmse_original}")

        trained_models[target] = model

    return trained_models


def predict_model(models, test_df):
    """
    Generates predictions for the test set using the trained models.

    Args:
        models (dict): Dictionary of trained models.
        test_df (pd.DataFrame): Test data containing features.

    Returns:
        pd.DataFrame: DataFrame containing 'id' and predicted targets.
    """
    # Identify feature columns based on the first model's feature names if available,
    # otherwise infer like in training but without targets.
    # To be safe and consistent with training, we filter columns.
    # Note: test_df does not have targets.
    exclude_cols = ["id", "file_path"]
    # We rely on the columns present in test_df that match training features.
    # However, XGBoost sklearn API handles column ordering if passed a dataframe.
    # We just need to drop non-features.

    # Get feature names from one of the models to ensure alignment
    first_model = next(iter(models.values()))
    feature_names = first_model.feature_names_in_

    X_test = test_df[feature_names]

    # Initialize results DataFrame
    results = pd.DataFrame()
    results["id"] = test_df["id"]

    for target, model in models.items():
        # Predict in log space
        preds_log = model.predict(X_test)

        # Inverse transform to original space
        preds_original = inverse_log_transform(preds_log)

        results[target] = preds_original

    return results
