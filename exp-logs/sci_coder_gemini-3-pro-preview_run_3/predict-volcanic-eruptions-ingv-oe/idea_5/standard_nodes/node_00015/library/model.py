import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from library.config import (
    LGBM_PARAMS,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    SUBMISSION_DIR,
    SUBMISSION_PATH,
    ID_COL,
)


def train_model(X_train, y_train, X_val, y_val, params=None):
    """
    Trains a LightGBM Regressor with early stopping and MAE monitoring.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training targets.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation targets.
        params (dict, optional): Hyperparameters to override defaults in config.

    Returns:
        lgb.LGBMRegressor: The trained model object.
    """
    # Use default params if none provided
    if params is None:
        params = LGBM_PARAMS.copy()

    # Drop segment_id if present in the feature set to avoid leakage/errors
    # We use errors='ignore' in case the column was already dropped or selected out
    X_train_clean = X_train.drop(columns=[ID_COL], errors="ignore")
    X_val_clean = X_val.drop(columns=[ID_COL], errors="ignore")

    # Initialize the model
    model = lgb.LGBMRegressor(**params)

    # Configure callbacks for early stopping and logging
    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
        lgb.log_evaluation(period=VERBOSE_EVAL),
    ]

    # Fit the model
    model.fit(
        X_train_clean,
        y_train,
        eval_set=[(X_val_clean, y_val)],
        eval_metric="mae",
        callbacks=callbacks,
    )

    # Calculate and print final validation metric with full precision
    val_preds = model.predict(X_val_clean)
    mae = np.mean(np.abs(y_val - val_preds))
    print(f"Final Validation MAE: {mae}")

    return model


def predict_model(model, X_test):
    """
    Generates predictions using the trained model.

    Args:
        model (lgb.LGBMRegressor): Trained model.
        X_test (pd.DataFrame): Test features.

    Returns:
        np.ndarray: Array of predicted time_to_eruption values.
    """
    # Ensure segment_id is not included in prediction features
    X_test_clean = X_test.drop(columns=[ID_COL], errors="ignore")

    return model.predict(X_test_clean)


def generate_submission(model, X_test):
    """
    Generates the submission file for the test set.

    Args:
        model (lgb.LGBMRegressor): Trained model.
        X_test (pd.DataFrame): Test features containing the 'segment_id' column.
    """
    # Validate input
    if ID_COL not in X_test.columns:
        raise ValueError(
            f"Test dataframe must contain '{ID_COL}' column for submission."
        )

    # Extract IDs and generate predictions
    segment_ids = X_test[ID_COL]
    predictions = predict_model(model, X_test)

    # Create submission directory
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    # Create DataFrame
    submission_df = pd.DataFrame({ID_COL: segment_ids, "time_to_eruption": predictions})

    # Save to CSV
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")
