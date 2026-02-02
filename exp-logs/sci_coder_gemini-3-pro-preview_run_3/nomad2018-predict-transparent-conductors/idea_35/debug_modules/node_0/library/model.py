import numpy as np
import pandas as pd
import xgboost as xgb
import os
from sklearn.base import BaseEstimator, RegressorMixin
from library.config import XGB_PARAMS, EARLY_STOPPING_ROUNDS, VERBOSE_EVAL, TARGET_COLS
from library.utils import calculate_rmsle, expm1_transform, save_submission


class DualTargetRegressor(BaseEstimator, RegressorMixin):
    """
    Wrapper class for training independent XGBoost regressors for each target variable.
    Handles log-transformed targets internally by expecting log-scale inputs for fit
    and returning original-scale predictions.
    """

    def __init__(self, params=None):
        self.params = params if params else XGB_PARAMS
        # Initialize a separate regressor for each target
        self.models = {col: xgb.XGBRegressor(**self.params) for col in TARGET_COLS}

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the models for each target.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.DataFrame): Training targets (log-transformed).
            X_val (pd.DataFrame, optional): Validation features.
            y_val (pd.DataFrame, optional): Validation targets (log-transformed).
        """
        scores = {}

        for target in TARGET_COLS:
            print(f"\nTraining XGBoost for target: {target}")
            model = self.models[target]

            # Prepare evaluation set for early stopping
            eval_set = []
            if X_val is not None and y_val is not None:
                # Extract specific target series
                y_t = y_train[target]
                y_v = y_val[target]
                eval_set = [(X_train, y_t), (X_val, y_v)]
            else:
                y_t = y_train[target]

            # Train the model
            model.fit(
                X_train,
                y_t,
                eval_set=eval_set,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=VERBOSE_EVAL,
            )

            # Evaluate on validation set if available
            if X_val is not None and y_val is not None:
                # Predict in log scale
                preds_log = model.predict(X_val)
                # Transform predictions back to original scale
                preds_orig = expm1_transform(preds_log)
                # Transform ground truth back to original scale
                y_true_orig = expm1_transform(y_val[target].values)

                # Calculate metric
                score = calculate_rmsle(y_true_orig, preds_orig)
                scores[target] = score
                print(f"Validation RMSLE for {target}: {score}")

        if scores:
            avg_score = np.mean(list(scores.values()))
            print(f"\nAverage Validation RMSLE: {avg_score}")

        return self

    def predict(self, X):
        """
        Generates predictions for all targets.

        Args:
            X (pd.DataFrame): Features.

        Returns:
            pd.DataFrame: Predictions in original scale with columns matching TARGET_COLS.
        """
        predictions = {}
        for target in TARGET_COLS:
            model = self.models[target]
            # Predict (model outputs log-scale values)
            pred_log = model.predict(X)
            # Inverse transform to original scale
            pred_orig = expm1_transform(pred_log)
            predictions[target] = pred_orig

        return pd.DataFrame(predictions)


def train_and_predict(train_df, val_df, test_df):
    """
    Orchestrates the training workflow:
    1. Separates features and targets.
    2. Trains the DualTargetRegressor.
    3. Generates predictions for the test set.
    4. Saves the submission file.

    Args:
        train_df (pd.DataFrame): Prepared training data.
        val_df (pd.DataFrame): Prepared validation data.
        test_df (pd.DataFrame): Prepared test data.

    Returns:
        DualTargetRegressor: The trained model instance.
    """
    # Identify feature columns (exclude ID and targets)
    feature_cols = [c for c in train_df.columns if c not in TARGET_COLS + ["id"]]

    print(f"Training with {len(feature_cols)} features.")

    # Prepare Training Data
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_COLS]

    # Prepare Validation Data
    X_val = val_df[feature_cols]
    y_val = val_df[TARGET_COLS]

    # Prepare Test Data
    X_test = test_df[feature_cols]
    test_ids = test_df["id"]

    # Initialize and Train Model
    model = DualTargetRegressor()
    model.fit(X_train, y_train, X_val, y_val)

    # Generate Test Predictions
    print("\nGenerating predictions for test set...")
    test_preds_df = model.predict(X_test)

    # Ensure correct column order for submission
    predictions = test_preds_df[TARGET_COLS].values

    # Save Submission
    save_submission(test_ids.values, predictions, filename="submission.csv")

    return model
