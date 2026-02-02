import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import warnings

# Import configuration
from library.config import (
    XGB_PARAMS,
    TARGET_COLS,
    LOG_TRANSFORM_TARGETS,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    RANDOM_SEED,
)


class DualEnergyPredictor:
    """
    Wrapper for training and predicting with two XGBoost models for formation and bandgap energy.
    Handles logarithmic transformation of targets to optimize for RMSLE.
    """

    def __init__(self):
        self.models = {}
        self.feature_cols = None
        # Initialize separate models for each target
        for target in TARGET_COLS:
            self.models[target] = xgb.XGBRegressor(**XGB_PARAMS)

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Selects feature columns from the dataframe.
        Drops ID, file paths, and targets.
        """
        # Exclude non-feature columns
        exclude = ["id", "file_path"] + TARGET_COLS

        # Get all available columns that are not excluded
        cols = [c for c in df.columns if c not in exclude]

        # If we have already trained, ensure we use the same columns
        if self.feature_cols is not None:
            # Check for missing columns and add them with 0s if necessary
            # (This handles cases where test set might miss some columns present in train, though unlikely with this pipeline)
            missing = set(self.feature_cols) - set(cols)
            for c in missing:
                df[c] = 0.0

            # Select only trained columns in correct order
            return df[self.feature_cols]

        return df[cols]

    def _transform_target(self, y):
        """
        Applies log(1+y) transformation if configured.
        """
        if LOG_TRANSFORM_TARGETS:
            return np.log1p(y)
        return y

    def _inverse_transform_target(self, z):
        """
        Applies exp(z) - 1 transformation if configured.
        Ensures non-negative predictions.
        """
        if LOG_TRANSFORM_TARGETS:
            return np.expm1(z)
        return z

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame = None):
        """
        Trains the dual models on the provided data.

        Args:
            train_df: Training data with features and targets.
            val_df: Validation data for early stopping.
        """
        # Prepare features
        X_train = self._get_features(train_df)

        # Save feature names from training set to ensure consistency during prediction
        self.feature_cols = X_train.columns.tolist()

        X_val = None
        if val_df is not None:
            X_val = self._get_features(val_df)

        print(
            f"Training on {len(X_train)} samples with {len(self.feature_cols)} features."
        )

        for target in TARGET_COLS:
            print(f"\nTraining model for target: {target}")

            y_train = train_df[target]
            y_train_trans = self._transform_target(y_train)

            eval_set = [(X_train, y_train_trans)]

            if val_df is not None:
                y_val = val_df[target]
                y_val_trans = self._transform_target(y_val)
                eval_set.append((X_val, y_val_trans))

            # Train
            self.models[target].fit(
                X_train,
                y_train_trans,
                eval_set=eval_set,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=VERBOSE_EVAL,
            )

            # Evaluate on validation set if available
            if val_df is not None:
                # Predict transformed values
                z_pred = self.models[target].predict(X_val)
                # Inverse transform
                y_pred = self._inverse_transform_target(z_pred)
                # Clip negative values just in case (energy shouldn't be negative here)
                y_pred = np.maximum(y_pred, 0)

                # Calculate RMSLE (which is RMSE on log1p transformed data)
                # Since we trained on log1p, the validation metric from XGBoost is essentially RMSLE^2 or RMSLE depending on objective
                # We calculate explicitly using sklearn for clarity
                rmsle = np.sqrt(mean_squared_error(np.log1p(y_val), np.log1p(y_pred)))
                print(f"Validation RMSLE for {target}: {rmsle}")

    def predict(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates predictions for the test set.

        Args:
            test_df: Test data with features.

        Returns:
            DataFrame with 'id' and predicted columns.
        """
        if self.feature_cols is None:
            raise RuntimeError("Model has not been trained yet.")

        X_test = self._get_features(test_df)

        predictions = pd.DataFrame()
        predictions["id"] = test_df["id"]

        for target in TARGET_COLS:
            # Predict in transformed space
            z_pred = self.models[target].predict(X_test)
            # Inverse transform
            y_pred = self._inverse_transform_target(z_pred)
            # Ensure non-negative
            y_pred = np.maximum(y_pred, 0)

            predictions[target] = y_pred

        return predictions

    def evaluate(self, df: pd.DataFrame):
        """
        Evaluates the model on a dataframe containing ground truth.
        Returns the mean column-wise RMSLE.
        """
        preds = self.predict(df)

        rmsle_scores = []
        for target in TARGET_COLS:
            y_true = df[target]
            y_pred = preds[target]

            score = np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
            rmsle_scores.append(score)

        mean_rmsle = np.mean(rmsle_scores)
        print(f"\nOverall Mean RMSLE: {mean_rmsle}")
        return mean_rmsle
