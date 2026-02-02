import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from library.config import Config


class GBDTPredictor:
    """
    Wrapper for Histogram-based Gradient Boosting Regressor.
    Handles training, evaluation, and persistence of the model for the Taxi Fare Prediction task.
    """

    def __init__(self, params=None):
        """
        Initialize the GBDT predictor.

        Args:
            params (dict, optional): Hyperparameters for HistGradientBoostingRegressor.
                                     Defaults to Config.GBDT_PARAMS.
        """
        self.params = params if params is not None else Config.GBDT_PARAMS
        self.model = HistGradientBoostingRegressor(**self.params)
        self.feature_cols = None

    def fit(self, train_df, val_df=None):
        """
        Trains the GBDT model using the provided training dataframe.
        Supports optional evaluation on a validation dataframe.

        Args:
            train_df (pd.DataFrame): Training data containing features and target.
            val_df (pd.DataFrame, optional): Validation data for evaluation.
        """
        # Identify feature columns dynamically
        # Exclude ID, Target, and raw Datetime columns
        exclude_cols = {Config.ID_COL, Config.TARGET_COL, Config.DATETIME_COL}
        self.feature_cols = [c for c in train_df.columns if c not in exclude_cols]

        # Prepare Training Data
        X_train = train_df[self.feature_cols]
        y_train = train_df[Config.TARGET_COL]

        print(
            f"Training GBDT with {len(X_train)} samples and {len(self.feature_cols)} features..."
        )

        # Fit the model
        # HistGradientBoostingRegressor handles early stopping internally via 'validation_fraction'
        self.model.fit(X_train, y_train)

        # Explicit Validation Evaluation (if provided)
        if val_df is not None:
            print("Evaluating on provided validation set...")
            X_val = val_df[self.feature_cols]
            y_val = val_df[Config.TARGET_COL]

            preds = self.model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))

            # Print full precision as requested
            print(f"Validation RMSE: {rmse}")

    def predict(self, df):
        """
        Generates predictions for the given dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing features.

        Returns:
            np.ndarray: Predicted fare amounts.
        """
        if self.feature_cols is None:
            # Attempt to infer features if not set (e.g. if loaded manually without .load())
            # Note: This assumes df has the same schema as training data minus target
            exclude_cols = {Config.ID_COL, Config.TARGET_COL, Config.DATETIME_COL}
            self.feature_cols = [c for c in df.columns if c not in exclude_cols]

        # Ensure we select the exact same columns in the same order
        X = df[self.feature_cols]

        return self.model.predict(X)

    def save(self, path):
        """
        Saves the model and the feature column list to disk.

        Args:
            path (str): File path to save the model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save both the model and the feature list to ensure consistency during inference
        artifact = {"model": self.model, "feature_cols": self.feature_cols}
        joblib.dump(artifact, path)
        print(f"GBDT model saved to {path}")

    def load(self, path):
        """
        Loads the model and feature column list from disk.

        Args:
            path (str): File path to load the model from.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")

        artifact = joblib.load(path)
        self.model = artifact["model"]
        self.feature_cols = artifact["feature_cols"]
        print(f"GBDT model loaded from {path}")
