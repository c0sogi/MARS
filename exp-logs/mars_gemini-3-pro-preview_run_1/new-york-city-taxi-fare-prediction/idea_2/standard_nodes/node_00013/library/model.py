import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from library import config


class FareModel:
    """
    Wrapper class for LightGBM Regressor optimized for Taxi Fare Prediction.
    """

    def __init__(self, params=None):
        """
        Initialize the model with hyperparameters.

        Args:
            params (dict, optional): LightGBM parameters. Defaults to config.LGBM_PARAMS.
        """
        self.params = params if params is not None else config.LGBM_PARAMS.copy()
        self.model = None
        self.feature_names = None

    def train(self, train_df, val_df, target_col="fare_amount", ignore_cols=None):
        """
        Trains the LightGBM model with early stopping.

        Args:
            train_df (pd.DataFrame): Training data containing features and target.
            val_df (pd.DataFrame): Validation data containing features and target.
            target_col (str): Name of the target column.
            ignore_cols (list): List of column names to exclude from features.
        """
        if ignore_cols is None:
            ignore_cols = ["key", "pickup_datetime"]

        # Prepare Feature Sets
        # Filter out target and ignored columns
        features = [
            c for c in train_df.columns if c != target_col and c not in ignore_cols
        ]
        self.feature_names = features

        print(f"Training on {len(features)} features: {features}")

        X_train = train_df[features]
        y_train = train_df[target_col]
        X_val = val_df[features]
        y_val = val_df[target_col]

        # Create LightGBM Datasets
        # free_raw_data=False ensures data isn't freed before we might need it,
        # though usually True is fine if we don't reuse the python object.
        train_set = lgb.Dataset(
            X_train, label=y_train, feature_name=features, free_raw_data=False
        )
        val_set = lgb.Dataset(
            X_val,
            label=y_val,
            feature_name=features,
            reference=train_set,
            free_raw_data=False,
        )

        # Callbacks
        callbacks = [
            lgb.early_stopping(stopping_rounds=config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=config.VERBOSE_EVAL),
        ]

        # Train
        print("Starting training...")
        self.model = lgb.train(
            params=self.params,
            train_set=train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Log best score with full precision
        if self.model.best_score:
            val_rmse = self.model.best_score["valid"]["rmse"]
            print(f"Best Validation RMSE: {val_rmse:.16f}")

    def predict(self, test_df, ignore_cols=None):
        """
        Generates predictions for the test set.

        Args:
            test_df (pd.DataFrame): Test data.
            ignore_cols (list): Columns to exclude.

        Returns:
            np.array: Predicted fare amounts.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        if ignore_cols is None:
            ignore_cols = ["key", "pickup_datetime", "fare_amount"]

        # Ensure we use the same features as training
        # If test_df has extra columns, ignore them. If missing, error will occur.
        X_test = test_df[self.feature_names]

        return self.model.predict(X_test, num_iteration=self.model.best_iteration)

    def get_feature_importance(self):
        """
        Returns a DataFrame of feature importances.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        importance = self.model.feature_importance(importance_type="gain")
        return pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values(by="importance", ascending=False)

    def save(self, filepath):
        """
        Saves the trained booster to a file.

        Args:
            filepath (str): Path to save the model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save booster
        self.model.save_model(filepath)

        # Save feature names separately to ensure consistency during loading/inference
        # We use a simple text file for this to avoid pickle
        feature_path = filepath + ".features"
        with open(feature_path, "w") as f:
            f.write("\n".join(self.feature_names))

        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """
        Loads a trained booster from a file.

        Args:
            filepath (str): Path to the model file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        self.model = lgb.Booster(model_file=filepath)

        # Load feature names
        feature_path = filepath + ".features"
        if os.path.exists(feature_path):
            with open(feature_path, "r") as f:
                self.feature_names = f.read().splitlines()
        else:
            print(
                "Warning: Feature names file not found. Feature consistency not guaranteed."
            )
            self.feature_names = self.model.feature_name()

        print(f"Model loaded from {filepath}")
