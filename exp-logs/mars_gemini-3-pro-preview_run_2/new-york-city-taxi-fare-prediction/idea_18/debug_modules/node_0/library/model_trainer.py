import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from library.config import Config


class XGBTrainer:
    """
    Trainer class for the Taxi Fare Prediction task using XGBoost.
    Implements the Multi-Moment Hierarchical Dual-Hygiene Gradient Boosting strategy.
    """

    def __init__(self):
        """
        Initialize the trainer with hyperparameters from Config.
        """
        self.params = Config.XGB_PARAMS.copy()
        self.model = xgb.XGBRegressor(**self.params)
        self.feature_cols = None

    def _prepare_data(self, df: pd.DataFrame, is_training: bool = True):
        """
        Separates features (X) and target (y) from the dataframe.
        Excludes metadata columns like key, pickup_datetime, and target.

        Args:
            df (pd.DataFrame): The dataframe to process.
            is_training (bool): Whether this is for training (requires target).

        Returns:
            tuple: (X, y) if is_training is True, else X.
        """
        # Columns to exclude from features
        exclude_cols = ["key", "fare_amount", "pickup_datetime"]

        # Identify feature columns dynamically
        # If we have already trained, use the stored feature columns to ensure consistency
        if self.feature_cols is None:
            feature_cols = [c for c in df.columns if c not in exclude_cols]
            # Ensure no object/string columns slip in (except categories if handled, but here we expect numerics)
            # XGBoost can handle categories, but our engineering pipeline produces numerics.
            self.feature_cols = feature_cols

        X = df[self.feature_cols]

        if is_training:
            if "fare_amount" not in df.columns:
                raise ValueError(
                    "Target 'fare_amount' not found in training dataframe."
                )
            y = df["fare_amount"]
            return X, y
        else:
            return X

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """
        Trains the XGBoost model with early stopping.

        Args:
            train_df (pd.DataFrame): Training data containing features and target.
            val_df (pd.DataFrame): Validation data for evaluation.
        """
        print("Preparing data for training...")
        X_train, y_train = self._prepare_data(train_df, is_training=True)
        X_val, y_val = self._prepare_data(val_df, is_training=True)

        print(f"Training with {len(self.feature_cols)} features: {self.feature_cols}")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

        # Fit the model
        # Note: early_stopping_rounds is passed to fit() in the sklearn API
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_metric="rmse",
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose=Config.VERBOSE_EVAL,
        )

        print("Training complete.")
        if hasattr(self.model, "best_score"):
            print(f"Best Validation RMSE: {self.model.best_score:.10f}")
        if hasattr(self.model, "best_iteration"):
            print(f"Best Iteration: {self.model.best_iteration}")

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions for the given dataframe.
        Applies post-processing (minimum fare floor).

        Args:
            df (pd.DataFrame): Dataframe containing features.

        Returns:
            np.ndarray: Predicted fare amounts.
        """
        X = self._prepare_data(df, is_training=False)

        # Generate raw predictions
        preds = self.model.predict(X)

        # Post-processing: Apply minimum fare floor ($2.50)
        # As per strategy: "Apply a minimum fare floor ($2.50)."
        preds = np.maximum(preds, 2.50)

        return preds

    def evaluate(self, df: pd.DataFrame) -> float:
        """
        Evaluates the model on a labeled dataset and returns RMSE.

        Args:
            df (pd.DataFrame): Dataframe containing features and target.

        Returns:
            float: Root Mean Squared Error.
        """
        X, y_true = self._prepare_data(df, is_training=True)
        y_pred = self.predict(df)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        print(f"Evaluation RMSE: {rmse:.10f}")
        return rmse

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Retrieves feature importance from the trained model.

        Returns:
            pd.DataFrame: Feature importance sorted by gain.
        """
        if self.feature_cols is None:
            return pd.DataFrame()

        importance = self.model.feature_importances_
        fi_df = pd.DataFrame(
            {"feature": self.feature_cols, "importance": importance}
        ).sort_values(by="importance", ascending=False)

        return fi_df

    def save_model(self, filepath: str):
        """
        Saves the trained model to a file.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_model(filepath)
        print(f"Model saved to {filepath}")

    def generate_submission_file(self, test_df: pd.DataFrame, output_path: str = None):
        """
        Generates predictions for the test set and saves the submission CSV.

        Args:
            test_df (pd.DataFrame): Processed test dataframe.
            output_path (str): Path to save the submission file. Defaults to Config.SUBMISSION_PATH.
        """
        if output_path is None:
            output_path = Config.SUBMISSION_PATH

        print("Generating predictions for submission...")
        predictions = self.predict(test_df)

        # Create submission dataframe
        # Ensure 'key' column exists in test_df
        if "key" not in test_df.columns:
            raise ValueError(
                "Column 'key' missing from test dataframe. Cannot generate submission."
            )

        submission_df = pd.DataFrame(
            {"key": test_df["key"], "fare_amount": predictions}
        )

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save to CSV
        submission_df.to_csv(output_path, index=False)
        print(f"Submission file saved to {output_path}")
        print(f"Submission shape: {submission_df.shape}")
        print(f"Submission head:\n{submission_df.head()}")
