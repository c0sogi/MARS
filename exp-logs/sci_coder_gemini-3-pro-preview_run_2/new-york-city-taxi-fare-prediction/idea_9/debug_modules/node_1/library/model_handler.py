import os
import numpy as np
import pandas as pd
import xgboost as xgb
from library import config


class ModelHandler:
    """
    Handles the training, evaluation, and inference of the XGBoost model.
    """

    def __init__(self):
        self.model = None
        self.model_path = os.path.join(config.WORKING_DIR, "xgb_model.json")
        # Columns to exclude from the feature matrix
        self.ignore_cols = {"key", "pickup_datetime", config.TARGET_COL}

    def _prepare_features(self, df):
        """
        Prepares the feature matrix by selecting numeric columns and excluding
        metadata or target columns.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: Cleaned feature matrix ready for XGBoost.
        """
        # Identify columns that are not in the ignore list
        potential_cols = [c for c in df.columns if c not in self.ignore_cols]

        # Select only numeric columns from the allowed list
        # This effectively drops string columns like 'key' or 'pickup_datetime' (if not parsed)
        X = df[potential_cols].select_dtypes(include=[np.number])

        return X

    def train_model(self, X_train, y_train, X_val, y_val):
        """
        Trains the XGBoost model with early stopping.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training targets.
            X_val (pd.DataFrame): Validation features.
            y_val (pd.Series): Validation targets.

        Returns:
            xgb.XGBRegressor: The trained model.
        """
        # Prepare feature matrices
        X_train_ready = self._prepare_features(X_train)
        X_val_ready = self._prepare_features(X_val)

        print(f"Training with features: {list(X_train_ready.columns)}")
        print(
            f"Training samples: {len(X_train_ready)}, Validation samples: {len(X_val_ready)}"
        )

        # Initialize model with configuration parameters
        self.model = xgb.XGBRegressor(
            **config.XGB_PARAMS,
            early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
        )

        print("Starting XGBoost training...")
        # Fit the model
        self.model.fit(
            X_train_ready,
            y_train,
            eval_set=[(X_train_ready, y_train), (X_val_ready, y_val)],
            verbose=config.VERBOSE_EVAL,
        )

        # Save the model artifact
        self.model.save_model(self.model_path)
        print(f"Model saved to {self.model_path}")

        # Calculate and print final validation metrics
        # We use the best iteration if early stopping occurred
        if hasattr(self.model, "best_iteration"):
            print(f"Best iteration: {self.model.best_iteration}")

        preds_val = self.model.predict(X_val_ready)
        mse = np.mean((y_val - preds_val) ** 2)
        rmse = np.sqrt(mse)

        print(f"Final Validation MSE: {mse}")
        print(f"Final Validation RMSE: {rmse}")

        return self.model

    def generate_predictions(self, X_test):
        """
        Generates predictions for the test set using the trained model.
        Applies post-processing (minimum fare floor).

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            np.ndarray: Predicted fare amounts.
        """
        # Load model if not currently in memory
        if self.model is None:
            if os.path.exists(self.model_path):
                print(f"Loading model from {self.model_path}...")
                self.model = xgb.XGBRegressor()
                self.model.load_model(self.model_path)
            else:
                raise RuntimeError(
                    "Model has not been trained and no saved model found."
                )

        # Prepare test features
        X_test_ready = self._prepare_features(X_test)

        print("Generating predictions on test set...")
        predictions = self.model.predict(X_test_ready)

        # Post-Processing: Apply minimum fare floor
        print(f"Applying minimum fare floor of ${config.MIN_FARE}...")
        predictions = np.maximum(predictions, config.MIN_FARE)

        return predictions

    def create_submission(self, test_df, predictions):
        """
        Creates and saves the submission CSV file.

        Args:
            test_df (pd.DataFrame): Original test dataframe containing 'key'.
            predictions (np.ndarray): Predicted fare amounts.
        """
        print("Formatting submission...")

        # Create submission DataFrame
        submission = pd.DataFrame({"key": test_df["key"], "fare_amount": predictions})

        # Ensure output directory exists
        os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

        # Save to CSV
        submission.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved successfully to {config.SUBMISSION_PATH}")
