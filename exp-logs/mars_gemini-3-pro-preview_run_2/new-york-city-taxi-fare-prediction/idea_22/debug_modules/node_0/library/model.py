import os
import numpy as np
import pandas as pd
import xgboost as xgb
from library.config import Config


class TaxiFareXGBoost:
    """
    XGBoost Regressor wrapper for the Taxi Fare Prediction task.
    Implements training with early stopping and prediction with post-processing.
    """

    def __init__(self):
        # Initialize the XGBoost Regressor with parameters from Config
        self.model = xgb.XGBRegressor(**Config.XGB_PARAMS)
        self.feature_names = None

    def _get_feature_columns(self, df):
        """
        Selects feature columns for the model.
        Excludes metadata (key, pickup_datetime) and target (fare_amount).
        """
        exclude_cols = {"key", "pickup_datetime", "fare_amount"}
        # Filter columns: keep if not in exclude list
        # We assume the dataframe contains only valid features + the excluded ones
        features = [c for c in df.columns if c not in exclude_cols]
        return features

    def train(self, train_df, val_df):
        """
        Trains the model using the provided Learner (train) and Validation sets.

        Args:
            train_df (pd.DataFrame): The featurized training dataset.
            val_df (pd.DataFrame): The featurized validation dataset.
        """
        print("Preparing datasets for training...")

        # Determine feature columns from the training set
        self.feature_names = self._get_feature_columns(train_df)
        print(f"Selected {len(self.feature_names)} features: {self.feature_names}")

        # Prepare X (Features) and y (Target)
        X_train = train_df[self.feature_names]
        y_train = train_df["fare_amount"]

        X_val = val_df[self.feature_names]
        y_val = val_df["fare_amount"]

        print("Starting XGBoost training with Early Stopping...")
        # Fit the model
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose=Config.VERBOSE_EVAL,
        )

        # Print the best score achieved
        # Note: best_score is the score of the best iteration on the last validation set
        if hasattr(self.model, "best_score"):
            print(f"Best Validation RMSE: {self.model.best_score}")

        # Save the model artifact for persistence
        model_path = os.path.join(Config.WORKING_DIR, "xgb_model.json")
        self.model.save_model(model_path)
        print(f"Model saved to {model_path}")

    def predict(self, test_df):
        """
        Generates predictions for the test dataset.

        Args:
            test_df (pd.DataFrame): The featurized test dataset.

        Returns:
            np.array: Predicted fare amounts.
        """
        # Ensure we use the same features as training
        if self.feature_names is None:
            # If feature_names not set (e.g. loaded model), infer from test_df
            self.feature_names = self._get_feature_columns(test_df)

        X_test = test_df[self.feature_names]

        print("Generating predictions...")
        predictions = self.model.predict(X_test)

        # Post-Processing: Apply Minimum Fare Floor
        # Taxi fares generally start at a base rate (approx $2.50).
        # We clip predictions to ensure no unrealistic low values.
        print("Applying post-processing (Min Fare Floor: $2.50)...")
        predictions = np.maximum(predictions, 2.50)

        return predictions

    def save_submission(self, test_df, predictions):
        """
        Saves the predictions to a CSV file in the required submission format.

        Args:
            test_df (pd.DataFrame): The original test dataframe containing 'key'.
            predictions (np.array): The predicted fare amounts.
        """
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")

        # Create submission DataFrame
        submission = pd.DataFrame({"key": test_df["key"], "fare_amount": predictions})

        # Ensure directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

        # Save to CSV
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print("Submission saved successfully.")
