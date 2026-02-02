import os
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from library import config


class FareRegressor:
    def __init__(self):
        """
        Initializes the FareRegressor with the model architecture and parameters.
        """
        self.params = config.MODEL_PARAMS
        self.model = HistGradientBoostingRegressor(**self.params)

        # Define the exact feature columns to use for training and prediction
        # These match the features generated in data_processor.py
        self.feature_names = [
            "pickup_longitude",
            "pickup_latitude",
            "dropoff_longitude",
            "dropoff_latitude",
            "passenger_count",
            "year",
            "month",
            "day",
            "day_of_week",
            "hour",
            "haversine_dist",
            "abs_diff_lon",
            "abs_diff_lat",
            # Landmark Distances
            "dist_pickup_JFK",
            "dist_dropoff_JFK",
            "dist_pickup_LGA",
            "dist_dropoff_LGA",
            "dist_pickup_EWR",
            "dist_dropoff_EWR",
            "dist_pickup_TSQ",
            "dist_dropoff_TSQ",
            # Rotated Coordinates
            "pickup_rot_x",
            "pickup_rot_y",
            "dropoff_rot_x",
            "dropoff_rot_y",
        ]

    def fit(self, train_df, val_df):
        """
        Trains the model using the provided training dataframe and evaluates on the validation dataframe.

        Args:
            train_df (pd.DataFrame): The training dataset including features and target.
            val_df (pd.DataFrame): The validation dataset for evaluation.
        """
        print("Preparing training data...")
        X_train = train_df[self.feature_names]
        y_train = train_df["fare_amount"]

        print(f"Training HistGradientBoostingRegressor with params: {self.params}")
        # The model handles internal validation for early stopping based on the configuration
        self.model.fit(X_train, y_train)

        print("Evaluating on validation set...")
        X_val = val_df[self.feature_names]
        y_val = val_df["fare_amount"]

        val_preds = self.model.predict(X_val)

        # Calculate RMSE
        mse = mean_squared_error(y_val, val_preds)
        rmse = np.sqrt(mse)

        # Print full precision as required
        print(f"Validation RMSE: {rmse}")

    def predict(self, test_df):
        """
        Generates predictions for the test dataset.

        Args:
            test_df (pd.DataFrame): The test dataset.

        Returns:
            np.ndarray: Array of predicted fare amounts.
        """
        X_test = test_df[self.feature_names]
        return self.model.predict(X_test)

    def save_submission(self, test_df, predictions):
        """
        Formats and saves the predictions to the submission CSV file.

        Args:
            test_df (pd.DataFrame): The test dataset (containing the 'key' column).
            predictions (np.ndarray): The predicted fare amounts.
        """
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

        submission_df = pd.DataFrame(
            {"key": test_df["key"], "fare_amount": predictions}
        )

        print(f"Saving submission to {config.SUBMISSION_PATH}")
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
