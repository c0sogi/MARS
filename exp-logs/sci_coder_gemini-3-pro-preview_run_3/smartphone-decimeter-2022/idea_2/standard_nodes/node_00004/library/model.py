import lightgbm as lgb
import pandas as pd
import numpy as np
import os
from library.config import LGBM_PARAMS, EARLY_STOPPING_ROUNDS, VERBOSE_EVAL, SEED


class LGBMRegressorWrapper:
    def __init__(self, n_estimators=None, **kwargs):
        """
        Wrapper for LightGBM Regressor to handle Latitude and Longitude error prediction.

        Args:
            n_estimators (int, optional): Number of boosting iterations. Overrides config if provided.
            **kwargs: Additional hyperparameters to override defaults in LGBM_PARAMS.
        """
        self.params = LGBM_PARAMS.copy()
        if n_estimators is not None:
            self.params["n_estimators"] = n_estimators

        # Allow overriding other params via kwargs
        self.params.update(kwargs)

        # Ensure random state is set for reproducibility
        self.params["random_state"] = SEED

        # Initialize separate models for Latitude and Longitude
        self.model_lat = lgb.LGBMRegressor(**self.params)
        self.model_lon = lgb.LGBMRegressor(**self.params)

    def train(self, X_train, y_train, X_val, y_val, max_samples=None):
        """
        Trains the Latitude and Longitude models.

        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.DataFrame): Training targets (must contain 'lat_error', 'lon_error').
            X_val (pd.DataFrame): Validation features.
            y_val (pd.DataFrame): Validation targets.
            max_samples (int, optional): Limit training data size for debugging.
        """
        # Apply sampling if requested for debugging
        if max_samples is not None and max_samples < len(X_train):
            print(f"Subsampling training data to {max_samples} rows.")
            X_train = X_train.iloc[:max_samples]
            y_train = y_train.iloc[:max_samples]

        # Define callbacks for early stopping and logging
        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=VERBOSE_EVAL),
        ]

        print("\n" + "=" * 40)
        print("Training Latitude Model")
        print("=" * 40)
        self.model_lat.fit(
            X_train,
            y_train["lat_error"],
            eval_set=[(X_train, y_train["lat_error"]), (X_val, y_val["lat_error"])],
            eval_names=["Train", "Valid"],
            eval_metric="mae",
            callbacks=callbacks,
        )

        print("\n" + "=" * 40)
        print("Training Longitude Model")
        print("=" * 40)
        self.model_lon.fit(
            X_train,
            y_train["lon_error"],
            eval_set=[(X_train, y_train["lon_error"]), (X_val, y_val["lon_error"])],
            eval_names=["Train", "Valid"],
            eval_metric="mae",
            callbacks=callbacks,
        )

    def predict(self, X_test):
        """
        Generates predictions for Latitude and Longitude errors.

        Args:
            X_test (pd.DataFrame): Test features.

        Returns:
            tuple: (pred_lat_error, pred_lon_error) as numpy arrays.
        """
        pred_lat = self.model_lat.predict(X_test)
        pred_lon = self.model_lon.predict(X_test)
        return pred_lat, pred_lon


def generate_submission(df_test, lat_preds, lon_preds, output_path):
    """
    Applies predicted residuals to baseline WLS positions and saves the submission.

    Args:
        df_test (pd.DataFrame): Test dataframe containing metadata and baseline 'lat_wls', 'lon_wls'.
        lat_preds (np.array): Predicted Latitude errors.
        lon_preds (np.array): Predicted Longitude errors.
        output_path (str): Path to save the CSV.
    """
    # Create a copy to avoid modifying original
    sub_df = df_test.copy()

    # Apply corrections: GT = WLS + Error
    sub_df["LatitudeDegrees"] = sub_df["lat_wls"] + lat_preds
    sub_df["LongitudeDegrees"] = sub_df["lon_wls"] + lon_preds

    # Keep only required columns for submission
    cols = ["tripId", "UnixTimeMillis", "LatitudeDegrees", "LongitudeDegrees"]
    submission = sub_df[cols]

    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
