import os
import numpy as np
import pandas as pd
import xgboost as xgb
from library.config import Config


class TaxiFareModel:
    """
    Wrapper for the XGBoost Regressor model.
    Handles feature selection, training with early stopping, model persistence,
    and prediction with post-processing.
    """

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.feature_names = None

    def _prepare_features(self, df, is_train=True):
        """
        Prepares the feature matrix X and target vector y from the DataFrame.
        Drops metadata columns and string keys used for joining priors.
        """
        # Columns to exclude from the feature set
        # key: ID
        # pickup_datetime: Raw timestamp (features extracted)
        # fare_amount: Target
        # key_*: String keys for multi-view encoding (stats already joined)
        exclude_cols = [
            "key",
            "pickup_datetime",
            "fare_amount",
            "key_fine",
            "key_coarse",
            "key_temporal",
        ]

        # Identify feature columns (all numeric columns that are not excluded)
        # We rely on the pipeline to ensure only valid features remain or are added.
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols]

        y = None
        if is_train:
            if "fare_amount" in df.columns:
                y = df["fare_amount"]
            else:
                raise ValueError("Target 'fare_amount' missing from training data.")

        return X, y, feature_cols

    def train(self, train_df, val_df):
        """
        Trains the XGBoost model using the provided training and validation sets.
        Implements Early Stopping and saves the model to disk.
        """
        print("Preparing training and validation data...")
        X_train, y_train, features = self._prepare_features(train_df, is_train=True)
        X_val, y_val, _ = self._prepare_features(val_df, is_train=True)

        self.feature_names = features
        print(f"Training with {len(features)} features: {features}")
        print(f"Training set size: {len(X_train)}, Validation set size: {len(X_val)}")

        # Initialize XGBoost Regressor with parameters from config
        # XGB_PARAMS includes 'early_stopping_rounds', 'tree_method', 'device', etc.
        self.model = xgb.XGBRegressor(**self.config.XGB_PARAMS)

        print("Starting XGBoost training...")
        # Fit the model
        # eval_set is required for early stopping and metric tracking
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=100,  # Print evaluation metrics every 100 rounds
        )

        # Retrieve and print the final validation metric
        results = self.model.evals_result()
        if results and "validation_1" in results:
            # validation_1 corresponds to X_val
            final_rmse = results["validation_1"]["rmse"][-1]
            print(f"Final Validation RMSE: {final_rmse}")

        # Save the trained model to the working directory
        model_path = self.config.get_cache_path("xgb_model.json")
        self.model.save_model(model_path)
        print(f"Model saved to {model_path}")

    def predict(self, test_df):
        """
        Generates predictions for the test set.
        Loads the model from disk if not already in memory.
        Applies post-processing (min fare floor).
        """
        model_path = self.config.get_cache_path("xgb_model.json")

        # Load model if necessary
        if self.model is None:
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}...")
                self.model = xgb.XGBRegressor(**self.config.XGB_PARAMS)
                self.model.load_model(model_path)
            else:
                raise ValueError(
                    "Model has not been trained and no cached model found."
                )

        print("Preparing test features...")
        X_test, _, feature_cols_test = self._prepare_features(test_df, is_train=False)

        # Align features with training set
        # If self.feature_names is known (from train), ensure X_test matches
        if self.feature_names:
            # Add missing columns as NaN (XGBoost handles missing values)
            missing_cols = set(self.feature_names) - set(X_test.columns)
            for c in missing_cols:
                X_test[c] = np.nan

            # Reorder columns to match training order
            X_test = X_test[self.feature_names]
        else:
            # If loaded from disk without training in this session, we assume
            # the pipeline produces consistent columns.
            pass

        print("Generating predictions...")
        predictions = self.model.predict(X_test)

        # Post-Processing
        # Apply minimum fare floor ($2.50) as per requirements
        # This also implicitly handles the "non-negative" requirement
        print("Applying post-processing (clamping to min $2.50)...")
        predictions = np.maximum(predictions, 2.50)

        return predictions

    def generate_submission(self, test_df, predictions):
        """
        Formats the predictions into the required submission CSV format
        and saves it to the configured output path.
        """
        print("Generating submission file...")

        if len(test_df) != len(predictions):
            raise ValueError(
                f"Length mismatch: Test DF has {len(test_df)} rows, Predictions has {len(predictions)}."
            )

        # Create submission DataFrame
        submission = pd.DataFrame({"key": test_df["key"], "fare_amount": predictions})

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.config.FINAL_SUBMISSION_PATH), exist_ok=True)

        # Save to CSV
        submission.to_csv(self.config.FINAL_SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.FINAL_SUBMISSION_PATH}")

        # Print first few rows for verification
        print(submission.head())
