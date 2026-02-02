import os
import numpy as np
import pandas as pd
import xgboost as xgb
from library.config import XGB_PARAMS, TRAIN_CONFIG, PATH_CONFIG
from library.data_processor import TaxiDataProcessor


class XGBTrainer:
    def __init__(self):
        self.processor = TaxiDataProcessor()
        self.model = None

    def train(self, load_cached_data=True):
        """
        Trains the XGBoost model with early stopping and saves the artifact.
        """
        # Load processed data
        train_df = self.processor.get_processed_data(
            "train", load_cached_data=load_cached_data
        )
        val_df = self.processor.get_processed_data(
            "val", load_cached_data=load_cached_data
        )

        # Prepare features and targets
        # Drop 'key' as it is an identifier, and 'fare_amount' as it is the target
        drop_cols = ["key", "fare_amount"]

        # Sanitize target variable in the base trainer
        # Cite solution_lesson_node_00017: Target Variable Sanitization is Critical
        # We clamp to a reasonable range [2.5, 500] to prevent L2 loss explosion
        train_df = train_df[
            (train_df["fare_amount"] >= 2.5) & (train_df["fare_amount"] < 500)
        ]

        X_train = train_df.drop(columns=drop_cols)
        y_train = train_df["fare_amount"]

        X_val = val_df.drop(columns=drop_cols)
        y_val = val_df["fare_amount"]

        # Initialize XGBoost Regressor
        # Map num_boost_round to n_estimators for sklearn API
        params = XGB_PARAMS.copy()
        params["n_estimators"] = TRAIN_CONFIG["num_boost_round"]
        params["early_stopping_rounds"] = TRAIN_CONFIG["early_stopping_rounds"]

        self.model = xgb.XGBRegressor(**params)

        # Train the model
        print("Starting training...")
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=TRAIN_CONFIG["verbose_eval"],
        )

        # Save the model
        model_path = PATH_CONFIG["model_save_path"]
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save_model(model_path)
        print(f"Model saved to {model_path}")

        # Validate and print metrics
        # If early stopping was triggered, predict uses the best iteration by default
        preds = self.model.predict(X_val)
        rmse = np.sqrt(np.mean((y_val - preds) ** 2))
        print(f"Validation RMSE: {rmse}")

    def generate_submission(self, load_cached_data=True):
        """
        Generates predictions for the test set and saves the submission file.
        """
        # Load test data
        test_df = self.processor.get_processed_data(
            "test", load_cached_data=load_cached_data
        )

        # Prepare features (drop key)
        X_test = test_df.drop(columns=["key"])

        # Load model if not present in memory
        if self.model is None:
            model_path = PATH_CONFIG["model_save_path"]
            if not os.path.exists(model_path):
                raise FileNotFoundError("Model not found. Train the model first.")

            # Re-initialize model structure to load weights
            params = XGB_PARAMS.copy()
            params["n_estimators"] = TRAIN_CONFIG["num_boost_round"]
            self.model = xgb.XGBRegressor(**params)
            self.model.load_model(model_path)

        # Predict
        preds = self.model.predict(X_test)

        # Post-processing: Apply lower bound of $2.50
        preds = np.maximum(preds, 2.50)

        # Create submission dataframe
        submission = pd.DataFrame({"key": test_df["key"], "fare_amount": preds})

        # Save submission
        output_path = PATH_CONFIG["submission_output"]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
