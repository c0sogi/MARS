import os
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from library.config import ProjectConfig
from library.data_pipeline import DataPipeline


class ModelTrainer:
    def __init__(self):
        self.config = ProjectConfig
        self.pipeline = DataPipeline()
        self.model = None
        self.features = None
        self.cache_dir = self.config.CACHE_DIR

        # Ensure working directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def train_model(self, load_cached_data: bool = True):
        """
        Orchestrates the training process:
        1. Loads processed data (Train, Val, Test) via DataPipeline.
        2. Prepares DMatrices for XGBoost.
        3. Trains the model with Early Stopping.
        4. Saves the trained model.
        5. Generates predictions on the Test set.
        """
        print("Initializing data pipeline...")
        train_df, val_df, test_df = self.pipeline.get_data(load_cached=load_cached_data)

        # Define input features
        # Exclude metadata, target, and intermediate columns like 'fold'
        exclude_cols = {"key", "fare_amount", "pickup_datetime", "fold"}
        self.features = [c for c in train_df.columns if c not in exclude_cols]

        print(f"Input Features ({len(self.features)}): {self.features}")

        # Prepare DMatrices
        # Using 'enable_categorical=False' as we treat geohashes/integers as numeric/ordinal
        print("Creating DMatrices...")
        dtrain = xgb.DMatrix(train_df[self.features], label=train_df["fare_amount"])
        dval = xgb.DMatrix(val_df[self.features], label=val_df["fare_amount"])

        # Clean up DataFrames to free memory
        del train_df, val_df
        gc.collect()

        # Training Configuration
        params = self.config.XGB_PARAMS.copy()
        num_boost_round = params.pop(
            "n_estimators", 3000
        )  # Extract n_estimators if present in dict

        print("Starting XGBoost training...")
        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=self.config.EARLY_STOPPING_ROUNDS,
            verbose_eval=self.config.VERBOSE_EVAL,
        )

        # Print Best Score
        print(f"Best validation score (RMSE): {self.model.best_score}")

        # Save Model
        model_path = os.path.join(self.cache_dir, "xgb_model.json")
        self.model.save_model(model_path)
        print(f"Model saved to {model_path}")

        # Feature Importance Analysis
        importance = self.model.get_score(importance_type="gain")
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 20 Feature Importances (Gain):")
        for f, score in sorted_importance[:20]:
            print(f"{f}: {score}")

        # Proceed to Prediction
        self.predict(test_df)

    def predict(self, test_df: pd.DataFrame):
        """
        Generates predictions for the test set and saves the submission file.
        """
        print("\nGenerating predictions for Test set...")

        # Ensure model is loaded
        if self.model is None:
            model_path = os.path.join(self.cache_dir, "xgb_model.json")
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}...")
                self.model = xgb.Booster()
                self.model.load_model(model_path)
            else:
                raise RuntimeError("Model not found. Please train the model first.")

        # Ensure features match training
        if self.features is None:
            # Fallback: infer features from test columns excluding key/pickup_datetime
            # This assumes test_df has exactly the features used in training + key/datetime
            exclude_cols = {"key", "pickup_datetime"}
            self.features = [c for c in test_df.columns if c not in exclude_cols]

        # Create DMatrix for Test
        dtest = xgb.DMatrix(test_df[self.features])

        # Predict
        predictions = self.model.predict(dtest)

        # Post-Processing: Apply minimum fare floor
        predictions = np.maximum(predictions, self.config.PRED_MIN_FARE)

        # Create Submission DataFrame
        submission = pd.DataFrame({"key": test_df["key"], "fare_amount": predictions})

        # Save Submission
        submission_path = os.path.join(self.config.SUBMISSION_DIR, "submission.csv")
        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
        print("Sample predictions:")
        print(submission.head())
