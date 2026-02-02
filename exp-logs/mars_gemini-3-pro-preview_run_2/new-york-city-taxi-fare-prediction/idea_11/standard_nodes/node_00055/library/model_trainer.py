import os
import gc
import numpy as np
import pandas as pd
import xgboost as xgb
from library.config import Config
from library.data_factory import DataFactory
from library.global_knowledge import KnowledgeBase
from library.feature_engine import MarginCalculator, FeatureEngineer


class ResidualXGBRegressor:
    """
    Implements the Hierarchical Residual Gradient Boosting model.
    Trains on 'residual' (Fare - Base_Margin) and predicts by adding the margin back.
    """

    def __init__(self):
        self.model = None
        self.features = None
        self.knowledge_base = None
        self.feature_engineer = None

    def _get_featurized_data(
        self, split_name, data_loader_func, is_training, load_cached_data=True
    ):
        """
        Loads data, applies feature engineering (including base_margin calculation),
        and handles caching of the fully featurized dataset.
        """
        # Define cache path for the featurized dataset
        cache_filename = f"featurized_{split_name}.parquet"
        cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        # 1. Try loading from cache
        if load_cached_data and os.path.exists(cache_path):
            print(f"Loading cached featurized {split_name} data from {cache_path}...")
            try:
                df = pd.read_parquet(cache_path)
                return df
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print(f"Generating featurized {split_name} data...")

        # Load base data (DataFactory handles its own caching of raw/subsampled data)
        df_base = data_loader_func(load_cached_data=load_cached_data)

        # Apply Feature Engineering
        # This calculates 'base_margin' and 'residual' (if target exists)
        df_feat = self.feature_engineer.process(df_base, is_training=is_training)

        # Save to cache
        print(f"Saving featurized {split_name} data to {cache_path}...")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df_feat.to_parquet(cache_path, index=False)

        return df_feat

    def train(self, load_cached_data=True):
        """
        Trains the XGBoost model on the residual errors.
        """
        print("Initializing Knowledge Base and Feature Engineer...")
        # 1. Build/Load Global Knowledge Base (Priors)
        self.knowledge_base = KnowledgeBase()
        fine_stats, coarse_stats, global_rate = self.knowledge_base.build(
            load_cached_data=load_cached_data
        )

        # 2. Setup Feature Engineer
        margin_calculator = MarginCalculator(fine_stats, coarse_stats, global_rate)
        self.feature_engineer = FeatureEngineer(margin_calculator)

        # 3. Load and Featurize Data
        print("Preparing Training Data...")
        train_df = self._get_featurized_data(
            "train",
            DataFactory.load_train_data,
            is_training=True,
            load_cached_data=load_cached_data,
        )

        print("Preparing Validation Data...")
        val_df = self._get_featurized_data(
            "val",
            DataFactory.load_val_data,
            is_training=False,  # No vector subtraction for val
            load_cached_data=load_cached_data,
        )

        # 4. Define Features
        # Exclude non-numeric, ID, target, and intermediate columns
        exclude_cols = {
            "key",
            "pickup_datetime",
            "fare_amount",
            "base_margin",
            "residual",
            "grid_lat",
            "grid_lon",
        }
        self.features = [c for c in train_df.columns if c not in exclude_cols]

        # Filter only numeric columns just in case
        self.features = [
            c for c in self.features if pd.api.types.is_numeric_dtype(train_df[c])
        ]

        print(f"Training with {len(self.features)} features: {self.features}")

        # 5. Create DMatrix
        # We train on 'residual' to minimize (Target - Base_Margin)
        print("Creating DMatrix objects...")
        X_train = train_df[self.features]
        y_train = train_df["residual"]
        dtrain = xgb.DMatrix(X_train, label=y_train)

        X_val = val_df[self.features]
        y_val = val_df["residual"]
        dval = xgb.DMatrix(X_val, label=y_val)

        # Clean up memory
        del train_df, val_df, X_train, y_train, X_val, y_val
        gc.collect()

        # 6. Train Model
        print("Starting XGBoost training...")
        self.model = xgb.train(
            params=Config.XGB_PARAMS,
            dtrain=dtrain,
            num_boost_round=Config.NUM_BOOST_ROUNDS,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose_eval=Config.VERBOSE_EVAL,
        )

        print(f"Best validation RMSE: {self.model.best_score}")

    def generate_submission(self, load_cached_data=True):
        """
        Generates predictions for the test set and saves the submission file.
        Prediction = Base_Margin + Predicted_Residual
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        print("Preparing Test Data...")
        # Ensure Feature Engineer is ready (if skipping train in this run context)
        if self.feature_engineer is None:
            self.knowledge_base = KnowledgeBase()
            fine_stats, coarse_stats, global_rate = self.knowledge_base.build(
                load_cached_data=load_cached_data
            )
            margin_calculator = MarginCalculator(fine_stats, coarse_stats, global_rate)
            self.feature_engineer = FeatureEngineer(margin_calculator)

        test_df = self._get_featurized_data(
            "test",
            DataFactory.load_test_data,
            is_training=False,
            load_cached_data=load_cached_data,
        )

        print("Generating predictions...")
        X_test = test_df[self.features]
        dtest = xgb.DMatrix(X_test)

        # Predict Residual
        predicted_residual = self.model.predict(dtest)

        # Final Prediction = Base Margin + Residual
        final_pred = test_df["base_margin"].values + predicted_residual

        # Post-processing
        # 1. Clip to minimum fare
        final_pred = np.maximum(final_pred, Config.MIN_FARE_PREDICTION)

        # Create Submission DataFrame
        submission = pd.DataFrame({"key": test_df["key"], "fare_amount": final_pred})

        # Save
        print(f"Saving submission to {Config.SUBMISSION_PATH}...")
        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)

        print("Submission generation complete.")

    def run(self, load_cached_data=True):
        """
        Orchestrates the full pipeline.
        """
        self.train(load_cached_data=load_cached_data)
        self.generate_submission(load_cached_data=load_cached_data)
