import os
import pandas as pd
import numpy as np
import xgboost as xgb

from library.config import (
    XGB_PARAMS,
    COUPLING_TYPES,
    EARLY_STOPPING_ROUNDS,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    STRUCTURES_PATH,
    WORKING_DIR,
    FINAL_SUBMISSION_PATH,
)
from library.utils import Timer, seed_everything
from library.feature_pipeline import FeaturePipeline


class StratifiedTrainer:
    def __init__(self):
        """
        Initializes the StratifiedTrainer.
        Sets up the model directory and initializes the feature pipeline.
        """
        self.model_dir = os.path.join(WORKING_DIR, "xgb_models")
        os.makedirs(self.model_dir, exist_ok=True)
        self.pipeline = FeaturePipeline(STRUCTURES_PATH)
        self.models = {}

    def train(self, load_cached_data=True):
        """
        Trains stratified XGBoost models for each coupling type.

        Args:
            load_cached_data (bool): Whether to load pre-computed features from cache.
        """
        seed_everything()

        print("Loading metadata for training...")
        train_meta = pd.read_csv(TRAIN_METADATA_PATH)
        val_meta = pd.read_csv(VAL_METADATA_PATH)

        # Generate Features (Pipeline handles caching)
        with Timer("Feature Generation (Train)"):
            df_train = self.pipeline.generate_features(
                train_meta, "train", load_cached_data=load_cached_data
            )

        with Timer("Feature Generation (Val)"):
            df_val = self.pipeline.generate_features(
                val_meta, "val", load_cached_data=load_cached_data
            )

        metrics = {}

        print("\nStarting Stratified Training...")
        for c_type in COUPLING_TYPES:
            print(f"\n[{c_type}] Preparing Data...")

            # Prepare datasets for specific coupling type
            X_train, y_train = self.pipeline.prepare_data_for_type(df_train, c_type)
            X_val, y_val = self.pipeline.prepare_data_for_type(df_val, c_type)

            if X_train.empty:
                print(f"[{c_type}] No training data found. Skipping.")
                continue

            print(f"[{c_type}] Training Shape: {X_train.shape}")

            # Initialize Model with config parameters
            model = xgb.XGBRegressor(
                **XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS
            )

            # Fit with Early Stopping
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=100,
            )

            # Save Model
            model_path = os.path.join(self.model_dir, f"xgb_{c_type}.json")
            model.save_model(model_path)
            self.models[c_type] = model

            # Record Metric
            # best_score is the score of the best iteration on the last eval set
            best_score = model.best_score
            metrics[c_type] = best_score
            print(f"[{c_type}] Best Validation MAE: {best_score}")

        # Summary Statistics
        print("\n=== Training Summary ===")
        log_maes = []
        for c_type, mae in metrics.items():
            log_mae = np.log(mae)
            log_maes.append(log_mae)
            print(f"{c_type}: MAE={mae:.9f}, LogMAE={log_mae:.9f}")

        if log_maes:
            avg_log_mae = np.mean(log_maes)
            print(f"Average Log MAE: {avg_log_mae:.9f}")
        else:
            print("No models trained.")

    def predict(self, load_cached_data=True):
        """
        Generates predictions for the test set and saves to submission file.

        Args:
            load_cached_data (bool): Whether to load pre-computed features from cache.
        """
        seed_everything()

        print("Loading test metadata...")
        test_meta = pd.read_csv(TEST_METADATA_PATH)

        # Generate Features for Test Set
        with Timer("Feature Generation (Test)"):
            df_test = self.pipeline.generate_features(
                test_meta, "test", load_cached_data=load_cached_data
            )

        prediction_chunks = []

        print("\nStarting Inference...")
        for c_type in COUPLING_TYPES:
            # Check if model exists
            model_path = os.path.join(self.model_dir, f"xgb_{c_type}.json")

            if c_type in self.models:
                model = self.models[c_type]
            elif os.path.exists(model_path):
                # Load model from disk
                model = xgb.XGBRegressor()
                model.load_model(model_path)
            else:
                print(f"[{c_type}] No model found. Skipping.")
                continue

            print(f"[{c_type}] Predicting...")

            # Get IDs for this type to align predictions later
            # We filter the original df_test to get IDs corresponding to the type
            type_subset = df_test[df_test["type"] == c_type]
            if type_subset.empty:
                continue

            ids = type_subset["id"].values

            # Prepare Features
            # prepare_data_for_type returns X with index reset, matching the order of type_subset
            X_test, _ = self.pipeline.prepare_data_for_type(df_test, c_type)

            if X_test.empty:
                continue

            # Predict
            preds = model.predict(X_test)

            # Store predictions with IDs
            chunk = pd.DataFrame({"id": ids, "scalar_coupling_constant": preds})
            prediction_chunks.append(chunk)

        # Combine all chunks
        if prediction_chunks:
            submission = pd.concat(prediction_chunks)
            submission = submission.sort_values("id")

            # Save to submission file
            submission.to_csv(FINAL_SUBMISSION_PATH, index=False)
            print(f"Submission saved to {FINAL_SUBMISSION_PATH}")
            print(f"Total Predictions: {len(submission)}")
        else:
            print("No predictions generated.")
