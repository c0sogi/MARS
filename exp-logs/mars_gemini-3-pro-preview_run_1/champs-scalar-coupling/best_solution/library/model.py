import numpy as np
import pandas as pd
import xgboost as xgb
import os
import json
import gc
from library.config import (
    XGB_PARAMS,
    COUPLING_TYPES,
    WORKING_DIR,
    RANDOM_STATE,
)
from library.utils import calculate_competition_metric


class StratifiedEnsemble:
    """
    A Stratified Ensemble of XGBoost Regressors.
    Trains a separate model for each scalar coupling type to capture distinct physical regimes.
    """

    def __init__(self):
        self.models = {}
        self.feature_metadata = {}
        self.model_dir = os.path.join(WORKING_DIR, "xgb_models")
        os.makedirs(self.model_dir, exist_ok=True)

    def _get_model_path(self, coupling_type):
        return os.path.join(self.model_dir, f"{coupling_type}.json")

    def _get_feature_path(self, coupling_type):
        return os.path.join(self.model_dir, f"{coupling_type}_features.json")

    def fit(self, train_df, val_df):
        """
        Trains the stratified ensemble.

        Args:
            train_df (pd.DataFrame): Training data with features and target.
            val_df (pd.DataFrame): Validation data with features and target.

        Returns:
            pd.DataFrame: Validation DataFrame with 'prediction' column added.
        """
        print(f"Starting Stratified Training on {len(COUPLING_TYPES)} types...")

        # Container for validation predictions
        val_preds = []

        # Columns to exclude from features
        exclude_cols = [
            "id",
            "molecule_name",
            "scalar_coupling_constant",
            "type",
            "file_path",
            "prediction",  # In case it exists
        ]

        for c_type in COUPLING_TYPES:
            print(f"\n=== Training Stratum: {c_type} ===")

            # 1. Stratify Data
            train_subset = train_df[train_df["type"] == c_type].copy()
            val_subset = val_df[val_df["type"] == c_type].copy()

            if len(train_subset) == 0:
                print(f"Warning: No training data for {c_type}. Skipping.")
                continue

            # 2. Feature Selection & Sanitization
            # Identify potential features
            candidates = [c for c in train_subset.columns if c not in exclude_cols]

            # Remove constant columns in this stratum
            # (e.g., if 'is_F' is always 0 for 1JHC)
            std = train_subset[candidates].std()
            active_features = std[std > 0].index.tolist()

            dropped_count = len(candidates) - len(active_features)
            print(
                f"Features: {len(active_features)} active ({dropped_count} constant dropped)"
            )

            # Save feature list for inference
            self.feature_metadata[c_type] = active_features
            with open(self._get_feature_path(c_type), "w") as f:
                json.dump(active_features, f)

            # 3. Prepare X and y
            X_train = train_subset[active_features]
            y_train = train_subset["scalar_coupling_constant"]
            X_val = val_subset[active_features]
            y_val = val_subset["scalar_coupling_constant"]

            # 4. Initialize and Train Model
            # Note: early_stopping_rounds is in XGB_PARAMS
            model = xgb.XGBRegressor(**XGB_PARAMS)

            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            # 5. Evaluate and Store
            best_iteration = model.best_iteration
            best_score = model.best_score
            print(f"Best Iteration: {best_iteration} | Best MAE: {best_score}")

            # Save Model
            model.save_model(self._get_model_path(c_type))
            self.models[c_type] = model

            # Generate predictions for validation set (using best iteration)
            # Note: predict uses best_iteration automatically if early stopping was used
            preds = model.predict(X_val)
            val_subset["prediction"] = preds
            val_preds.append(val_subset)

            # Cleanup to save memory
            del train_subset, val_subset, X_train, y_train, X_val, y_val, model
            gc.collect()

        # Concatenate all validation predictions
        full_val_preds = pd.concat(val_preds, axis=0)

        # Calculate Overall Metric
        print("\n=== Global Validation Evaluation ===")
        metric = calculate_competition_metric(
            full_val_preds,
            prediction_col="prediction",
            target_col="scalar_coupling_constant",
            type_col="type",
        )
        print(f"Final Log MAE Score: {metric}")

        return full_val_preds

    def predict(self, test_df):
        """
        Generates predictions for the test set using the trained stratified models.

        Args:
            test_df (pd.DataFrame): Test data with features.

        Returns:
            pd.DataFrame: DataFrame with 'id' and 'scalar_coupling_constant' columns.
        """
        print("Starting Inference...")
        results = []

        # Ensure test_df has the type column
        if "type" not in test_df.columns:
            raise ValueError(
                "Test DataFrame must contain 'type' column for stratified inference."
            )

        unique_types = test_df["type"].unique()

        for c_type in unique_types:
            # Check if we have a model for this type
            model_path = self._get_model_path(c_type)
            feature_path = self._get_feature_path(c_type)

            if not os.path.exists(model_path) or not os.path.exists(feature_path):
                print(f"Warning: No model found for {c_type}. Filling with 0.")
                subset = test_df[test_df["type"] == c_type].copy()
                subset["scalar_coupling_constant"] = 0.0
                results.append(subset[["id", "scalar_coupling_constant"]])
                continue

            # Load features
            with open(feature_path, "r") as f:
                active_features = json.load(f)

            # Load model
            model = xgb.XGBRegressor()
            model.load_model(model_path)

            # Prepare Data
            subset = test_df[test_df["type"] == c_type].copy()

            # Ensure all features exist (fill missing with 0 if any, though shouldn't happen)
            # and select only active features in correct order
            X_test = subset[active_features]

            # Predict
            preds = model.predict(X_test)
            subset["scalar_coupling_constant"] = preds

            results.append(subset[["id", "scalar_coupling_constant"]])

            # Cleanup
            del model, subset, X_test
            gc.collect()

        # Combine results
        final_submission = pd.concat(results, axis=0).sort_values("id")
        return final_submission
