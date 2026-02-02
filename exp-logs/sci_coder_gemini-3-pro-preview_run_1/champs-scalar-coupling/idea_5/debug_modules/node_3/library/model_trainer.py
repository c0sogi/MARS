import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from library import config
from library import utils


class StratifiedModelManager:
    """
    Manages the training and inference of a Stratified Ensemble of XGBoost models.
    Implements the Multi-Hop Topological Stratified Ensemble strategy.
    """

    def __init__(self, verbose=True):
        """
        Initialize the model manager.

        Args:
            verbose (bool): Whether to print progress and metrics.
        """
        self.verbose = verbose
        # Directory to save trained models and feature lists
        self.model_dir = os.path.join(config.WORKING_DIR, "xgb_models")
        os.makedirs(self.model_dir, exist_ok=True)

    def _get_base_features(self, df):
        """
        Identify potential feature columns by excluding metadata.

        Args:
            df (pd.DataFrame): The dataframe containing all columns.

        Returns:
            list: List of column names to be considered as features.
        """
        # Metadata columns to exclude from training features
        # Note: atom_index_0/1 are excluded to force reliance on topological features
        meta_cols = [
            "id",
            "molecule_name",
            "scalar_coupling_constant",
            "type",
            "file_path",
            "atom_index_0",
            "atom_index_1",
        ]
        return [c for c in df.columns if c not in meta_cols]

    def train_all_types(self, train_df, val_df):
        """
        Trains a separate XGBoost model for each coupling type using the provided data.

        Args:
            train_df (pd.DataFrame): Training data.
            val_df (pd.DataFrame): Validation data.

        Returns:
            dict: A dictionary mapping coupling types to their validation Log MAE scores.
        """
        scores = {}

        if self.verbose:
            print(
                f"Starting Stratified Training on {len(config.COUPLING_TYPES)} types..."
            )

        for coupling_type in config.COUPLING_TYPES:
            if self.verbose:
                print(f"\n{'='*40}\nProcessing Type: {coupling_type}\n{'='*40}")

            # 1. Filter Data for the specific coupling type
            train_subset = train_df[train_df["type"] == coupling_type].reset_index(
                drop=True
            )
            val_subset = val_df[val_df["type"] == coupling_type].reset_index(drop=True)

            if len(train_subset) == 0:
                print(f"Warning: No training data for {coupling_type}. Skipping.")
                continue

            # 2. Dynamic Feature Selection
            # Identify all potential feature columns
            candidates = self._get_base_features(train_subset)

            # Remove constant features specific to this stratum
            # (e.g., for 1JHC, atom_0 type is always C, so remove it)
            X_train_candidates = train_subset[candidates]
            std_devs = X_train_candidates.std()

            # Keep only features with non-zero variance
            active_features = std_devs[std_devs > 0].index.tolist()

            if self.verbose:
                n_dropped = len(candidates) - len(active_features)
                print(
                    f"Features selected: {len(active_features)} (Dropped {n_dropped} constant features)"
                )

            # Prepare Training and Validation matrices
            X_train = train_subset[active_features]
            y_train = train_subset["scalar_coupling_constant"]

            X_val = val_subset[active_features]
            y_val = val_subset["scalar_coupling_constant"]

            # 3. Initialize Model
            # Using parameters from config designed for high capacity
            model = xgb.XGBRegressor(
                **config.XGB_PARAMS,
                early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
            )

            # 4. Train with Early Stopping
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=config.VERBOSE_EVAL,
            )

            # 5. Save Model and Feature Metadata
            # We must save the active_features list to ensure consistent inference
            model_path = os.path.join(self.model_dir, f"xgb_{coupling_type}.joblib")
            features_path = os.path.join(
                self.model_dir, f"features_{coupling_type}.json"
            )

            joblib.dump(model, model_path)
            with open(features_path, "w") as f:
                json.dump(active_features, f)

            if self.verbose:
                print(f"Model saved to {model_path}")

            # 6. Evaluate
            # Predict on validation set
            preds = model.predict(X_val)

            # Calculate Log MAE for this type
            score = utils.calculate_log_mae(y_val, preds, val_subset["type"])
            scores[coupling_type] = score

            print(f"Type {coupling_type} Log MAE: {score}")

        # Summary Statistics
        if scores:
            avg_score = np.mean(list(scores.values()))
            print(f"\nTraining Complete.")
            print(f"Average Log MAE across types: {avg_score}")

        return scores

    def predict_all_types(self, test_df):
        """
        Generates predictions for the test set using the trained stratified models.

        Args:
            test_df (pd.DataFrame): Test data containing features.

        Returns:
            pd.DataFrame: A dataframe with 'id' and 'scalar_coupling_constant' columns.
        """
        all_predictions = []

        if self.verbose:
            print(f"Starting Stratified Inference...")

        for coupling_type in config.COUPLING_TYPES:
            # Filter test data for the current type
            test_subset = test_df[test_df["type"] == coupling_type].copy()

            if len(test_subset) == 0:
                continue

            # Define paths
            model_path = os.path.join(self.model_dir, f"xgb_{coupling_type}.joblib")
            features_path = os.path.join(
                self.model_dir, f"features_{coupling_type}.json"
            )

            # Check if model exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Model for {coupling_type} not found at {model_path}. Train models first."
                )

            # Load Model and Feature List
            model = joblib.load(model_path)
            with open(features_path, "r") as f:
                active_features = json.load(f)

            # Prepare Input Features
            # Ensure we strictly use the features the model was trained on
            # If a column is missing in test (unlikely), fill with 0
            for feat in active_features:
                if feat not in test_subset.columns:
                    test_subset[feat] = 0

            X_test = test_subset[active_features]

            # Generate Predictions
            preds = model.predict(X_test)

            # Store Results
            result_df = pd.DataFrame(
                {"id": test_subset["id"], "scalar_coupling_constant": preds}
            )
            all_predictions.append(result_df)

            if self.verbose:
                print(f"Predicted {len(preds)} samples for {coupling_type}")

        # Combine all predictions
        if not all_predictions:
            raise ValueError(
                "No predictions generated. Check if test data matches coupling types."
            )

        final_submission = pd.concat(all_predictions, axis=0)

        # Sort by ID to match submission format requirements
        final_submission = final_submission.sort_values("id")

        return final_submission
