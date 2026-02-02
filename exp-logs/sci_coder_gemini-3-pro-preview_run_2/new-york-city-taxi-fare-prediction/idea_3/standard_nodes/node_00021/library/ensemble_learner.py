import os
import numpy as np
import pandas as pd
import xgboost as xgb
from library import config


class EnsembleLearner:
    def __init__(self):
        """
        Initialize the EnsembleLearner with configuration parameters.
        """
        self.xgb_params = config.XGB_PARAMS
        self.ensemble_config = config.ENSEMBLE_CONFIG
        self.working_dir = config.WORKING_DIR
        self.clean_params = config.CLEANING_PARAMS

        # Ensure working directory exists for model artifacts
        os.makedirs(self.working_dir, exist_ok=True)

    def _get_features_and_target(self, df, is_train=True):
        """
        Extracts feature matrix X and target vector y from the dataframe.
        Excludes non-feature columns like 'key' and 'pickup_datetime'.
        """
        # Columns to exclude from features
        exclude_cols = ["key", "pickup_datetime", "fare_amount"]

        # Identify feature columns
        feature_cols = [c for c in df.columns if c not in exclude_cols]

        X = df[feature_cols]
        y = None

        if is_train:
            if "fare_amount" in df.columns:
                y = df["fare_amount"]
            else:
                raise ValueError(
                    "Target column 'fare_amount' missing from training data."
                )

        return X, y, feature_cols

    def train_model(self, subset_df, val_df, model_index):
        """
        Trains a single XGBoost model on a specific data subset.

        Args:
            subset_df (pd.DataFrame): The training subset.
            val_df (pd.DataFrame): The validation set.
            model_index (int): Index of the model in the ensemble.

        Returns:
            float: Best validation RMSE for this model.
        """
        print(f"Preparing data for Model {model_index}...")

        # Prepare Data
        X_train, y_train, feat_names = self._get_features_and_target(
            subset_df, is_train=True
        )
        X_val, y_val, _ = self._get_features_and_target(val_df, is_train=True)

        # Create DMatrix
        # nthread is set via n_jobs in params, but DMatrix can also take it.
        # We rely on global config params passed to train.
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feat_names)

        # Watchlist for monitoring
        watchlist = [(dtrain, "train"), (dval, "eval")]

        print(f"Training Model {model_index} on {len(subset_df)} rows...")

        # Train
        model = xgb.train(
            params=self.xgb_params,
            dtrain=dtrain,
            num_boost_round=self.xgb_params.get("n_estimators", 1000),
            evals=watchlist,
            early_stopping_rounds=self.xgb_params.get("early_stopping_rounds", 50),
            verbose_eval=False,
        )

        # Save Model
        model_path = os.path.join(self.working_dir, f"model_{model_index}.json")
        model.save_model(model_path)
        print(f"Model {model_index} saved to {model_path}")

        # Retrieve best score
        # attribute name depends on objective, usually 'best_score'
        best_score = model.best_score
        print(f"Model {model_index} Best Validation RMSE: {best_score}")

        return best_score

    def train_ensemble_loop(self, subsets, val_df):
        """
        Iterates through subsets and trains the ensemble.

        Args:
            subsets (list of pd.DataFrame): List of training data partitions.
            val_df (pd.DataFrame): Validation set.
        """
        n_models = len(subsets)
        print(f"Starting Ensemble Training: {n_models} models scheduled.")

        scores = []

        for i, subset in enumerate(subsets):
            print(f"\n--- Training Ensemble Model {i+1}/{n_models} ---")
            score = self.train_model(subset, val_df, i)
            scores.append(score)

        print("\n=== Ensemble Training Complete ===")
        print(f"Individual Model RMSEs: {scores}")
        print(f"Average Ensemble RMSE (Validation): {np.mean(scores)}")

    def predict_ensemble(self, test_df):
        """
        Loads all trained models, predicts on test set, and averages results.

        Args:
            test_df (pd.DataFrame): Test data.

        Returns:
            np.ndarray: Aggregated predictions.
        """
        n_models = self.ensemble_config["n_models"]
        print(f"Predicting with Ensemble of {n_models} models...")

        # Prepare Test Data
        X_test, _, feat_names = self._get_features_and_target(test_df, is_train=False)
        dtest = xgb.DMatrix(X_test, feature_names=feat_names)

        # Array to store sum of predictions
        total_preds = np.zeros(len(test_df))
        models_loaded = 0

        for i in range(n_models):
            model_path = os.path.join(self.working_dir, f"model_{i}.json")

            if not os.path.exists(model_path):
                print(f"Warning: {model_path} not found. Skipping.")
                continue

            # Load model
            booster = xgb.Booster()
            booster.load_model(model_path)

            # Predict
            preds = booster.predict(dtest)
            total_preds += preds
            models_loaded += 1

        if models_loaded == 0:
            raise RuntimeError("No models were loaded for prediction!")

        # Average
        avg_preds = total_preds / models_loaded

        # Post-processing: Apply Minimum Fare Floor
        min_floor = self.clean_params.get("min_fare_floor", 2.50)
        avg_preds = np.maximum(avg_preds, min_floor)

        return avg_preds

    def generate_submission(self, test_df, output_path):
        """
        Generates predictions and saves the submission CSV.

        Args:
            test_df (pd.DataFrame): Test data.
            output_path (str): Path to save CSV.
        """
        print("Generating Submission...")

        predictions = self.predict_ensemble(test_df)

        submission = pd.DataFrame({"key": test_df["key"], "fare_amount": predictions})

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        submission.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path}")
        print(f"Head:\n{submission.head()}")
