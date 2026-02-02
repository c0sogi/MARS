import os
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from library.config import Config
from library.utils import seed_everything, compute_log_loss
from library.feature_engineering import (
    extract_meta_features,
    calculate_uncertainty_stats,
)


class XGBoostBlender:
    """
    Level 2 Meta-Learner using XGBoost.
    Combines predictions from base experts with meta-features and uncertainty statistics
    to produce the final authorship probability.
    """

    def __init__(self, params=None):
        """
        Args:
            params (dict, optional): XGBoost hyperparameters.
                                     Defaults to Config.XGB_PARAMS.
        """
        self.params = params if params is not None else Config.XGB_PARAMS.copy()
        self.model = None
        self.feature_names = None

    def assemble_features(
        self, base_probs_dict, texts, dataset_name, load_cached_data=True
    ):
        """
        Constructs the feature matrix for the meta-learner.

        Features include:
        1. Raw probabilities from each base expert.
        2. Uncertainty statistics (entropy, std, max_prob) for each base expert.
        3. Explicit meta-features (text length, punctuation counts) extracted from raw text.

        Args:
            base_probs_dict (dict): Dictionary mapping model names to probability arrays
                                    (shape: [n_samples, n_classes]).
            texts (list or pd.Series): Raw text samples corresponding to the probabilities.
            dataset_name (str): Identifier for the dataset (e.g., 'val', 'test') for caching.
            load_cached_data (bool): Whether to use cached meta-features.

        Returns:
            np.ndarray: The assembled feature matrix of shape [n_samples, n_features].
        """
        seed_everything()

        # 1. Extract Meta-Features (Length, Punctuation)
        # The extract_meta_features function handles caching internally.
        meta_df = extract_meta_features(
            texts, dataset_name, load_cached_data=load_cached_data
        )

        feature_arrays = [meta_df.values]
        feature_names = list(meta_df.columns)

        # 2. Process Base Model Predictions
        # Sort keys to ensure consistent feature order
        sorted_model_names = sorted(base_probs_dict.keys())

        for model_name in sorted_model_names:
            probs = base_probs_dict[model_name]

            # A. Raw Probabilities
            feature_arrays.append(probs)
            feature_names.extend(
                [f"{model_name}_prob_{c}" for c in range(probs.shape[1])]
            )

            # B. Uncertainty Statistics
            uncertainty_stats = calculate_uncertainty_stats(probs)
            feature_arrays.append(uncertainty_stats)
            feature_names.extend(
                [f"{model_name}_entropy", f"{model_name}_std", f"{model_name}_max_conf"]
            )

        # 3. Concatenate all features
        X = np.hstack(feature_arrays)

        # Store feature names for potential importance analysis
        self.feature_names = feature_names

        return X

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        num_boost_round=1000,
        early_stopping_rounds=50,
    ):
        """
        Trains the XGBoost classifier.

        Args:
            X_train (np.ndarray): Training feature matrix.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray, optional): Validation feature matrix.
            y_val (np.ndarray, optional): Validation labels.
            num_boost_round (int): Maximum number of boosting iterations.
            early_stopping_rounds (int): Rounds of no improvement to trigger early stopping.
        """
        seed_everything()

        print(f"Training XGBoost Meta-Learner with input shape: {X_train.shape}")

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)

        evals = [(dtrain, "train")]
        dval = None

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            evals.append((dval, "validation"))

        # Train
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,  # Suppress per-round printing to keep output clean
        )

        # Report final validation score
        if dval is not None:
            # Predict on validation set to get exact metric
            val_preds = self.model.predict(dval)
            val_loss = compute_log_loss(y_val, val_preds)
            print(f"XGBoost Training Complete. Best Validation Log Loss: {val_loss}")

        return self

    def predict_proba(self, X):
        """
        Generates probability predictions.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        if self.model is None:
            raise RuntimeError("XGBoost model has not been fitted yet.")

        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dtest)

    def save(self, filepath):
        """
        Saves the trained model to disk.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # We use joblib to save the wrapper class to preserve feature_names and params
        # Alternatively, self.model.save_model() saves just the booster
        joblib.dump(self, filepath)
        print(f"Meta-learner saved to {filepath}")

    def load(self, filepath):
        """
        Loads the trained model from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        loaded_obj = joblib.load(filepath)
        self.model = loaded_obj.model
        self.params = loaded_obj.params
        self.feature_names = loaded_obj.feature_names
        print(f"Meta-learner loaded from {filepath}")
        return self
