import os
import logging
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from library.config import Config

# Attempt to import CatBoost, handle gracefully if missing
try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class TriEnsemble:
    """
    Unified Heterogeneous Tri-Ensemble Model.
    Wraps LightGBM, XGBoost, and CatBoost (if available) with a consistent API.
    """

    def __init__(self, config=Config):
        self.config = config
        self.models = {}
        self.logger = logging.getLogger(__name__)

        # Define metadata columns to exclude from features
        self.metadata_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
            "datetime",
            "video_path_endzone",
            "video_path_sideline",
            "video_path_all29",
        ]

    def _get_feature_cols(self, df):
        """Identifies feature columns by excluding metadata."""
        return [c for c in df.columns if c not in self.metadata_cols]

    def fit(self, df_train, df_val=None, model_names=["lgbm", "xgb", "cat"]):
        """
        Trains the specified models in the ensemble.

        Args:
            df_train (pd.DataFrame): Training data with features and 'contact' column.
            df_val (pd.DataFrame, optional): Validation data.
            model_names (list): List of model keys to train ('lgbm', 'xgb', 'cat').
        """
        feature_cols = self._get_feature_cols(df_train)
        X_train = df_train[feature_cols]
        y_train = df_train["contact"]

        X_val = None
        y_val = None
        if df_val is not None:
            X_val = df_val[feature_cols]
            y_val = df_val["contact"]

        self.logger.info(
            f"Training on {len(feature_cols)} features: {feature_cols[:5]}..."
        )

        # --- LightGBM ---
        if "lgbm" in model_names:
            self.logger.info("Training LightGBM...")
            params = self.config.LGBM_PARAMS.copy()
            n_estimators = params.pop("n_estimators", 1000)
            early_stopping_rounds = params.pop("early_stopping_rounds", 50)

            dtrain = lgb.Dataset(X_train, label=y_train)
            valid_sets = [dtrain]
            if X_val is not None:
                dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
                valid_sets.append(dval)

            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds, verbose=False
                ),
                lgb.log_evaluation(period=0),  # Silent
            ]

            self.models["lgbm"] = lgb.train(
                params,
                dtrain,
                num_boost_round=n_estimators,
                valid_sets=valid_sets,
                callbacks=callbacks,
            )

        # --- XGBoost ---
        if "xgb" in model_names:
            self.logger.info("Training XGBoost...")
            params = self.config.XGB_PARAMS.copy()
            n_estimators = params.pop("n_estimators", 1000)
            early_stopping_rounds = params.pop("early_stopping_rounds", 50)

            dtrain = xgb.DMatrix(X_train, label=y_train)
            evals = [(dtrain, "train")]
            if X_val is not None:
                dval = xgb.DMatrix(X_val, label=y_val)
                evals.append((dval, "eval"))

            self.models["xgb"] = xgb.train(
                params,
                dtrain,
                num_boost_round=n_estimators,
                evals=evals,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=False,
            )

        # --- CatBoost ---
        if "cat" in model_names:
            if CATBOOST_AVAILABLE:
                self.logger.info("Training CatBoost...")
                params = self.config.CAT_PARAMS.copy()
                # CatBoostClassifier handles fit differently
                model = CatBoostClassifier(**params)

                eval_set = None
                if X_val is not None:
                    eval_set = (X_val, y_val)

                model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
                self.models["cat"] = model
            else:
                self.logger.warning(
                    "CatBoost requested but not installed/importable. Skipping."
                )

    def predict_proba(self, df):
        """
        Generates averaged probability predictions from all trained models.

        Args:
            df (pd.DataFrame): Data to predict on.

        Returns:
            np.ndarray: Array of probabilities (Class 1).
        """
        feature_cols = self._get_feature_cols(df)
        X = df[feature_cols]

        predictions = []

        # LightGBM
        if "lgbm" in self.models:
            # lgb.train.predict returns raw probabilities for binary
            pred = self.models["lgbm"].predict(X)
            predictions.append(pred)

        # XGBoost
        if "xgb" in self.models:
            dtest = xgb.DMatrix(X)
            pred = self.models["xgb"].predict(dtest)
            predictions.append(pred)

        # CatBoost
        if "cat" in self.models:
            # predict_proba returns [prob_0, prob_1]
            pred = self.models["cat"].predict_proba(X)[:, 1]
            predictions.append(pred)

        if not predictions:
            self.logger.warning("No models trained! Returning zeros.")
            return np.zeros(len(df))

        # Unweighted Average Ensemble
        return np.mean(predictions, axis=0)

    def save_models(self, path_dict):
        """
        Saves trained models to disk.

        Args:
            path_dict (dict): Map of model_key ('lgbm', 'xgb', 'cat') to file path.
        """
        for key, path in path_dict.items():
            if key in self.models:
                self.logger.info(f"Saving {key} model to {path}")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                joblib.dump(self.models[key], path)

    def load_models(self, path_dict):
        """
        Loads models from disk.

        Args:
            path_dict (dict): Map of model_key ('lgbm', 'xgb', 'cat') to file path.
        """
        for key, path in path_dict.items():
            if os.path.exists(path):
                self.logger.info(f"Loading {key} model from {path}")
                self.models[key] = joblib.load(path)
            else:
                self.logger.warning(f"Model file for {key} not found at {path}")
