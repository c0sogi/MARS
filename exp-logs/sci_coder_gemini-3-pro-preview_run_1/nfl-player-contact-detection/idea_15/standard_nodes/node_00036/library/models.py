import os
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library import config, utils

# Columns to exclude from training features
METADATA_COLS = [
    "contact_id",
    "game_play",
    "step",
    "nfl_player_id_1",
    "nfl_player_id_2",
    "contact",
    "datetime",
    "p2_str",
    "video_path_endzone",
    "video_path_sideline",
    "video_path_all29",
]


def get_feature_cols(df):
    """
    Identifies feature columns by excluding metadata columns.
    """
    return [c for c in df.columns if c not in METADATA_COLS]


class LGBMHandler:
    def __init__(self, model_name="lgbm_model.joblib"):
        self.model = None
        self.feature_cols = None
        self.model_name = model_name

    def fit(self, df_train, df_val):
        """
        Trains the LightGBM model with early stopping.
        """
        # Identify features
        self.feature_cols = get_feature_cols(df_train)

        X_train = df_train[self.feature_cols]
        y_train = df_train["contact"]
        X_val = df_val[self.feature_cols]
        y_val = df_val["contact"]

        # Prepare Datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Prepare Params (Handle n_estimators mapping)
        params = config.LGBM_PARAMS.copy()
        num_boost_round = params.pop("n_estimators", 1000)

        print(f"Training LightGBM with {len(self.feature_cols)} features...")

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ]

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Log Metrics
        best_iter = self.model.best_iteration
        train_preds = self.model.predict(X_train, num_iteration=best_iter)
        val_preds = self.model.predict(X_val, num_iteration=best_iter)

        train_auc = roc_auc_score(y_train, train_preds)
        val_auc = roc_auc_score(y_val, val_preds)

        print(f"LightGBM Training Completed. Best Iteration: {best_iter}")
        print(f"Train AUC: {train_auc}")
        print(f"Valid AUC: {val_auc}")

        # Save model and feature list
        utils.save_model(self.model, self.model_name)
        utils.save_model(self.feature_cols, f"{self.model_name}_features.joblib")

    def predict_proba(self, df):
        """
        Generates probability predictions.
        """
        if self.model is None:
            try:
                self.model = utils.load_model(self.model_name)
                self.feature_cols = utils.load_model(
                    f"{self.model_name}_features.joblib"
                )
            except FileNotFoundError:
                raise Exception("LGBM Model not trained or found.")

        X = df[self.feature_cols]
        iteration = (
            self.model.best_iteration if hasattr(self.model, "best_iteration") else 0
        )

        # If iteration is 0 or None (depending on lgb version/save method), default to all
        if iteration and iteration > 0:
            return self.model.predict(X, num_iteration=iteration)
        return self.model.predict(X)


class XGBHandler:
    def __init__(self, model_name="xgb_model.joblib"):
        self.model = None
        self.feature_cols = None
        self.model_name = model_name

    def fit(self, df_train, df_val):
        """
        Trains the XGBoost model with early stopping.
        """
        self.feature_cols = get_feature_cols(df_train)

        X_train = df_train[self.feature_cols]
        y_train = df_train["contact"]
        X_val = df_val[self.feature_cols]
        y_val = df_val["contact"]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Prepare Params
        params = config.XGB_PARAMS.copy()
        num_boost_round = params.pop("n_estimators", 1000)

        # Map sklearn params to native params if necessary
        if "n_jobs" in params:
            params["nthread"] = params.pop("n_jobs")

        # Remove params not used by train (like enable_categorical if using DMatrix might be fine, but safer to keep standard)
        # XGBoost 3.x is robust, but let's ensure clean dict

        print(f"Training XGBoost with {len(self.feature_cols)} features...")

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dval, "valid")],
            early_stopping_rounds=50,
            verbose_eval=100,
        )

        # Log Metrics
        # predict in xgb native uses iteration_range=(start, end)
        limit = self.model.best_iteration + 1
        train_preds = self.model.predict(dtrain, iteration_range=(0, limit))
        val_preds = self.model.predict(dval, iteration_range=(0, limit))

        train_auc = roc_auc_score(y_train, train_preds)
        val_auc = roc_auc_score(y_val, val_preds)

        print(
            f"XGBoost Training Completed. Best Iteration: {self.model.best_iteration}"
        )
        print(f"Train AUC: {train_auc}")
        print(f"Valid AUC: {val_auc}")

        utils.save_model(self.model, self.model_name)
        utils.save_model(self.feature_cols, f"{self.model_name}_features.joblib")

    def predict_proba(self, df):
        """
        Generates probability predictions.
        """
        if self.model is None:
            try:
                self.model = utils.load_model(self.model_name)
                self.feature_cols = utils.load_model(
                    f"{self.model_name}_features.joblib"
                )
            except FileNotFoundError:
                raise Exception("XGB Model not trained or found.")

        X = df[self.feature_cols]
        dtest = xgb.DMatrix(X)

        try:
            limit = self.model.best_iteration + 1
            return self.model.predict(dtest, iteration_range=(0, limit))
        except (AttributeError, TypeError):
            # Fallback if best_iteration is not available/applicable
            return self.model.predict(dtest)


class EnsemblePredictor:
    def __init__(self, lgbm_handler, xgb_handler):
        self.lgbm = lgbm_handler
        self.xgb = xgb_handler

    def predict_proba(self, df):
        """
        Aggregates predictions from both models via unweighted averaging.
        """
        p_lgbm = self.lgbm.predict_proba(df)
        p_xgb = self.xgb.predict_proba(df)
        return (p_lgbm + p_xgb) / 2.0

    def predict(self, df, threshold=0.5):
        """
        Returns binary predictions based on threshold.
        """
        probs = self.predict_proba(df)
        return (probs >= threshold).astype(int)
