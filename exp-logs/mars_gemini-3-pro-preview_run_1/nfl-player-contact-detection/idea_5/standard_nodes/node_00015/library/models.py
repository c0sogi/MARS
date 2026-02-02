import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import joblib
from sklearn.metrics import matthews_corrcoef, log_loss
from library.config import Config


class LightGBMModel:
    def __init__(self):
        self.params = Config.LGBM_PARAMS.copy()
        self.model = None
        self.n_estimators = Config.N_ESTIMATORS
        self.early_stopping_rounds = Config.EARLY_STOPPING_ROUNDS
        self.verbose_eval = Config.VERBOSE_EVAL

    def train(self, X_train, y_train, X_val, y_val):
        print("Training LightGBM Model...")

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        callbacks = [
            lgb.early_stopping(stopping_rounds=self.early_stopping_rounds),
            lgb.log_evaluation(period=self.verbose_eval),
        ]

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

    def predict(self, X):
        if self.model is None:
            raise ValueError("LightGBM model has not been trained yet.")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)


class XGBoostModel:
    def __init__(self):
        self.params = Config.XGB_PARAMS.copy()
        self.model = None
        self.n_estimators = Config.N_ESTIMATORS
        self.early_stopping_rounds = Config.EARLY_STOPPING_ROUNDS
        self.verbose_eval = Config.VERBOSE_EVAL

    def train(self, X_train, y_train, X_val, y_val):
        print("Training XGBoost Model...")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        evals = [(dtrain, "train"), (dval, "valid")]

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.verbose_eval,
        )

    def predict(self, X):
        if self.model is None:
            raise ValueError("XGBoost model has not been trained yet.")
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)


class UnifiedEnsemble:
    def __init__(self):
        self.lgbm = LightGBMModel()
        self.xgb = XGBoostModel()
        self.feature_cols = None

    def _get_feature_columns(self, df):
        """
        Excludes metadata columns to return only feature columns.
        """
        exclude_cols = [
            "contact_id",
            "game_play",
            "step",
            "nfl_player_id_1",
            "nfl_player_id_2",
            "contact",
            "datetime",
        ]
        return [c for c in df.columns if c not in exclude_cols]

    def train(self, train_df, val_df, target_col="contact"):
        """
        Trains both LightGBM and XGBoost models on the provided data.
        """
        # Identify feature columns
        self.feature_cols = self._get_feature_columns(train_df)
        print(f"Training with {len(self.feature_cols)} features.")

        X_train = train_df[self.feature_cols]
        y_train = train_df[target_col]
        X_val = val_df[self.feature_cols]
        y_val = val_df[target_col]

        # Train LightGBM
        self.lgbm.train(X_train, y_train, X_val, y_val)

        # Evaluate LightGBM
        preds_lgbm = self.lgbm.predict(X_val)
        loss_lgbm = log_loss(y_val, preds_lgbm)
        mcc_lgbm = matthews_corrcoef(y_val, (preds_lgbm > 0.5).astype(int))
        print(f"LightGBM Validation - LogLoss: {loss_lgbm}, MCC: {mcc_lgbm}")

        # Train XGBoost
        self.xgb.train(X_train, y_train, X_val, y_val)

        # Evaluate XGBoost
        preds_xgb = self.xgb.predict(X_val)
        loss_xgb = log_loss(y_val, preds_xgb)
        mcc_xgb = matthews_corrcoef(y_val, (preds_xgb > 0.5).astype(int))
        print(f"XGBoost Validation - LogLoss: {loss_xgb}, MCC: {mcc_xgb}")

        # Evaluate Ensemble
        preds_ensemble = (preds_lgbm + preds_xgb) / 2.0
        loss_ensemble = log_loss(y_val, preds_ensemble)
        mcc_ensemble = matthews_corrcoef(y_val, (preds_ensemble > 0.5).astype(int))
        print(f"Ensemble Validation - LogLoss: {loss_ensemble}, MCC: {mcc_ensemble}")

    def predict_proba(self, test_df):
        """
        Generates averaged probabilities from the ensemble.
        """
        if self.feature_cols is None:
            # If loaded from disk, we might not have feature_cols explicitly set in this instance
            # We infer from the dataframe but ensure strict matching is usually handled by the user pipeline
            self.feature_cols = self._get_feature_columns(test_df)

        # Ensure columns match training
        X_test = test_df[self.feature_cols]

        p_lgbm = self.lgbm.predict(X_test)
        p_xgb = self.xgb.predict(X_test)

        return (p_lgbm + p_xgb) / 2.0

    def save(self, directory):
        """
        Saves both models to the specified directory.
        """
        os.makedirs(directory, exist_ok=True)
        self.lgbm.save(os.path.join(directory, "lgbm_model.joblib"))
        self.xgb.save(os.path.join(directory, "xgb_model.joblib"))

        # Save feature columns to ensure consistency during inference
        joblib.dump(self.feature_cols, os.path.join(directory, "feature_cols.joblib"))
        print(f"Models saved to {directory}")

    def load(self, directory):
        """
        Loads models from the specified directory.
        """
        self.lgbm.load(os.path.join(directory, "lgbm_model.joblib"))
        self.xgb.load(os.path.join(directory, "xgb_model.joblib"))

        feat_path = os.path.join(directory, "feature_cols.joblib")
        if os.path.exists(feat_path):
            self.feature_cols = joblib.load(feat_path)
        print(f"Models loaded from {directory}")
