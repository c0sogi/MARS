import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, log_loss
from library.config import Config


class TriEnsemble:
    """
    Unified Heterogeneous Tri-Ensemble Manager.
    Manages LightGBM, XGBoost, and HistGradientBoosting models.
    """

    def __init__(self):
        self.models = {}
        self._init_models()

    def _init_models(self):
        """Initialize the three expert models with config parameters."""
        # 1. LightGBM
        self.lgbm = lgb.LGBMClassifier(**Config.LGBM_PARAMS)

        # 2. XGBoost
        self.xgb = xgb.XGBClassifier(**Config.XGB_PARAMS)

        # 3. HistGradientBoosting (CatBoost Proxy)
        self.hgb = HistGradientBoostingClassifier(**Config.HGB_PARAMS)

        self.models = {"lgbm": self.lgbm, "xgb": self.xgb, "hgb": self.hgb}

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains all three models with early stopping where supported.

        Args:
            X_train, y_train: Training features and targets (targets may be soft labels).
            X_val, y_val: Validation features and binary targets.
        """
        print("Starting Tri-Ensemble Training...")

        # --- 1. Train LightGBM ---
        print("\nTraining LightGBM Expert...")
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),  # Suppress internal logging
        ]

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]

        self.lgbm.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_metric="logloss",
            callbacks=callbacks,
        )
        self._print_metrics("LightGBM", self.lgbm, X_val, y_val)

        # --- 2. Train XGBoost ---
        print("\nTraining XGBoost Expert...")
        # XGBoost handles soft labels natively in binary:logistic
        self.xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        self._print_metrics("XGBoost", self.xgb, X_val, y_val)

        # --- 3. Train HistGradientBoosting ---
        print("\nTraining HistGradientBoosting Expert...")
        # HGB Classifier requires discrete classes. If y_train is soft (float), binarize it.
        # We assume threshold of 0.5 for binarization if needed.
        y_train_hgb = y_train
        if y_train.dtype == float:
            # Check if values are strictly 0.0 or 1.0, or soft
            is_soft = np.any((y_train > 0) & (y_train < 1))
            if is_soft:
                y_train_hgb = (y_train > 0.5).astype(int)

        # HGB uses internal validation for early stopping if early_stopping=True in params
        self.hgb.fit(X_train, y_train_hgb)
        self._print_metrics("HistGradientBoosting", self.hgb, X_val, y_val)

    def predict_proba(self, X):
        """
        Returns the averaged probability predictions from the ensemble.
        """
        p_lgbm = self.lgbm.predict_proba(X)[:, 1]
        p_xgb = self.xgb.predict_proba(X)[:, 1]
        p_hgb = self.hgb.predict_proba(X)[:, 1]

        # Unweighted Average
        avg_pred = (p_lgbm + p_xgb + p_hgb) / 3.0
        return avg_pred

    def predict(self, X, threshold=0.5):
        """
        Returns binary predictions based on the averaged probability.
        """
        probs = self.predict_proba(X)
        return (probs > threshold).astype(int)

    def save_models(self, suffix=""):
        """
        Saves the trained models to the configured model directory.
        """
        suffix_str = f"_{suffix}" if suffix else ""

        path_lgbm = os.path.join(Config.MODEL_DIR, f"expert_lgbm{suffix_str}.joblib")
        path_xgb = os.path.join(Config.MODEL_DIR, f"expert_xgb{suffix_str}.joblib")
        path_hgb = os.path.join(Config.MODEL_DIR, f"expert_hgb{suffix_str}.joblib")

        joblib.dump(self.lgbm, path_lgbm)
        joblib.dump(self.xgb, path_xgb)
        joblib.dump(self.hgb, path_hgb)

        print(f"Models saved with suffix '{suffix}' to {Config.MODEL_DIR}")

    def load_models(self, suffix=""):
        """
        Loads models from the configured model directory.
        """
        suffix_str = f"_{suffix}" if suffix else ""

        path_lgbm = os.path.join(Config.MODEL_DIR, f"expert_lgbm{suffix_str}.joblib")
        path_xgb = os.path.join(Config.MODEL_DIR, f"expert_xgb{suffix_str}.joblib")
        path_hgb = os.path.join(Config.MODEL_DIR, f"expert_hgb{suffix_str}.joblib")

        if os.path.exists(path_lgbm):
            self.lgbm = joblib.load(path_lgbm)
        if os.path.exists(path_xgb):
            self.xgb = joblib.load(path_xgb)
        if os.path.exists(path_hgb):
            self.hgb = joblib.load(path_hgb)

        print(f"Models loaded with suffix '{suffix}'")

    def _print_metrics(self, name, model, X_val, y_val):
        """Helper to print full precision metrics."""
        if X_val is None or y_val is None:
            return

        # Get probabilities
        probs = model.predict_proba(X_val)[:, 1]

        # Calculate Log Loss
        ll = log_loss(y_val, probs)

        # Calculate MCC (using default 0.5 threshold for reporting)
        preds = (probs > 0.5).astype(int)
        mcc = matthews_corrcoef(y_val, preds)

        print(f"{name} - Validation LogLoss: {ll}")
        print(f"{name} - Validation MCC: {mcc}")
