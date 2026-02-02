import os
import joblib
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, log_loss
import library.config as config


class TriEnsemble:
    """
    A heterogeneous ensemble combining LightGBM, XGBoost, and HistGradientBoosting.
    Implements the VASM-E strategy's expert tier.
    """

    def __init__(self):
        # Initialize LightGBM
        self.lgbm = lgb.LGBMClassifier(**config.LGBM_PARAMS)

        # Initialize XGBoost
        self.xgb = xgb.XGBClassifier(**config.XGB_PARAMS)

        # Initialize HistGradientBoosting (Scikit-learn)
        # Replacing CatBoost to adhere to environment while maintaining structural diversity
        self.hgb = HistGradientBoostingClassifier(**config.HGB_PARAMS)

        self.models = {
            "LightGBM": self.lgbm,
            "XGBoost": self.xgb,
            "HistGradientBoosting": self.hgb,
        }

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains all three models in the ensemble.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features (optional, used for early stopping).
            y_val: Validation targets (optional).
        """
        print(f"Starting TriEnsemble training on {len(X_train)} samples...")

        # --- 1. Train LightGBM ---
        print("Training LightGBM Expert...")
        callbacks = []
        if X_val is not None and y_val is not None:
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=config.EARLY_STOPPING_ROUNDS, verbose=False
                )
            )
            eval_set = [(X_val, y_val)]
        else:
            eval_set = None

        self.lgbm.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_metric="average_precision",
            callbacks=callbacks,
        )

        if X_val is not None and y_val is not None:
            val_pred = self.lgbm.predict_proba(X_val)[:, 1]
            score = average_precision_score(y_val, val_pred)
            print(f"LightGBM Validation AP: {score:.8f}")

        # --- 2. Train XGBoost ---
        print("Training XGBoost Expert...")
        # XGBoost handles early stopping via fit params
        early_stopping_rounds = (
            config.EARLY_STOPPING_ROUNDS if (X_val is not None) else None
        )

        self.xgb.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)] if X_val is not None else None,
            verbose=False,
        )
        # Note: XGBoost with sklearn API and early stopping might not automatically use best iteration
        # for prediction unless strictly configured, but modern versions usually handle this.

        if X_val is not None and y_val is not None:
            val_pred = self.xgb.predict_proba(X_val)[:, 1]
            score = average_precision_score(y_val, val_pred)
            print(f"XGBoost Validation AP: {score:.8f}")

        # --- 3. Train HistGradientBoosting ---
        print("Training HistGradientBoosting Expert...")
        # HGB uses internal validation split for early stopping (configured in params)
        # We do not pass X_val explicitly to fit() as it doesn't support eval_set in the same way.
        self.hgb.fit(X_train, y_train)

        if X_val is not None and y_val is not None:
            val_pred = self.hgb.predict_proba(X_val)[:, 1]
            score = average_precision_score(y_val, val_pred)
            print(f"HistGradientBoosting Validation AP: {score:.8f}")

        print("TriEnsemble Training Complete.")

    def predict_proba(self, X):
        """
        Predicts class probabilities for X.
        Returns the unweighted average of the three models.
        """
        # Get probabilities for the positive class (index 1)
        p_lgbm = self.lgbm.predict_proba(X)[:, 1]
        p_xgb = self.xgb.predict_proba(X)[:, 1]
        p_hgb = self.hgb.predict_proba(X)[:, 1]

        # Ensemble averaging
        avg_proba = (p_lgbm + p_xgb + p_hgb) / 3.0
        return avg_proba

    def predict(self, X, threshold=0.5):
        """
        Predicts binary class labels based on threshold.
        """
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def save(self, filename="tri_ensemble.joblib"):
        """
        Saves the ensemble to disk.
        """
        path = os.path.join(config.MODEL_DIR, filename)
        print(f"Saving model to {path}...")
        joblib.dump(self, path)

    @staticmethod
    def load(filename="tri_ensemble.joblib"):
        """
        Loads the ensemble from disk.
        """
        path = os.path.join(config.MODEL_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found at {path}")
        print(f"Loading model from {path}...")
        return joblib.load(path)
