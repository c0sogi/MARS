import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from library.config import LGBM_PARAMS, XGB_PARAMS


class LGBMWrapper:
    def __init__(self, params=None):
        self.params = params.copy() if params else LGBM_PARAMS.copy()

        # Extract early_stopping_rounds to use in callbacks
        self.early_stopping_rounds = self.params.pop("early_stopping_rounds", 50)

        # Ensure verbose is set to -1 to suppress internal logging
        self.params["verbose"] = -1

        self.model = None
        self.le = None

    def fit(self, X_train, y_train, X_val, y_val):
        # Encode targets to 0-indexed integers
        self.le = LabelEncoder()
        y_train_enc = self.le.fit_transform(y_train)
        y_val_enc = self.le.transform(y_val)

        # Initialize classifier
        self.model = lgb.LGBMClassifier(**self.params)

        # Define callbacks for early stopping and logging suppression
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=self.early_stopping_rounds, verbose=False
            ),
            lgb.log_evaluation(period=0),
        ]

        # Train model
        self.model.fit(
            X_train,
            y_train_enc,
            eval_set=[(X_val, y_val_enc)],
            eval_metric="multi_logloss",
            callbacks=callbacks,
        )

        # Calculate and print validation metric
        val_preds = self.model.predict_proba(X_val)
        score = log_loss(y_val_enc, val_preds)
        print(f"LGBM Validation LogLoss: {score}")

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class XGBWrapper:
    def __init__(self, params=None):
        self.params = params.copy() if params else XGB_PARAMS.copy()

        # Fix parameters for XGBoost 3.0+ compatibility
        # Map 'gpu_id' to 'device'
        if "gpu_id" in self.params:
            gpu_id = self.params.pop("gpu_id")
            if "device" not in self.params:
                self.params["device"] = f"cuda:{gpu_id}"

        # Map 'gpu_hist' to 'hist' (modern GPU tree method)
        if self.params.get("tree_method") == "gpu_hist":
            self.params["tree_method"] = "hist"
            # Ensure device is set to cuda if not already
            if "device" not in self.params:
                self.params["device"] = "cuda"

        self.model = None
        self.le = None

    def fit(self, X_train, y_train, X_val, y_val):
        # Encode targets to 0-indexed integers
        self.le = LabelEncoder()
        y_train_enc = self.le.fit_transform(y_train)
        y_val_enc = self.le.transform(y_val)

        # Initialize classifier
        # Note: early_stopping_rounds is passed in params for modern XGBoost
        self.model = xgb.XGBClassifier(**self.params)

        # Train model
        self.model.fit(
            X_train, y_train_enc, eval_set=[(X_val, y_val_enc)], verbose=False
        )

        # Calculate and print validation metric
        val_preds = self.model.predict_proba(X_val)
        score = log_loss(y_val_enc, val_preds)
        print(f"XGB Validation LogLoss: {score}")

    def predict_proba(self, X):
        return self.model.predict_proba(X)
