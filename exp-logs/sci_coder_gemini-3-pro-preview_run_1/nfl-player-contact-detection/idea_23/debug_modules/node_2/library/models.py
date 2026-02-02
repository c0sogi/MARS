import os
import joblib
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from library.config import Config


class BaseModel:
    """
    Abstract base class for the Unified Heterogeneous Tri-Ensemble models.
    """

    def __init__(self, name):
        self.name = name
        self.model = None
        self.config = Config
        self.model_dir = self.config.MODEL_DIR
        os.makedirs(self.model_dir, exist_ok=True)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Trains the model.
        Args:
            X (pd.DataFrame or np.ndarray): Training features.
            y (pd.Series or np.ndarray): Training targets (can be soft labels).
            X_val (pd.DataFrame or np.ndarray): Validation features.
            y_val (pd.Series or np.ndarray): Validation targets.
        """
        raise NotImplementedError

    def predict(self, X):
        """
        Predicts probabilities.
        Args:
            X (pd.DataFrame or np.ndarray): Features.
        Returns:
            np.ndarray: Probability of contact (Class 1).
        """
        raise NotImplementedError

    def save(self, filename=None):
        """Saves the model to the configured model directory."""
        if filename is None:
            filename = f"{self.name}.joblib"
        path = os.path.join(self.model_dir, filename)
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")

    def load(self, filename=None):
        """Loads the model from the configured model directory."""
        if filename is None:
            filename = f"{self.name}.joblib"
        path = os.path.join(self.model_dir, filename)
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"Model loaded from {path}")
        else:
            print(f"Model file {path} not found.")


class LGBMWrapper(BaseModel):
    def __init__(self):
        super().__init__("lgbm_model")
        self.params = self.config.LGBM_PARAMS.copy()

    def fit(self, X, y, X_val=None, y_val=None):
        print(f"[{self.name}] Training LightGBM...")

        # Create dataset
        dtrain = lgb.Dataset(X, label=y)
        valid_sets = [dtrain]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            valid_sets.append(dval)
            valid_names.append("valid")

        # Callbacks for early stopping and logging
        callbacks = [
            lgb.early_stopping(stopping_rounds=self.config.EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=self.config.VERBOSE_EVAL),
        ]

        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=self.config.NUM_BOOST_ROUND,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        if X_val is not None:
            preds = self.predict(X_val)
            # Threshold y_val for metric calculation if it is soft
            y_val_binary = (y_val > 0.5).astype(int)
            score = average_precision_score(y_val_binary, preds)
            print(f"[{self.name}] Validation AP: {score:.16f}")

    def predict(self, X):
        return self.model.predict(X, num_iteration=self.model.best_iteration)


class XGBWrapper(BaseModel):
    def __init__(self):
        super().__init__("xgb_model")
        self.params = self.config.XGB_PARAMS.copy()

    def fit(self, X, y, X_val=None, y_val=None):
        print(f"[{self.name}] Training XGBoost...")

        dtrain = xgb.DMatrix(X, label=y)
        evals = [(dtrain, "train")]

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "valid"))

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.config.NUM_BOOST_ROUND,
            evals=evals,
            early_stopping_rounds=self.config.EARLY_STOPPING_ROUNDS,
            verbose_eval=self.config.VERBOSE_EVAL,
        )

        if X_val is not None:
            preds = self.predict(X_val)
            y_val_binary = (y_val > 0.5).astype(int)
            score = average_precision_score(y_val_binary, preds)
            print(f"[{self.name}] Validation AP: {score:.16f}")

    def predict(self, X):
        # XGBoost requires DMatrix for prediction when using the low-level API
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)


class HistGBWrapper(BaseModel):
    def __init__(self):
        super().__init__("histgb_model")
        self.params = self.config.HISTGB_PARAMS.copy()

    def fit(self, X, y, X_val=None, y_val=None):
        print(f"[{self.name}] Training HistGradientBoostingClassifier...")

        # Scikit-learn classifiers require discrete class labels.
        # If y contains soft labels (floats), we must binarize them.
        # We use 0.5 as the threshold.
        y_binary = (y > 0.5).astype(int)

        self.model = HistGradientBoostingClassifier(**self.params)

        # HistGradientBoostingClassifier uses internal validation set (validation_fraction)
        # for early stopping. We pass the training data directly.
        self.model.fit(X, y_binary)

        if X_val is not None:
            preds = self.predict(X_val)
            y_val_binary = (y_val > 0.5).astype(int)
            score = average_precision_score(y_val_binary, preds)
            print(f"[{self.name}] Validation AP: {score:.16f}")

    def predict(self, X):
        # predict_proba returns [prob_0, prob_1]
        return self.model.predict_proba(X)[:, 1]
