import os
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import matthews_corrcoef, log_loss

from library.config import Config
from library.utils import Timer


class BaseModel:
    """
    Abstract base class for all models in the ensemble.
    Enforces a consistent interface for training, prediction, and persistence.
    """

    def __init__(self, name, params):
        self.name = name
        self.params = params.copy()
        self.model = None
        self.features = Config.FEATURES

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def save(self, output_dir):
        raise NotImplementedError

    def load(self, input_dir):
        raise NotImplementedError

    def _filter_features(self, X):
        """Ensures only the defined features are used."""
        if isinstance(X, pd.DataFrame):
            # Check if all features exist
            missing = [f for f in self.features if f not in X.columns]
            if missing:
                raise ValueError(f"Missing features in input data: {missing}")
            return X[self.features]
        return X


class LGBMModel(BaseModel):
    """
    LightGBM wrapper implementing the Leaf-wise growth strategy.
    Supports soft labels via binary logloss.
    """

    def __init__(self):
        super().__init__("lgbm", Config.LGBM_PARAMS)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train_filt = self._filter_features(X_train)

        callbacks = []
        eval_set = None

        if X_val is not None and y_val is not None:
            X_val_filt = self._filter_features(X_val)
            eval_set = [(X_val_filt, y_val)]
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=False
                )
            )
            callbacks.append(lgb.log_evaluation(period=100))

        with Timer(f"Train {self.name}"):
            self.model = lgb.LGBMClassifier(**self.params)
            self.model.fit(
                X_train_filt,
                y_train,
                eval_set=eval_set,
                eval_metric="logloss",
                callbacks=callbacks,
            )

            if eval_set:
                # Log validation score
                val_preds = self.model.predict_proba(X_val_filt)[:, 1]
                loss = log_loss(y_val, val_preds)
                print(
                    f"[{self.name}] Best Iteration: {self.model.best_iteration_}, Val LogLoss: {loss:.6f}"
                )

    def predict_proba(self, X):
        X_filt = self._filter_features(X)
        return self.model.predict_proba(X_filt)[:, 1]

    def save(self, output_dir):
        path = os.path.join(output_dir, f"{self.name}_model.joblib")
        joblib.dump(self.model, path)
        print(f"[{self.name}] Saved to {path}")

    def load(self, input_dir):
        path = os.path.join(input_dir, f"{self.name}_model.joblib")
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"[{self.name}] Loaded from {path}")
        else:
            print(f"[{self.name}] No model found at {path}")


class XGBModel(BaseModel):
    """
    XGBoost wrapper implementing the Level-wise growth strategy.
    """

    def __init__(self):
        super().__init__("xgb", Config.XGB_PARAMS)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train_filt = self._filter_features(X_train)

        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_filt = self._filter_features(X_val)
            eval_set = [(X_val_filt, y_val)]

        with Timer(f"Train {self.name}"):
            self.model = xgb.XGBClassifier(**self.params)
            # XGBoost 3.0+ supports early_stopping_rounds in fit
            self.model.fit(
                X_train_filt,
                y_train,
                eval_set=eval_set,
                verbose=False,
                early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            )

            if eval_set:
                val_preds = self.model.predict_proba(X_val_filt)[:, 1]
                loss = log_loss(y_val, val_preds)
                # best_iteration is available in the booster
                print(
                    f"[{self.name}] Best Iteration: {self.model.best_iteration}, Val LogLoss: {loss:.6f}"
                )

    def predict_proba(self, X):
        X_filt = self._filter_features(X)
        return self.model.predict_proba(X_filt)[:, 1]

    def save(self, output_dir):
        path = os.path.join(output_dir, f"{self.name}_model.joblib")
        joblib.dump(self.model, path)
        print(f"[{self.name}] Saved to {path}")

    def load(self, input_dir):
        path = os.path.join(input_dir, f"{self.name}_model.joblib")
        if os.path.exists(path):
            self.model = joblib.load(path)
            print(f"[{self.name}] Loaded from {path}")
        else:
            print(f"[{self.name}] No model found at {path}")


class EnsemblePredictor:
    """
    Orchestrates the Unified Heterogeneous Tri-Ensemble (LGBM, XGB, CatBoost).
    Manages training, prediction averaging, and persistence.
    """

    def __init__(self):
        self.models = [LGBMModel(), XGBModel()]

    def train(self, X_train, y_train, X_val, y_val):
        """
        Trains all models in the ensemble sequentially.
        """
        print(f"Starting Ensemble Training on {len(X_train)} samples...")
        for model in self.models:
            model.fit(X_train, y_train, X_val, y_val)

    def predict_proba(self, X):
        """
        Returns the unweighted average of probabilities from all models.
        """
        preds = []
        for model in self.models:
            p = model.predict_proba(X)
            preds.append(p)

        # Average predictions
        avg_preds = np.mean(preds, axis=0)
        return avg_preds

    def predict(self, X, threshold=0.5):
        """
        Returns binary class predictions based on threshold.
        """
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def save_models(self, output_dir=Config.WORKING_DIR):
        """
        Saves all trained models to the specified directory.
        """
        os.makedirs(output_dir, exist_ok=True)
        for model in self.models:
            model.save(output_dir)

    def load_models(self, input_dir=Config.WORKING_DIR):
        """
        Loads all models from the specified directory.
        """
        for model in self.models:
            model.load(input_dir)

    def evaluate(self, X_val, y_val):
        """
        Evaluates the ensemble on validation data and prints metrics.
        Returns the optimal threshold for MCC.
        """
        probs = self.predict_proba(X_val)

        # Calculate LogLoss
        loss = log_loss(y_val, probs)
        print(f"Ensemble Validation LogLoss: {loss:.10f}")

        # Find best threshold for MCC
        thresholds = np.linspace(0.1, 0.9, 81)
        best_mcc = -1
        best_thresh = 0.5

        # Convert y_val to binary if it's soft labels for evaluation
        y_true = (y_val >= 0.5).astype(int) if y_val.dtype == float else y_val

        for t in thresholds:
            preds = (probs >= t).astype(int)
            mcc = matthews_corrcoef(y_true, preds)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thresh = t

        print(f"Ensemble Best MCC: {best_mcc:.10f} at Threshold: {best_thresh:.4f}")
        return best_thresh
