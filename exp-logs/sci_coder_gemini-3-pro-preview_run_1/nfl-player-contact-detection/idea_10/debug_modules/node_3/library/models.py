import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import joblib
from library.config import Config
from library.utils import setup_logger, calc_mcc

logger = setup_logger()


class ModelWrapper:
    """
    Abstract base class for model wrappers.
    Enforces a consistent interface for training, saving, loading, and predicting.
    """

    def __init__(self, name):
        self.name = name
        self.model = None
        self.model_path = os.path.join(Config.MODEL_DIR, f"{self.name}.joblib")

    def fit(self, X_train, y_train, X_val, y_val):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def save(self):
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        else:
            logger.warning("No model to save.")

    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
            return True
        return False


class LGBMWrapper(ModelWrapper):
    """
    Wrapper for LightGBM training and inference.
    Uses params from Config.LGBM_PARAMS.
    """

    def __init__(self, name="lgbm"):
        super().__init__(name)
        self.params = Config.LGBM_PARAMS.copy()

    def fit(self, X_train, y_train, X_val, y_val):
        logger.info(f"[{self.name}] Training LightGBM model...")

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        # Callbacks for early stopping and logging
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=False
            ),
            lgb.log_evaluation(period=Config.VERBOSE_EVAL),
        ]

        self.model = lgb.train(
            self.params,
            dtrain,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )

        # Log validation score
        # Best iteration is handled automatically by prediction if using model.predict
        logger.info(
            f"[{self.name}] Training finished. Best iteration: {self.model.best_iteration}"
        )

    def predict(self, X):
        if self.model is None:
            if not self.load():
                raise ValueError(f"[{self.name}] Model not trained or loaded.")

        # LightGBM predict returns raw probabilities for binary classification
        return self.model.predict(X, num_iteration=self.model.best_iteration)


class XGBWrapper(ModelWrapper):
    """
    Wrapper for XGBoost training and inference.
    Uses params from Config.XGB_PARAMS.
    Calculates scale_pos_weight dynamically to handle imbalance.
    """

    def __init__(self, name="xgb"):
        super().__init__(name)
        self.params = Config.XGB_PARAMS.copy()

    def fit(self, X_train, y_train, X_val, y_val):
        logger.info(f"[{self.name}] Training XGBoost model...")

        # Calculate scale_pos_weight for imbalance
        # ratio = neg / pos
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        ratio = n_neg / max(1, n_pos)
        self.params["scale_pos_weight"] = ratio
        logger.info(f"[{self.name}] Calculated scale_pos_weight: {ratio:.4f}")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params.get("n_estimators", 2000),
            evals=[(dtrain, "train"), (dval, "valid")],
            early_stopping_rounds=Config.EARLY_STOPPING_ROUNDS,
            verbose_eval=Config.VERBOSE_EVAL,
        )

        logger.info(
            f"[{self.name}] Training finished. Best iteration: {self.model.best_iteration}"
        )

    def predict(self, X):
        if self.model is None:
            if not self.load():
                raise ValueError(f"[{self.name}] Model not trained or loaded.")

        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)


class Ensemble:
    """
    Manages a collection of models and aggregates their predictions.
    Supports threshold optimization for MCC.
    """

    def __init__(self):
        self.models = []
        self.best_threshold = 0.5

    def add_model(self, model: ModelWrapper):
        self.models.append(model)

    def predict_proba(self, X):
        """
        Returns the averaged probability from all models.
        """
        if not self.models:
            raise ValueError("No models added to ensemble.")

        total_probs = None
        for model in self.models:
            probs = model.predict(X)
            if total_probs is None:
                total_probs = probs
            else:
                total_probs += probs

        avg_probs = total_probs / len(self.models)
        return avg_probs

    def predict(self, X, threshold=None):
        """
        Returns binary class predictions based on threshold.
        """
        if threshold is None:
            threshold = self.best_threshold

        probs = self.predict_proba(X)
        return (probs > threshold).astype(int)

    def optimize_threshold(self, X_val, y_val):
        """
        Finds the threshold that maximizes MCC on the validation set.
        """
        logger.info("Optimizing ensemble threshold...")
        probs = self.predict_proba(X_val)
        y_true = y_val.values if hasattr(y_val, "values") else y_val

        best_mcc = -1.0
        best_thresh = 0.5

        # Search space
        thresholds = np.arange(0.01, 1.00, 0.01)

        for thresh in thresholds:
            preds = (probs > thresh).astype(int)
            score = calc_mcc(y_true, preds)

            if score > best_mcc:
                best_mcc = score
                best_thresh = thresh

        self.best_threshold = best_thresh
        logger.info(f"Best Threshold: {best_thresh:.4f}")
        logger.info(f"Best Validation MCC: {best_mcc}")  # Full precision print

        # Save threshold
        thresh_path = os.path.join(Config.MODEL_DIR, "best_threshold.npy")
        np.save(thresh_path, np.array([best_thresh]))

        return best_thresh, best_mcc

    def load_threshold(self):
        thresh_path = os.path.join(Config.MODEL_DIR, "best_threshold.npy")
        if os.path.exists(thresh_path):
            self.best_threshold = float(np.load(thresh_path)[0])
            logger.info(f"Loaded best threshold: {self.best_threshold:.4f}")
        else:
            logger.warning("Threshold file not found, using default 0.5")
