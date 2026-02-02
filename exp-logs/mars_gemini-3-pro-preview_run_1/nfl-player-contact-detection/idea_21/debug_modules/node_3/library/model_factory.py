import os
import joblib
import numpy as np
import lightgbm as lgb
import xgboost as xgb

from library.config import LGBM_PARAMS, XGB_PARAMS, MODEL_DIR, SEED
from library.utils import get_logger

logger = get_logger("model_factory")


class LGBMExpert:
    def __init__(self, params=None):
        self.params = params if params else LGBM_PARAMS.copy()
        self.model = None
        self.model_path = os.path.join(MODEL_DIR, "lgbm_expert.joblib")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the LightGBM model.
        """
        logger.info("Training LightGBM Expert...")

        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = []
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets = [val_data]

        # Callbacks for early stopping and logging
        callbacks = [lgb.log_evaluation(period=100)]
        if valid_sets:
            callbacks.append(lgb.early_stopping(stopping_rounds=100, verbose=False))

        self.model = lgb.train(
            self.params, train_data, valid_sets=valid_sets, callbacks=callbacks
        )

        # Log best iteration info
        if self.model.best_iteration:
            logger.info(f"LGBM Best Iteration: {self.model.best_iteration}")

    def predict(self, X):
        """
        Predicts probabilities.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")
        return self.model.predict(X, num_iteration=self.model.best_iteration)

    def save(self):
        joblib.dump(self.model, self.model_path)
        logger.info(f"LGBM model saved to {self.model_path}")

    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            logger.info(f"LGBM model loaded from {self.model_path}")
            return True
        return False


class XGBExpert:
    def __init__(self, params=None):
        self.params = params if params else XGB_PARAMS.copy()
        self.model = None
        self.model_path = os.path.join(MODEL_DIR, "xgb_expert.joblib")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the XGBoost model.
        """
        logger.info("Training XGBoost Expert...")

        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = []

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dval, "validation")]

        early_stopping = 100 if evals else None
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params.get("n_estimators", 3000),
            evals=evals,
            early_stopping_rounds=early_stopping,
            verbose_eval=100,
        )

        if hasattr(self.model, "best_iteration"):
            logger.info(f"XGB Best Iteration: {self.model.best_iteration}")

    def predict(self, X):
        """
        Predicts probabilities.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")

        # XGBoost requires DMatrix for prediction if trained with train()
        dtest = xgb.DMatrix(X)
        return self.model.predict(
            dtest, iteration_range=(0, self.model.best_iteration + 1)
        )

    def save(self):
        joblib.dump(self.model, self.model_path)
        logger.info(f"XGB model saved to {self.model_path}")

    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            logger.info(f"XGB model loaded from {self.model_path}")
            return True
        return False


class EnsemblePredictor:
    """
    Aggregates predictions from the Ensemble (LGBM, XGB).
    """

    def __init__(self, models):
        """
        Args:
            models (list): List of trained model instances (LGBMExpert, XGBExpert).
        """
        self.models = models

    def predict(self, X):
        """
        Computes the unweighted average of probabilities from all models.

        Args:
            X (pd.DataFrame or np.array): Feature matrix.

        Returns:
            np.array: Averaged probabilities.
        """
        if not self.models:
            raise ValueError("No models provided to EnsemblePredictor.")

        preds_sum = None

        for i, model in enumerate(self.models):
            p = model.predict(X)
            if preds_sum is None:
                preds_sum = p
            else:
                preds_sum += p

        return preds_sum / len(self.models)
