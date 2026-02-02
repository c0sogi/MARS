import os
import joblib
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

from library.config import LGBM_PARAMS, XGB_PARAMS, CAT_PARAMS, MODEL_DIR, SEED
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
        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=100),
        ]

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

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params.get("n_estimators", 3000),
            evals=evals,
            early_stopping_rounds=100,
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


class CatBoostExpert:
    def __init__(self, params=None):
        self.params = params if params else CAT_PARAMS.copy()
        self.model = None
        self.model_path = os.path.join(MODEL_DIR, "cat_expert.joblib")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the CatBoost model.
        """
        logger.info("Training CatBoost Expert...")

        train_pool = Pool(X_train, y_train)
        val_pool = None
        if X_val is not None and y_val is not None:
            val_pool = Pool(X_val, y_val)

        # Initialize classifier wrapper
        self.model = CatBoostClassifier(**self.params)

        self.model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=100,
            verbose=100,
            use_best_model=True,
        )

        logger.info(f"CatBoost Best Iteration: {self.model.get_best_iteration()}")

    def predict(self, X):
        """
        Predicts probabilities.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded.")

        # predict_proba returns (N, 2), we want the probability of class 1
        return self.model.predict_proba(X)[:, 1]

    def save(self):
        # CatBoost has its own save method, but joblib is convenient for wrappers
        # However, CatBoost objects are picklable.
        joblib.dump(self.model, self.model_path)
        logger.info(f"CatBoost model saved to {self.model_path}")

    def load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            logger.info(f"CatBoost model loaded from {self.model_path}")
            return True
        return False


class EnsemblePredictor:
    """
    Aggregates predictions from the Tri-Ensemble (LGBM, XGB, CatBoost).
    """

    def __init__(self, models):
        """
        Args:
            models (list): List of trained model instances (LGBMExpert, XGBExpert, CatBoostExpert).
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
