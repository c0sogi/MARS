import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from library.config import (
    LGBM_PARAMS,
    XGB_PARAMS,
    HGB_PARAMS,
    RIDGE_PARAMS,
    EARLY_STOPPING_ROUNDS,
    VERBOSE_EVAL,
    SEED,
)


class LGBMRegressorWrapper:
    """
    Wrapper for LightGBM Regressor with Early Stopping support.
    """

    def __init__(self):
        self.params = LGBM_PARAMS.copy()
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Trains the LightGBM model.
        """
        train_set = lgb.Dataset(X, label=y)
        valid_sets = []

        if X_val is not None and y_val is not None:
            valid_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
            valid_sets.append(valid_set)

        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=VERBOSE_EVAL),
        ]

        self.model = lgb.train(
            self.params,
            train_set,
            valid_sets=valid_sets,
            valid_names=["valid"],
            callbacks=callbacks,
        )

        # Manually print final metric if validation set was present
        if X_val is not None and y_val is not None:
            preds = self.predict(X_val)
            score = mean_absolute_error(y_val, preds)
            print(f"[LGBM] Best Validation MAE: {score}")

    def predict(self, X):
        return self.model.predict(X, num_iteration=self.model.best_iteration)


class XGBRegressorWrapper:
    """
    Wrapper for XGBoost Regressor with Early Stopping support.
    """

    def __init__(self):
        self.params = XGB_PARAMS.copy()
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Trains the XGBoost model.
        """
        dtrain = xgb.DMatrix(X, label=y)
        evals = []

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "valid"))

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.params.get("n_estimators", 5000),
            evals=evals,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=VERBOSE_EVAL,
        )

        if X_val is not None and y_val is not None:
            # XGBoost handles logging internally, but we ensure the best score is noted
            if hasattr(self.model, "best_score"):
                print(f"[XGB] Best Validation MAE: {self.model.best_score}")

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(
            dtest, iteration_range=(0, self.model.best_iteration + 1)
        )


class HGBRegressorWrapper:
    """
    Wrapper for Scikit-Learn HistGradientBoostingRegressor.
    Acts as a proxy for CatBoost in this ensemble.
    """

    def __init__(self):
        self.params = HGB_PARAMS.copy()
        self.model = HistGradientBoostingRegressor(**self.params)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Trains the HistGradientBoosting model.
        Note: Sklearn HGB does not support external validation sets for early stopping
        in the same way as XGB/LGBM (it uses internal splitting).
        We fit on the provided training data.
        """
        self.model.fit(X, y)

        if X_val is not None and y_val is not None:
            preds = self.predict(X_val)
            score = mean_absolute_error(y_val, preds)
            print(f"[HGB] Validation MAE: {score}")

    def predict(self, X):
        return self.model.predict(X)


class RidgeMetaLearnerWrapper:
    """
    Wrapper for Ridge Regression used as the Level 1 Meta Learner.
    """

    def __init__(self):
        self.params = RIDGE_PARAMS.copy()
        self.model = Ridge(**self.params)

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Trains the Ridge Meta Learner.
        """
        self.model.fit(X, y)

        if X_val is not None and y_val is not None:
            preds = self.predict(X_val)
            score = mean_absolute_error(y_val, preds)
            print(f"[Meta-Ridge] Validation MAE: {score}")

    def predict(self, X):
        return self.model.predict(X)
