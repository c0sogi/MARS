import os
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import joblib
from library.config import Config


class LGBMWrapper:
    """
    Wrapper for LightGBM Classifier implementing the Expert Tier specifications.
    Uses leaf-wise growth and deep trees as defined in Config.
    """

    def __init__(self):
        self.params = Config.LGBM_PARAMS.copy()
        # Ensure seed is set
        self.params["random_state"] = Config.SEED
        self.model = lgb.LGBMClassifier(**self.params)
        self.model_name = "lgbm"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the LightGBM model with Early Stopping.
        """
        callbacks = []
        eval_set = None

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            # Add early stopping callback
            callbacks.append(
                lgb.early_stopping(
                    stopping_rounds=Config.EARLY_STOPPING_ROUNDS, verbose=False
                )
            )
            # Add logging callback to suppress verbose output but keep critical info if needed
            callbacks.append(lgb.log_evaluation(period=0))  # 0 means no logging

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            eval_metric=self.params.get("metric", "binary_logloss"),
            callbacks=callbacks,
        )

        # Log best score if validation was used
        if self.model.best_score_:
            # Retrieve the metric name from params or default
            metric_name = self.params.get("metric", "binary_logloss")
            # best_score_ is a dict: {'valid_0': {'binary_logloss': 0.123}}
            if "valid_0" in self.model.best_score_:
                val_score = self.model.best_score_["valid_0"].get(metric_name, 0.0)
                print(f"[LGBM] Best Validation Score ({metric_name}): {val_score}")

    def predict(self, X):
        """
        Returns probability of class 1 (Contact).
        """
        # predict_proba returns [prob_0, prob_1]
        return self.model.predict_proba(X)[:, 1]

    def save(self, filepath):
        """
        Saves the wrapper object using joblib.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    def load(self, filepath):
        """
        Loads the wrapper object from disk.
        """
        loaded = joblib.load(filepath)
        self.model = loaded.model
        self.params = loaded.params
        return self


class XGBWrapper:
    """
    Wrapper for XGBoost Classifier implementing the Expert Tier specifications.
    Uses level-wise growth and deep trees.
    """

    def __init__(self):
        self.params = Config.XGB_PARAMS.copy()
        # Ensure seed is set
        self.params["random_state"] = Config.SEED
        self.model = xgb.XGBClassifier(**self.params)
        self.model_name = "xgb"

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the XGBoost model with Early Stopping.
        """
        eval_set = None
        early_stopping_rounds = None

        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            early_stopping_rounds = Config.EARLY_STOPPING_ROUNDS

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            early_stopping_rounds=early_stopping_rounds,
            verbose=False,
        )

        if hasattr(self.model, "best_score"):
            print(f"[XGB] Best Validation Score: {self.model.best_score}")

    def predict(self, X):
        """
        Returns probability of class 1 (Contact).
        """
        return self.model.predict_proba(X)[:, 1]

    def save(self, filepath):
        """
        Saves the wrapper object using joblib.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    def load(self, filepath):
        """
        Loads the wrapper object from disk.
        """
        loaded = joblib.load(filepath)
        self.model = loaded.model
        self.params = loaded.params
        return self


class Ensemble:
    """
    Unified Heterogeneous Dual-Ensemble.
    Averages predictions from the provided list of models.
    """

    def __init__(self, models):
        """
        Args:
            models (list): List of instantiated and trained model wrapper objects
                           (e.g., [lgbm_wrapper, xgb_wrapper]).
        """
        self.models = models

    def predict(self, X):
        """
        Aggregates predictions from all models via unweighted averaging.

        Args:
            X (pd.DataFrame or np.array): Feature matrix.

        Returns:
            np.array: Averaged probabilities.
        """
        if not self.models:
            raise ValueError("No models provided to Ensemble.")

        preds = []
        for model in self.models:
            p = model.predict(X)
            preds.append(p)

        # Stack and average
        preds_stack = np.vstack(preds)
        avg_preds = np.mean(preds_stack, axis=0)

        return avg_preds
